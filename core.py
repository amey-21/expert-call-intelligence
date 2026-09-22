"""
core.py - pure analysis logic, zero Streamlit imports.

This file is intentionally framework-agnostic: every function takes plain
data in (strings, dicts, lists) and returns plain data out. app.py is the
only place that knows Streamlit exists. If this ever needs to move behind
FastAPI for a larger transcript set, this file moves unchanged.
"""

import json
import os
import re

from openai import OpenAI

MODEL = "gpt-4o-mini"  # swap for whatever model you have access to


# ---------------------------------------------------------------------------
# Step 1: Parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_BLOCK_RE = re.compile(
    r"^(\d{1,2}:\d{2})\s*\n(.+?)(?=\n\d{1,2}:\d{2}\s*\n|\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_transcript(text: str, expert_num: int) -> dict:
    """
    Parse one transcript file into a normalized structure.

    Returns:
        {
            "expert_num": 1,
            "expert_name": "Dr. Jean Martin",
            "role": "Head of Urology",
            "market": "France",
            "turns": [
                {
                    "turn_id": "E01-T01",
                    "speaker": "Interviewer",
                    "is_expert": False,
                    "timestamp": "00:00",
                    "text": "Thanks for joining. ...",
                },
                ...
            ],
        }
    """
    lines = text.strip().splitlines()

    # --- header: first 3 non-empty lines ---
    header_lines = [l.strip() for l in lines[:3]]

    # Line 0: "Expert 1 – Dr. Jean Martin"  (en dash or hyphen)
    header_line = header_lines[0]
    parts = re.split(r"[–-]", header_line, maxsplit=1)
    expert_name = parts[1].strip() if len(parts) > 1 else header_line.strip()

    # Line 1: "Role: Head of Urology"
    role = header_lines[1].split(":", 1)[1].strip() if ":" in header_lines[1] else ""

    # Line 2: "Market: France"
    market = header_lines[2].split(":", 1)[1].strip() if ":" in header_lines[2] else ""

    # --- turns: everything after the header ---
    body = "\n".join(lines[3:])

    turns = []
    for i, match in enumerate(_TIMESTAMP_BLOCK_RE.finditer(body), start=1):
        timestamp = match.group(1).strip()
        block_text = match.group(2).strip()

        # First "Speaker: text" line, rest is continuation of the same turn.
        if ":" in block_text:
            speaker, spoken_text = block_text.split(":", 1)
            speaker = speaker.strip()
            spoken_text = spoken_text.strip()
        else:
            speaker = "Unknown"
            spoken_text = block_text.strip()

        is_expert = speaker.lower() != "interviewer"

        turns.append(
            {
                "turn_id": f"E{expert_num:02d}-T{i:02d}",
                "speaker": speaker,
                "is_expert": is_expert,
                "timestamp": timestamp,
                "text": spoken_text,
            }
        )

    return {
        "expert_num": expert_num,
        "expert_name": expert_name,
        "role": role,
        "market": market,
        "turns": turns,
    }


def parse_guide(text: str) -> list[str]:
    """
    Parse the interview guide into a flat list of question strings.
    Expects numbered lines like '1. How would you describe...'
    """
    questions = []
    for line in text.strip().splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            questions.append(m.group(1).strip())
    return questions


def expert_turns(parsed_transcript: dict) -> list[dict]:
    """Return only the turns eligible as evidence (excludes the interviewer)."""
    return [t for t in parsed_transcript["turns"] if t["is_expert"]]


def turn_lookup(parsed_transcript: dict) -> dict:
    """turn_id -> turn dict, for O(1) lookup when validating/rendering citations."""
    return {t["turn_id"]: t for t in parsed_transcript["turns"]}


# ---------------------------------------------------------------------------
# Step 3: Guide answers (LLM analysis)
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    """
    Reads OPENAI_API_KEY from the environment (app.py loads .env before
    calling this). Raises a clear error if it's missing rather than letting
    the SDK's generic error surface.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or your "
            "environment before running analysis."
        )
    return OpenAI(api_key=api_key)


def format_turns_for_prompt(turns: list[dict]) -> str:
    """
    Render eligible turns as a numbered block the model can cite by turn_id.
    Shared by guide analysis and Q&A so both ground on identical text.
    """
    lines = []
    for t in turns:
        lines.append(f"[{t['turn_id']}] {t['speaker']}: {t['text']}")
    return "\n".join(lines)


GUIDE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["answered", "partial", "not_covered"],
                    },
                    "answer": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "turn_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["turn_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["question_number", "status", "answer", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

ANALYSIS_SYSTEM_PROMPT = """You are analyzing one expert-call transcript for a market research project.

Rules, non-negotiable:
- Answer ONLY using the numbered turns provided below. Do not use outside knowledge about robotic surgery, healthcare markets, or anything else.
- Every quote must be copied EXACTLY (character for character) from the turn text you cite. Do not paraphrase inside a quote.
- Every claim in "answer" must be supported by at least one evidence entry citing a real turn_id from the list provided.
- If the transcript does not address a question, set status to "not_covered" and leave the answer brief (e.g. "Not addressed in this transcript.") with empty evidence.
- If a question is only partially addressed, use status "partial" and only cite what's actually there.
- Never invent a turn_id. Only use turn_ids exactly as given in the source list.
- Preserve any conditions, qualifiers, or scope the expert stated (e.g. "in stronger centres", "not across the whole market"). Do not generalize a qualified statement into an unqualified one.
"""


def build_analysis_user_prompt(parsed_transcript: dict, questions: list[str]) -> str:
    eligible = expert_turns(parsed_transcript)
    turns_block = format_turns_for_prompt(eligible)

    questions_block = "\n".join(
        f"{i}. {q}" for i, q in enumerate(questions, start=1)
    )

    return f"""Expert: {parsed_transcript['expert_name']} ({parsed_transcript['role']}, {parsed_transcript['market']})

Source turns (only these are eligible as evidence):
{turns_block}

Questions to answer:
{questions_block}

Return one answer object per question, in question order."""


def analyze_expert(parsed_transcript: dict, questions: list[str], client: OpenAI = None) -> dict:
    """
    One LLM call per expert covering all guide questions.
    Returns the parsed JSON dict matching GUIDE_ANALYSIS_SCHEMA, with
    expert metadata attached.
    """
    if client is None:
        client = get_openai_client()

    user_prompt = build_analysis_user_prompt(parsed_transcript, questions)

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "guide_analysis",
                "schema": GUIDE_ANALYSIS_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )

    result = json.loads(response.output_text)
    result["expert_num"] = parsed_transcript["expert_num"]
    result["expert_name"] = parsed_transcript["expert_name"]
    result["market"] = parsed_transcript["market"]
    result["role"] = parsed_transcript["role"]
    return result


# ---------------------------------------------------------------------------
# Step 4: Quote validation
# ---------------------------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def combined_turn_lookup(parsed_transcripts: list[dict]) -> dict:
    """turn_id -> turn dict across ALL experts. turn_ids are globally unique
    (E01-*, E02-*, E03-*), so this is safe to merge without collisions."""
    lookup = {}
    for p in parsed_transcripts:
        lookup.update(turn_lookup(p))
    return lookup


def _validate_evidence_list(evidence: list[dict], lookup: dict) -> tuple[list[dict], bool]:
    """
    Shared primitive: given a raw evidence list (turn_id + quote pairs from
    an LLM) and a turn_id -> turn lookup, verify each quote is a verbatim
    (whitespace-normalized) substring of that turn's real text. Timestamp
    and speaker are looked up here, never trusted from the model.

    Used by Step 4 (single-expert guide answers), Step 7 (cross-expert
    synthesis), and Step 9 (Q&A) — one validation mechanism everywhere.
    """
    valid = []
    had_invalid = False

    for ev in evidence:
        turn = lookup.get(ev.get("turn_id"))
        if turn is None:
            had_invalid = True
            continue

        quote_norm = _normalize_whitespace(ev.get("quote", ""))
        text_norm = _normalize_whitespace(turn["text"])

        if quote_norm and quote_norm in text_norm:
            valid.append(
                {
                    "turn_id": ev["turn_id"],
                    "quote": ev["quote"],
                    "timestamp": turn["timestamp"],
                    "speaker": turn["speaker"],
                }
            )
        else:
            had_invalid = True

    return valid, had_invalid


def validate_quotes(analysis: dict, parsed_transcript: dict) -> dict:
    """
    Verify every cited quote in a single expert's guide-answer set is real.

    Invalid evidence (bad turn_id, or quote not actually in that turn) is
    dropped. If an answer had status "answered"/"partial" but ends up with
    zero valid evidence after dropping, its status becomes "unverified" so
    the UI can render it distinctly rather than showing an unsupported claim.

    Returns a new dict; does not mutate the input.
    """
    lookup = turn_lookup(parsed_transcript)
    validated_answers = []

    for ans in analysis["answers"]:
        valid_evidence, had_invalid = _validate_evidence_list(ans.get("evidence", []), lookup)

        new_ans = dict(ans)
        new_ans["evidence"] = valid_evidence
        new_ans["has_unverified_quotes"] = had_invalid

        if had_invalid and not valid_evidence and ans["status"] != "not_covered":
            new_ans["status"] = "unverified"

        validated_answers.append(new_ans)

    result = dict(analysis)
    result["answers"] = validated_answers
    return result


# ---------------------------------------------------------------------------
# Step 7: Cross-call synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {"type": "array", "items": {"$ref": "#/$defs/synthesis_item"}},
        "disagreements": {"type": "array", "items": {"$ref": "#/$defs/synthesis_item"}},
        "caveats": {"type": "array", "items": {"$ref": "#/$defs/synthesis_item"}},
    },
    "required": ["themes", "disagreements", "caveats"],
    "additionalProperties": False,
    "$defs": {
        "synthesis_item": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn_id": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["turn_id", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["title", "description", "evidence"],
            "additionalProperties": False,
        }
    },
}

SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing findings across 3 expert-call transcripts (France, Germany, UK) about robotic surgery adoption.

You are given each expert's already-validated answers to a 6-question interview guide, with supporting evidence (turn_id + exact quote) for each answer.

Rules, non-negotiable:
- When citing evidence, reuse turn_id + quote pairs EXACTLY as given below. Do not invent new quotes, alter wording, or cite a turn_id not shown below.
- Themes: points where 2+ experts converge, even if worded differently.
- Disagreements: points where experts give materially different answers to the same or a related question.
- Do NOT average or collapse the three growth forecasts (Q5) into one number. Each expert stated a different scope/basis for their forecast — if you mention growth, preserve that scope rather than presenting a single blended figure.
- Do NOT imply the German procurement expert and the two clinical experts are interchangeable "expert consensus" — note when a point comes from a procurement vs. clinical vantage point.
- Do NOT flatten the UK position (economics and clinical strategy are described as balanced) into "finance decides everywhere."
- If unsure whether two answers actually agree, put it under caveats rather than themes.
"""


def build_synthesis_user_prompt(validated_analyses: list[dict]) -> str:
    blocks = []
    for a in validated_analyses:
        lines = [f"### {a['expert_name']} — {a['role']} ({a['market']})"]
        for ans in a["answers"]:
            if ans["status"] in ("not_covered", "unverified"):
                continue
            lines.append(f"- Q{ans['question_number']} [{ans['status']}]: {ans['answer']}")
            for ev in ans["evidence"]:
                lines.append(f'  evidence [{ev["turn_id"]}]: "{ev["quote"]}"')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def synthesize_themes(validated_analyses: list[dict], client: OpenAI = None) -> dict:
    """
    One LLM call over all 3 experts' already-validated guide answers.
    Input is the extracted Q&A JSON, not raw transcripts — the model can
    only reuse citations it's already been given, which is what
    _validate_evidence_list below double-checks.
    """
    if client is None:
        client = get_openai_client()

    user_prompt = build_synthesis_user_prompt(validated_analyses)

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "synthesis",
                "schema": SYNTHESIS_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )

    return json.loads(response.output_text)


def validate_synthesis(synthesis: dict, parsed_transcripts: list[dict]) -> dict:
    """Same trusted validator as Step 4, applied across all 3 experts' turns."""
    lookup = combined_turn_lookup(parsed_transcripts)

    def process(items):
        out = []
        for item in items:
            valid_evidence, had_invalid = _validate_evidence_list(item.get("evidence", []), lookup)
            new_item = dict(item)
            new_item["evidence"] = valid_evidence
            new_item["has_unverified_quotes"] = had_invalid
            out.append(new_item)
        return out

    result = dict(synthesis)
    result["themes"] = process(synthesis.get("themes", []))
    result["disagreements"] = process(synthesis.get("disagreements", []))
    result["caveats"] = process(synthesis.get("caveats", []))
    return result


def q5_growth_caveat(validated_analyses: list[dict]) -> dict:
    """
    Deterministic (no LLM) caveat: pulls each expert's Q5 growth-outlook
    answer + its already-validated evidence directly. This one is
    guaranteed to appear and guaranteed grounded — it doesn't depend on
    the synthesis model remembering to flag it.
    """
    lines = []
    evidence = []
    for a in validated_analyses:
        ans = next((x for x in a["answers"] if x["question_number"] == 5), None)
        if not ans or ans["status"] in ("not_covered", "unverified"):
            continue
        lines.append(f"- {a['expert_name']} ({a['market']}): {ans['answer']}")
        evidence.extend(ans["evidence"])

    description = (
        "Each expert's 3–5 year growth outlook uses a different basis and scope "
        '(e.g. "in stronger centres", "in procedure volumes", "in some areas"). '
        "These are not directly comparable and should not be averaged or treated "
        "as one market-wide figure.\n\n" + "\n".join(lines)
    )
    return {
        "title": "Growth forecasts are not directly comparable",
        "description": description,
        "evidence": evidence,
        "has_unverified_quotes": False,
    }


def expert_mix_caveat(parsed_transcripts: list[dict]) -> dict:
    """Deterministic caveat built from role metadata, not the LLM."""
    lines = [f"- {p['expert_name']}: {p['role']} ({p['market']})" for p in parsed_transcripts]
    description = (
        "This sample mixes a procurement perspective with clinical perspectives, "
        "not three like-for-like viewpoints:\n\n" + "\n".join(lines)
    )
    return {
        "title": "Mixed expert types, not a like-for-like sample",
        "description": description,
        "evidence": [],
        "has_unverified_quotes": False,
    }


def full_synthesis(validated_analyses: list[dict], parsed_transcripts: list[dict], client: OpenAI = None) -> dict:
    """Runs the LLM synthesis, validates it, and prepends the two guaranteed
    deterministic caveats so they're always present regardless of what the
    model does or doesn't notice."""
    raw = synthesize_themes(validated_analyses, client=client)
    validated = validate_synthesis(raw, parsed_transcripts)
    validated["caveats"] = [
        q5_growth_caveat(validated_analyses),
        expert_mix_caveat(parsed_transcripts),
    ] + validated["caveats"]
    return validated


# ---------------------------------------------------------------------------
# Step 9: Cross-transcript Q&A
# ---------------------------------------------------------------------------

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "partial", "insufficient_evidence"],
        },
        "answer": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["turn_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "answer", "evidence"],
    "additionalProperties": False,
}

QA_SYSTEM_PROMPT = """You are answering a free-form question using expert-call transcripts from three markets (France, Germany, UK).

Rules, non-negotiable:
- Answer ONLY using the numbered turns provided below. Do not use outside knowledge.
- Every quote must be copied EXACTLY (character for character) from the turn text you cite.
- Every claim must be supported by at least one evidence entry citing a real turn_id from the list provided.
- If the transcripts don't address the question (e.g. it asks for something like market size in dollars, or a topic no expert discussed), set status to "insufficient_evidence" and leave evidence empty. Do not guess or fill the gap with general knowledge.
- If experts across markets say different things, say so explicitly rather than picking one.
- Preserve each expert's stated scope/qualifiers rather than generalizing them.
"""


def format_combined_turns_for_prompt(parsed_transcripts: list[dict]) -> str:
    lines = []
    for p in parsed_transcripts:
        for t in expert_turns(p):
            lines.append(f"[{t['turn_id']}] ({p['market']}) {t['speaker']}: {t['text']}")
    return "\n".join(lines)


def build_qa_user_prompt(question: str, parsed_transcripts: list[dict]) -> str:
    turns_block = format_combined_turns_for_prompt(parsed_transcripts)
    return f"""Source turns from all three expert calls (only these are eligible as evidence):
{turns_block}

Question: {question}

Answer using only the turns above."""


def answer_question(question: str, parsed_transcripts: list[dict], client: OpenAI = None) -> dict:
    if client is None:
        client = get_openai_client()

    user_prompt = build_qa_user_prompt(question, parsed_transcripts)

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "qa_answer",
                "schema": QA_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )

    return json.loads(response.output_text)


def validate_qa_answer(qa_result: dict, parsed_transcripts: list[dict]) -> dict:
    lookup = combined_turn_lookup(parsed_transcripts)
    valid_evidence, had_invalid = _validate_evidence_list(qa_result.get("evidence", []), lookup)

    new_result = dict(qa_result)
    new_result["evidence"] = valid_evidence
    new_result["has_unverified_quotes"] = had_invalid

    if had_invalid and not valid_evidence and qa_result["status"] != "insufficient_evidence":
        new_result["status"] = "unverified"

    return new_result
