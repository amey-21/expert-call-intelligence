"""
Tests Step 3 without a live API call:
1. Prompt construction looks right (spot-check by eye).
2. JSON schema is well-formed.
3. Response parsing + metadata attachment works, using a mocked client
   that returns a fake-but-realistic structured response.

The actual OpenAI call itself needs to be tested on your machine — this
sandbox can't reach api.openai.com (network allowlist blocks it).
"""

import json
from core import (
    parse_transcript,
    parse_guide,
    build_analysis_user_prompt,
    analyze_expert,
    GUIDE_ANALYSIS_SCHEMA,
)

with open("data/Transcript_1_France.txt", encoding="utf-8") as f:
    martin = parse_transcript(f.read(), 1)

with open("data/Interview_Guide.txt", encoding="utf-8") as f:
    questions = parse_guide(f.read())

# --- 1. Prompt looks right ---
prompt = build_analysis_user_prompt(martin, questions)
print("=" * 70)
print("USER PROMPT PREVIEW")
print("=" * 70)
print(prompt)

assert "E01-T01" not in prompt, "Interviewer turn leaked into prompt context!"
assert "E01-T02" in prompt, "Expert turn missing from prompt!"
assert "6." in prompt, "Not all 6 questions made it into the prompt!"
print("\n[PASS] Interviewer turns excluded, expert turns + all 6 questions present.")

# --- 2. Schema is well-formed JSON ---
json.dumps(GUIDE_ANALYSIS_SCHEMA)
print("[PASS] Schema serializes cleanly.")


# --- 3. Mock the OpenAI client, test parsing + metadata attachment ---
class FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class FakeResponses:
    def create(self, **kwargs):
        # Return a plausible structured answer for Q1 only, to keep it short.
        fake_json = {
            "answers": [
                {
                    "question_number": 1,
                    "status": "answered",
                    "answer": "Adoption is growing but concentrated in larger academic hospitals.",
                    "evidence": [
                        {
                            "turn_id": "E01-T02",
                            "quote": "Adoption is growing, but it is still concentrated in larger academic hospitals and private centres with stronger capital budgets.",
                        }
                    ],
                },
                {
                    "question_number": 2,
                    "status": "not_covered",
                    "answer": "Not addressed in this transcript.",
                    "evidence": [],
                },
            ]
        }
        return FakeResponse(json.dumps(fake_json))


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


result = analyze_expert(martin, questions, client=FakeClient())

print("=" * 70)
print("PARSED RESULT (mocked API response)")
print("=" * 70)
print(json.dumps(result, indent=2))

assert result["expert_name"] == "Dr. Jean Martin"
assert result["market"] == "France"
assert result["answers"][0]["evidence"][0]["turn_id"] == "E01-T02"
print("\n[PASS] Metadata attached correctly, evidence structure intact.")
