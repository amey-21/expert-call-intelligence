import os

import streamlit as st
from dotenv import load_dotenv

from core import (
    parse_transcript,
    parse_guide,
    analyze_expert,
    validate_quotes,
    full_synthesis,
    answer_question,
    validate_qa_answer,
)

load_dotenv()

st.set_page_config(page_title="Expert Call Analyzer", layout="wide")
st.title("Expert Call Analyzer")

# Resolve paths relative to this file's own location, not the shell's
# current working directory (which varies by how Streamlit gets launched).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SAMPLE_FILES = [
    (os.path.join(DATA_DIR, "Transcript_1_France.txt"), 1),
    (os.path.join(DATA_DIR, "Transcript_2_Germany.txt"), 2),
    (os.path.join(DATA_DIR, "Transcript_3_UK.txt"), 3),
]
SAMPLE_GUIDE = os.path.join(DATA_DIR, "Interview_Guide.txt")

STATUS_LABELS = {
    "answered": "✅ Answered",
    "partial": "🟡 Partial",
    "not_covered": "⚪ Not covered",
    "insufficient_evidence": "⚪ Insufficient evidence in transcripts",
    "unverified": "🔴 Unverified (quote failed validation)",
}


# ---------------------------------------------------------------------------
# Step 2: Load data
# ---------------------------------------------------------------------------

def load_sample_data():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Expected a 'data' folder next to app.py at: {DATA_DIR}\n"
            "Make sure the data/ directory (with the 3 transcripts + guide) "
            "was copied alongside app.py and core.py."
        )
    parsed = []
    for path, num in SAMPLE_FILES:
        with open(path, encoding="utf-8") as f:
            parsed.append(parse_transcript(f.read(), num))
    with open(SAMPLE_GUIDE, encoding="utf-8") as f:
        questions = parse_guide(f.read())
    return parsed, questions


with st.sidebar:
    st.header("Data")
    use_sample = st.checkbox("Use bundled sample case", value=True)

    parsed_transcripts = None
    questions = None

    if use_sample:
        try:
            parsed_transcripts, questions = load_sample_data()
            st.success(f"Loaded {len(parsed_transcripts)} sample transcripts.")
        except FileNotFoundError as e:
            st.error(str(e))
    else:
        uploaded_transcripts = st.file_uploader(
            "Upload transcript .txt files", type="txt", accept_multiple_files=True
        )
        uploaded_guide = st.file_uploader("Upload interview guide .txt", type="txt")

        if uploaded_transcripts and uploaded_guide:
            parsed_transcripts = [
                parse_transcript(f.read().decode("utf-8"), i)
                for i, f in enumerate(uploaded_transcripts, start=1)
            ]
            questions = parse_guide(uploaded_guide.read().decode("utf-8"))
            st.success(f"Loaded {len(parsed_transcripts)} uploaded transcripts.")
        else:
            st.info("Upload transcripts + a guide, or check the sample-case box above.")

    if parsed_transcripts:
        st.divider()
        for p in parsed_transcripts:
            expert_n = sum(1 for t in p["turns"] if t["is_expert"])
            st.caption(f"**{p['expert_name']}** ({p['market']}) — {expert_n} expert turns")

    st.divider()
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not set. Add it to a .env file to run analysis.")


if not parsed_transcripts or not questions:
    st.stop()


# ---------------------------------------------------------------------------
# Step 3 + 4: Run analysis. Results live in session_state below, so normal
# Streamlit reruns do not re-hit the API. Do not cache this function: a user
# needs a genuinely fresh model call when retrying an incomplete response.
# ---------------------------------------------------------------------------

def run_analysis(_parsed_transcripts, _questions):
    results = []
    for t in _parsed_transcripts:
        raw = analyze_expert(t, _questions)
        validated = validate_quotes(raw, t)
        results.append(validated)
    return results


tab_guide, tab_quotes, tab_themes, tab_ask = st.tabs(
    ["Interview Guide", "Quotes", "Themes & Disagreements", "Ask the Transcripts"]
)

with tab_guide:
    run = st.button("Run guide analysis", type="primary")

    if "guide_results" not in st.session_state:
        st.session_state["guide_results"] = None

    if run:
        with st.spinner("Analyzing transcripts..."):
            try:
                st.session_state["guide_results"] = run_analysis(parsed_transcripts, questions)
            except RuntimeError as e:
                st.error(str(e))

    results = st.session_state["guide_results"]

    if results is None:
        st.info("Click **Run guide analysis** to answer all 6 guide questions for each expert.")
    else:
        for qi, question in enumerate(questions, start=1):
            with st.expander(f"**Q{qi}. {question}**", expanded=False):
                cols = st.columns(len(results))
                for col, expert_result in zip(cols, results):
                    with col:
                        st.markdown(f"**{expert_result['expert_name']}** ({expert_result['market']})")
                        # Structured output should contain one item per guide
                        # question, but never let an incomplete model response
                        # crash the whole demo page.
                        ans = next(
                            (
                                a
                                for a in expert_result["answers"]
                                if a["question_number"] == qi
                            ),
                            None,
                        )
                        if ans is None:
                            st.caption("🔴 Unverified (incomplete model response)")
                            st.warning(
                                "No answer was returned for this question. "
                                "Run the guide analysis again to retry."
                            )
                            continue
                        st.caption(STATUS_LABELS.get(ans["status"], ans["status"]))
                        st.write(ans["answer"])
                        for ev in ans["evidence"]:
                            st.markdown(
                                f"> \"{ev['quote']}\"\n\n"
                                f"— *{ev['speaker']}, {ev['timestamp']} [{ev['turn_id']}]*"
                            )

with tab_quotes:
    results = st.session_state.get("guide_results")

    if not results:
        st.info("Run the guide analysis on the **Interview Guide** tab first — this tab reuses that evidence, no extra API call.")
    else:
        expert_names = [r["expert_name"] for r in results]
        filter_choice = st.selectbox("Filter by expert", ["All"] + expert_names)

        for expert_result in results:
            if filter_choice != "All" and expert_result["expert_name"] != filter_choice:
                continue

            has_any = any(a["evidence"] for a in expert_result["answers"])
            if not has_any:
                continue

            st.subheader(f"{expert_result['expert_name']} — {expert_result['market']}")
            for ans in expert_result["answers"]:
                for ev in ans["evidence"]:
                    st.markdown(
                        f"> \"{ev['quote']}\"\n\n"
                        f"— *{ev['speaker']}, {ev['timestamp']} [{ev['turn_id']}]* "
                        f"(Q{ans['question_number']})"
                    )

with tab_themes:
    results = st.session_state.get("guide_results")

    if not results:
        st.info("Run the guide analysis on the **Interview Guide** tab first — synthesis builds on those validated answers.")
    else:
        run_synth = st.button("Run cross-call synthesis", type="primary")

        if "synthesis_result" not in st.session_state:
            st.session_state["synthesis_result"] = None

        if run_synth:
            with st.spinner("Synthesizing across all 3 experts..."):
                try:
                    parsed_by_num = {p["expert_num"]: p for p in parsed_transcripts}
                    ordered_transcripts = [parsed_by_num[r["expert_num"]] for r in results]
                    st.session_state["synthesis_result"] = full_synthesis(results, ordered_transcripts)
                except RuntimeError as e:
                    st.error(str(e))

        synthesis = st.session_state["synthesis_result"]

        if synthesis is None:
            st.info("Click **Run cross-call synthesis** to identify common themes, disagreements, and caveats.")
        else:
            st.markdown("### Common themes")
            if not synthesis["themes"]:
                st.caption("No themes returned.")
            for item in synthesis["themes"]:
                with st.expander(item["title"]):
                    st.write(item["description"])
                    for ev in item["evidence"]:
                        st.markdown(f"> \"{ev['quote']}\"\n\n— *{ev['speaker']}, {ev['timestamp']} [{ev['turn_id']}]*")

            st.markdown("### Disagreements")
            if not synthesis["disagreements"]:
                st.caption("No disagreements returned.")
            for item in synthesis["disagreements"]:
                with st.expander(item["title"]):
                    st.write(item["description"])
                    for ev in item["evidence"]:
                        st.markdown(f"> \"{ev['quote']}\"\n\n— *{ev['speaker']}, {ev['timestamp']} [{ev['turn_id']}]*")

            st.markdown("### Caveats")
            st.caption("The first two are always checked directly from the data, not left to the model to remember.")
            for item in synthesis["caveats"]:
                with st.expander(item["title"]):
                    st.write(item["description"])
                    for ev in item["evidence"]:
                        st.markdown(f"> \"{ev['quote']}\"\n\n— *{ev['speaker']}, {ev['timestamp']} [{ev['turn_id']}]*")

with tab_ask:
    question = st.text_input("Ask a question across all three transcripts")
    ask_clicked = st.button("Ask", type="primary")

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    if ask_clicked and question.strip():
        with st.spinner("Searching transcripts..."):
            try:
                raw = answer_question(question, parsed_transcripts)
                validated = validate_qa_answer(raw, parsed_transcripts)
                st.session_state["qa_history"].insert(0, (question, validated))
            except RuntimeError as e:
                st.error(str(e))

    for q, a in st.session_state["qa_history"]:
        st.markdown(f"**Q: {q}**")
        st.caption(STATUS_LABELS.get(a["status"], a["status"]))
        st.write(a["answer"])
        for ev in a["evidence"]:
            st.markdown(f"> \"{ev['quote']}\"\n\n— *{ev['speaker']}, {ev['timestamp']} [{ev['turn_id']}]*")
        st.divider()
