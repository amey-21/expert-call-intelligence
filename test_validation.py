import json
from core import parse_transcript, validate_quotes

with open("data/Transcript_1_France.txt", encoding="utf-8") as f:
    martin = parse_transcript(f.read(), 1)

fake_analysis = {
    "answers": [
        {
            # Valid: real substring of E01-T02
            "question_number": 1,
            "status": "answered",
            "answer": "Adoption is growing but concentrated in larger hospitals.",
            "evidence": [
                {
                    "turn_id": "E01-T02",
                    "quote": "Adoption is growing, but it is still concentrated in larger academic hospitals",
                }
            ],
        },
        {
            # Invalid: fabricated quote not actually in E01-T04
            "question_number": 2,
            "status": "answered",
            "answer": "The main barrier is regulatory approval from the health ministry.",
            "evidence": [
                {
                    "turn_id": "E01-T04",
                    "quote": "The main barrier is regulatory approval from the health ministry.",
                }
            ],
        },
        {
            # Invalid turn_id entirely (hallucinated)
            "question_number": 3,
            "status": "answered",
            "answer": "Budgets matter a lot.",
            "evidence": [
                {"turn_id": "E01-T99", "quote": "anything"}
            ],
        },
        {
            # not_covered, no evidence expected, should pass through untouched
            "question_number": 4,
            "status": "not_covered",
            "answer": "Not addressed in this transcript.",
            "evidence": [],
        },
    ]
}

result = validate_quotes(fake_analysis, martin)
print(json.dumps(result, indent=2))

q1, q2, q3, q4 = result["answers"]

assert q1["has_unverified_quotes"] is False
assert q1["status"] == "answered"
assert q1["evidence"][0]["timestamp"] == "00:18"  # real timestamp looked up, not from model
print("\n[PASS] Q1 valid quote kept, real timestamp attached.")

assert q2["has_unverified_quotes"] is True
assert q2["evidence"] == []
assert q2["status"] == "unverified"
print("[PASS] Q2 fabricated quote dropped, status downgraded to 'unverified'.")

assert q3["has_unverified_quotes"] is True
assert q3["evidence"] == []
assert q3["status"] == "unverified"
print("[PASS] Q3 hallucinated turn_id caught and dropped.")

assert q4["status"] == "not_covered"
assert q4["has_unverified_quotes"] is False
print("[PASS] Q4 not_covered passes through unchanged.")
