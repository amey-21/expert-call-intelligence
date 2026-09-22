import json
from core import parse_transcript, parse_guide, expert_turns

files = [
    ("data/Transcript_1_France.txt", 1),
    ("data/Transcript_2_Germany.txt", 2),
    ("data/Transcript_3_UK.txt", 3),
]

all_parsed = []
for path, num in files:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parsed = parse_transcript(text, num)
    all_parsed.append(parsed)

    print("=" * 70)
    print(f"Expert {parsed['expert_num']}: {parsed['expert_name']}")
    print(f"Role: {parsed['role']} | Market: {parsed['market']}")
    print(f"Total turns: {len(parsed['turns'])}  |  Expert turns: {len(expert_turns(parsed))}")
    print("-" * 70)
    for t in parsed["turns"]:
        tag = "EXPERT" if t["is_expert"] else "interviewer"
        print(f"[{t['turn_id']}] {t['timestamp']} ({tag}) {t['speaker']}: {t['text'][:70]}...")

print("=" * 70)
print("GUIDE QUESTIONS")
with open("data/Interview_Guide.txt", "r", encoding="utf-8") as f:
    guide_text = f.read()
questions = parse_guide(guide_text)
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

print("=" * 70)
print(f"Sanity checks:")
print(f"  Experts parsed: {len(all_parsed)} (expect 3)")
print(f"  Guide questions: {len(questions)} (expect 6)")
for p in all_parsed:
    print(f"  {p['market']}: {len(p['turns'])} turns, {len(expert_turns(p))} expert turns")
