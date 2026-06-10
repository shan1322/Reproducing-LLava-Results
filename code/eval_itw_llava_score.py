import os
import json
import csv
import time
from groq import Groq
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv('../.env')

client = Groq(api_key=os.getenv('GROQ_KEY'))

CSV_IN  = '../outputs/evals/llava_bench_results.csv'
CSV_OUT = '../outputs/evals/llava_bench_scored.csv'
RESULTS_OUT = '../outputs/evals/results_summary.json'

def get_score(row, retries=3):
    prompt = f"""You are an impartial judge evaluating the quality of an AI assistant's response to a visual question.

[Question]: {row['question']}
[Image Caption]: {row['caption']}
[Reference Answer (GPT-4)]: {row['gpt4_answer']}
[Candidate Answer (LLaVA)]: {row['llava_answer']}

Compare the candidate answer to the reference answer.
Give the candidate answer a score from 1 to 10, where:
- 10 = equally good or better than the reference
- 1 = completely wrong or irrelevant

Respond in this exact JSON format:
{{"score": <number>, "explanation": "<brief reason>"}}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            text = response.choices[0].message.content
            parsed = json.loads(text)
            return parsed['score'], parsed['explanation']
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None, "failed after retries"

with open(CSV_IN, 'r') as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows):
    print(f"Scoring {i+1}/{len(rows)} | {row['image']} | {row['category']}")
    score, explanation = get_score(row)
    row['llava_score'] = score if score is not None else ''
    row['explanation'] = explanation
    row['relative_score'] = round((score / 10) * 100, 1) if score is not None else ''
    time.sleep(0.5)

fieldnames = list(rows[0].keys())
with open(CSV_OUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for r in rows:
    if r['relative_score'] != '':
        groups[r['category']].append(float(r['relative_score']))

category_scores = {cat: round(sum(s)/len(s), 1) for cat, s in sorted(groups.items())}
all_scores = [float(r['relative_score']) for r in rows if r['relative_score'] != '']
overall = round(sum(all_scores)/len(all_scores), 1)

results = {
    "overall_relative_score": overall,
    "total_questions_scored": len(all_scores),
    "total_questions": len(rows),
    "category_scores": category_scores,
    "paper_llava_scores": {
        "overall": 67.3,
        "conv": 57.3,
        "detail": 52.5,
        "complex": 81.7
    }
}

with open(RESULTS_OUT, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*40}")
print(json.dumps(results, indent=2))
print(f"\nScored CSV: {CSV_OUT}")
print(f"Results summary: {RESULTS_OUT}")