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

def get_scores(row, retries=3):
    prompt = f"""You are a helpful and precise assistant for checking the quality of answers.

[Context]
{row['caption']}

[Question]
{row['question']}

[Assistant 1]
{row['gpt4_answer']}

[End of Assistant 1]

[Assistant 2]
{row['llava_answer']}

[End of Assistant 2]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, and level of detail of their responses.
Give each assistant a score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please output two scores on the first line, separated by a space, like: 8 6
Then provide a brief explanation on the second line."""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.2
            )
            text = response.choices[0].message.content.strip()
            first_line = text.split('\n')[0].replace(',', ' ').strip()
            parts = first_line.split()
            gpt4_score = float(parts[0])
            llava_score = float(parts[1])
            explanation = text.split('\n')[1] if len(text.split('\n')) > 1 else ''
            return gpt4_score, llava_score, explanation
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None, None, "failed after retries"

with open(CSV_IN, 'r') as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows):
    print(f"Scoring {i+1}/{len(rows)} | {row['image']} | {row['category']}")
    gpt4_score, llava_score, explanation = get_scores(row)

    if gpt4_score is not None and llava_score is not None:
        row['gpt4_judge_score'] = gpt4_score
        row['llava_score'] = llava_score
        row['explanation'] = explanation
        row['relative_score'] = round((llava_score / gpt4_score) * 100, 1) if gpt4_score > 0 else ''
    else:
        row['gpt4_judge_score'] = ''
        row['llava_score'] = ''
        row['explanation'] = explanation
        row['relative_score'] = ''

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