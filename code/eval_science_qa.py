import os
import re
import json
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

OUTPUT = '../outputs/evals/scienceqa_results.json'
os.makedirs('../outputs/evals', exist_ok=True)

device = torch.device('cuda:0')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16).to(device)
llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

W_proj = nn.Linear(1024, 2048, bias=False).to(torch.bfloat16).to(device)
W_proj.load_state_dict(torch.load('../checkpoints/W_proj_ft_final.pt', map_location=device))
llm.load_state_dict(torch.load('../checkpoints/llm_ft_final.pt', map_location=device))

ds = load_from_disk('../data/scienceqa')
samples = [s for s in ds if s['image'] is not None]

results = []
correct_total = 0
subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
grade_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

pbar = tqdm(samples, desc="ScienceQA Eval")
for sample in pbar:
    choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(sample['choices'])])
    prompt = f"### Human: {sample['question']}\n{choices_str}\nChoose one letter: A, B, C, or D.\n### Assistant: The answer is"

    image_tensor = processor(images=sample['image'], return_tensors='pt')['pixel_values'].to(torch.bfloat16).to(device)

    with torch.no_grad():
        vision_out = vision_model(pixel_values=image_tensor, output_hidden_states=True)
        visual_embeds = W_proj(vision_out.hidden_states[-2][:, 1:, :])
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        text_embeds = llm.model.embed_tokens(input_ids)
        combined = torch.cat([visual_embeds, text_embeds], dim=1)
        out = llm.generate(
            inputs_embeds=combined,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    match = re.search(r'\b([ABCD])\b', raw)
    predicted = match.group(1) if match else '?'
    correct = chr(65 + sample['answer'])
    is_correct = predicted == correct

    if is_correct:
        correct_total += 1
    subject_stats[sample['subject']]['total'] += 1
    subject_stats[sample['subject']]['correct'] += int(is_correct)
    grade_stats[sample['grade']]['total'] += 1
    grade_stats[sample['grade']]['correct'] += int(is_correct)

    results.append({
        'question': sample['question'],
        'subject': sample['subject'],
        'grade': sample['grade'],
        'predicted': predicted,
        'correct': correct,
        'is_correct': is_correct
    })

    acc = correct_total / len(results) * 100
    pbar.set_postfix(acc=f"{acc:.1f}%")

overall_acc = correct_total / len(results) * 100
subject_acc = {s: round(v['correct']/v['total']*100, 1) for s, v in subject_stats.items()}
grade_acc = {g: round(v['correct']/v['total']*100, 1) for g, v in sorted(grade_stats.items())}

summary = {
    "overall_accuracy": round(overall_acc, 1),
    "total_questions": len(results),
    "correct": correct_total,
    "subject_accuracy": subject_acc,
    "grade_accuracy": grade_acc,
    "paper_llava_accuracy": 90.92
}

with open(OUTPUT, 'w') as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

print(f"\n{'='*40}")
print(json.dumps(summary, indent=2))
print(f"\nSaved to {OUTPUT}")