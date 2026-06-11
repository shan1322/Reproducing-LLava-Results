import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import json
import torch
import torch.nn as nn
import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.generation").setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

from datasets import load_from_disk
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

OUTPUT = '../outputs/evals/scienceqa_results.json'
os.makedirs('../outputs/evals', exist_ok=True)

OPTIONS = ["A", "B", "C", "D", "E"]

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

def parse_answer(raw, choices, num_choices):
    # level 1: direct letter
    if raw in OPTIONS[:num_choices]:
        return raw
    # level 2: "A. " format
    if len(raw) >= 3 and raw[0] in OPTIONS[:num_choices] and raw[1:3] == ". ":
        return raw[0]
    # level 3: "The answer is X"
    pattern = re.compile(r'The answer is ([A-E])\.?')
    res = pattern.findall(raw)
    if len(res) == 1:
        return res[0]
    # level 4: match choice text in response
    raw_lower = raw.lower()
    for i, choice in enumerate(choices[:num_choices]):
        if choice.lower()[:25] in raw_lower:
            return OPTIONS[i]
    return "FAILED"

results = []
correct_total = 0
failed_total = 0
answered_total = 0
subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
grade_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
debug_count = 0

pbar = tqdm(samples, desc="ScienceQA Eval")
for sample in pbar:
    num_choices = len(sample['choices'])
    choices_str = "\n".join([f"{OPTIONS[i]}. {c}" for i, c in enumerate(sample['choices'])])
    prompt = f"### Human: {sample['question']}\n{choices_str}\n### Assistant: The answer is"

    image_tensor = processor(images=sample['image'], return_tensors='pt')['pixel_values'].to(torch.bfloat16).to(device)

    with torch.no_grad():
        vision_out = vision_model(pixel_values=image_tensor, output_hidden_states=True)
        visual_embeds = W_proj(vision_out.hidden_states[-2][:, 1:, :])
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        text_embeds = llm.model.embed_tokens(input_ids)
        combined = torch.cat([visual_embeds, text_embeds], dim=1)
        out = llm.generate(
            inputs_embeds=combined,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    predicted = parse_answer(raw, sample['choices'], num_choices)
    correct = OPTIONS[sample['answer']]

    if debug_count < 5:
        print(f"raw={repr(raw)} | predicted={predicted} | correct={correct}", flush=True)
        debug_count += 1

    if predicted == "FAILED":
        failed_total += 1
    else:
        answered_total += 1
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
        'is_correct': predicted == correct if predicted != "FAILED" else False
    })

    acc = (correct_total / answered_total * 100) if answered_total > 0 else 0
    fail_pct = (failed_total / len(results) * 100)
    pbar.set_postfix(acc=f"{acc:.1f}%", failed=f"{fail_pct:.1f}%")

overall_acc = (correct_total / answered_total * 100) if answered_total > 0 else 0
subject_acc = {s: round(v['correct']/v['total']*100, 1) for s, v in subject_stats.items() if v['total'] > 0}
grade_acc = {g: round(v['correct']/v['total']*100, 1) for g, v in sorted(grade_stats.items()) if v['total'] > 0}

summary = {
    "overall_accuracy": round(overall_acc, 1),
    "total_questions": len(results),
    "answered": answered_total,
    "failed_parse": failed_total,
    "failed_pct": round(failed_total / len(results) * 100, 1),
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