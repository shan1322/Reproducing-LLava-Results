import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import json
import textwrap

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

N = 3
START = 110000
QUESTION = "Describe this image in detail."
OUTPUT_DIR = '../outputs/evals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda:0')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16).to(device)
llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

W_proj = nn.Linear(1024, 2048, bias=False).to(torch.bfloat16).to(device)
W_proj.load_state_dict(torch.load('../checkpoints/W_proj_ft_final.pt', map_location=device))
llm.load_state_dict(torch.load('../checkpoints/llm_ft_final.pt', map_location=device))

data = json.load(open('../data/LLaVA-Instruct-150K/llava_instruct_150k.json'))
samples = data[START:START+N]

IMG_W = 400
TEXT_W = 500
ROW_H = 420
PADDING = 20
FONT_SIZE = 14

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
    font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()
    font_bold = font

total_h = ROW_H * N + PADDING * 2
total_w = IMG_W + TEXT_W + PADDING * 3
collage = Image.new('RGB', (total_w, total_h), color=(245, 245, 245))
draw = ImageDraw.Draw(collage)

for idx, sample in enumerate(samples):
    print(f"Processing {START+idx}...")
    y_offset = PADDING + idx * ROW_H

    # load image
    image_path = f"../data/COCO/images/train2017/{sample['image']}"
    image = Image.open(image_path).convert('RGB')
    image.thumbnail((IMG_W, ROW_H - PADDING * 2))
    collage.paste(image, (PADDING, y_offset + PADDING))

    # generate answer
    image_tensor = processor(images=image, return_tensors='pt')['pixel_values'].to(torch.bfloat16).to(device)
    prompt = f"### Human: {QUESTION}\n### Assistant:"
    with torch.no_grad():
        vision_out = vision_model(pixel_values=image_tensor, output_hidden_states=True)
        visual_embeds = W_proj(vision_out.hidden_states[-2][:, 1:, :])
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        text_embeds = llm.model.embed_tokens(input_ids)
        combined = torch.cat([visual_embeds, text_embeds], dim=1)
        out = llm.generate(
            inputs_embeds=combined,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.2,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.5
        )
    answer = tokenizer.decode(out[0], skip_special_tokens=True)

    # draw text panel
    tx = IMG_W + PADDING * 2
    ty = y_offset + PADDING

    draw.text((tx, ty), f"Sample {START+idx} | {sample['image']}", font=font_bold, fill=(50, 50, 50))
    ty += 22
    draw.text((tx, ty), f"Q: {QUESTION}", font=font_bold, fill=(0, 80, 160))
    ty += 22

    # wrap answer text
    wrapped = textwrap.wrap(answer, width=55)
    for line in wrapped[:15]:  # max 15 lines
        draw.text((tx, ty), line, font=font, fill=(30, 30, 30))
        ty += 18

    # separator line
    draw.line([(0, y_offset + ROW_H - 5), (total_w, y_offset + ROW_H - 5)], fill=(200, 200, 200), width=1)

out_path = os.path.join(OUTPUT_DIR, 'collage.png')
collage.save(out_path)
print(f"Saved to {out_path}")