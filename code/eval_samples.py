import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import json
import base64
from io import BytesIO

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ---- CONFIG ----
N = 3
START = 50000
QUESTION = "Describe this image in detail."
OUTPUT_DIR = '../outputs/evals'
# ----------------

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

html = "<html><body style='font-family:monospace;max-width:900px;margin:auto;padding:20px'>"

for idx, sample in enumerate(samples):
    print(f"Processing sample {START+idx}...")

    # load and encode image to base64
    image_path = f"../data/COCO/images/train2017/{sample['image']}"
    image = Image.open(image_path).convert('RGB')
    buf = BytesIO()
    image.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # original conversation
    conv_html = ""
    for turn in sample['conversations']:
        role = "Human" if turn['from'] == 'human' else "Assistant"
        text = turn['value'].replace('<image>', '[IMAGE]').replace('\n', '<br>')
        color = "#003366" if role == "Human" else "#006600"
        conv_html += f"<p><b style='color:{color}'>{role}:</b> {text}</p>"

    # model answer
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.5
        )
    answer = tokenizer.decode(out[0], skip_special_tokens=True)

    html += f"""
    <hr>
    <h2>Sample {START+idx} | {sample['image']}</h2>
    <img src='data:image/jpeg;base64,{img_b64}' style='max-width:400px;border:1px solid #ccc'><br><br>
    <h3>Original Conversation</h3>
    {conv_html}
    <h3>Model Answer to: "{QUESTION}"</h3>
    <p style='background:#f0f0f0;padding:10px'>{answer.replace(chr(10), '<br>')}</p>
    """

html += "</body></html>"

out_path = os.path.join(OUTPUT_DIR, 'eval.html')
with open(out_path, 'w') as f:
    f.write(html)

print(f"Saved to {out_path}")