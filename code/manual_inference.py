import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

device = torch.device('cuda:0')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16).to(device)
llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

W_proj = nn.Linear(1024, 2048, bias=False).to(torch.bfloat16).to(device)
W_proj.load_state_dict(torch.load('../checkpoints/W_proj_ft_final.pt', map_location=device))
llm.load_state_dict(torch.load('../checkpoints/llm_ft_final.pt', map_location=device))

image = Image.open('../data/manual_test/001.jpg').convert('RGB')
image_tensor = processor(images=image, return_tensors='pt')['pixel_values'].to(torch.bfloat16).to(device)

question = "Describe this image in detail and what the animal might be doing."
prompt = f"### Human: {question}\n### Assistant:"

with torch.no_grad():
    vision_out = vision_model(pixel_values=image_tensor, output_hidden_states=True)
    visual_embeds = W_proj(vision_out.hidden_states[-2][:, 1:, :])
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    text_embeds = llm.model.embed_tokens(input_ids)
    combined = torch.cat([visual_embeds, text_embeds], dim=1)
    out = llm.generate(inputs_embeds=combined,max_new_tokens=512,do_sample=True,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.5
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))