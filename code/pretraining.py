import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from tqdm import tqdm
import os

load_dotenv('../.env')

class LLaVADataset(Dataset):
    def __init__(self, chat_json_path, image_folder, processor):
        with open(chat_json_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(f"{self.image_folder}/{sample['image']}").convert('RGB')
        image_tensor = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        human = sample['conversations'][0]['value']
        gpt = sample['conversations'][1]['value']
        return image_tensor, human, gpt

class LLaVA(nn.Module):
    def __init__(self, vision_model, llm, tokenizer):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.tokenizer = tokenizer
        self.W_proj = nn.Linear(1024, 2048, bias=False).to('cuda:0')

        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, images, humans, gpts):
        vision_outputs = self.vision_model(
            pixel_values=images.to('cuda:0'),
            output_hidden_states=True
        )
        patch_features = vision_outputs.hidden_states[-2][:, 1:, :]
        visual_tokens = self.W_proj(patch_features)

        human_tokens = self.tokenizer(list(humans), return_tensors='pt', padding=True).to('cuda:0')
        gpt_tokens = self.tokenizer(list(gpts), return_tensors='pt', padding=True).to('cuda:0')

        human_embeddings = self.llm.model.embed_tokens(human_tokens['input_ids'])
        gpt_embeddings = self.llm.model.embed_tokens(gpt_tokens['input_ids'])

        combined = torch.cat([visual_tokens, human_embeddings, gpt_embeddings], dim=1)
        outputs = self.llm(inputs_embeds=combined.to(self.llm.dtype))
        logits = outputs.logits

        N = gpt_tokens['input_ids'].shape[1]
        answer_logits = logits[:, -(N+1):-1, :]
        answer_labels = gpt_tokens['input_ids']

        loss = F.cross_entropy(
            answer_logits.reshape(-1, 32000),
            answer_labels.reshape(-1)
        )
        return loss

if __name__ == '__main__':
    device = torch.device('cuda:0')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    dataset = LLaVADataset(
        chat_json_path='../data/CC3M/chat.json',
        image_folder='../data/CC3M/images',
        processor=processor
    )
    dataset.data = dataset.data[:50000]

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = LLaVA(vision_model, llm, tokenizer).to(device)
    optimizer = torch.optim.Adam(model.W_proj.parameters(), lr=2e-3)

    for epoch in range(1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (images, humans, gpts) in enumerate(pbar):
            optimizer.zero_grad()
            loss = model(images, humans, gpts)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.W_proj.state_dict(), '../checkpoints/W_proj_final.pt')