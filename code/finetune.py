import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from tqdm import tqdm

from data_loader_instruct import LLaVAInstructDataset, collate_fn, IGNORE_INDEX

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


class LLaVA(nn.Module):
    def __init__(self, vision_model, llm, tokenizer):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.tokenizer = tokenizer
        self.W_proj = nn.Linear(1024, 2048, bias=False)

        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.llm.parameters():
            param.requires_grad = True

    def forward(self, images, input_ids, labels, attention_mask, image_positions):
        # get visual embeddings
        vision_outputs = self.vision_model(
            pixel_values=images,
            output_hidden_states=True
        )
        patch_features = vision_outputs.hidden_states[-2][:, 1:, :]          # (B, 256, 1024)
        visual_embeds = self.W_proj(patch_features.to(self.W_proj.weight.dtype))  # (B, 256, 2048)

        # get text embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)                  # (B, T, 2048)

        B, num_visual, dim = visual_embeds.shape
        T = text_embeds.size(1)

        # insert visual embeddings inline at image_position for each sample
        combined_list = []
        full_labels_list = []
        full_mask_list = []

        for b in range(B):
            pos = image_positions[b].item()

            # split text embeddings at image position
            text_before = text_embeds[b, :pos, :]        # (pos, dim)
            text_after = text_embeds[b, pos:, :]         # (T-pos, dim)

            # concat: text_before + visual + text_after
            combined = torch.cat([text_before, visual_embeds[b], text_after], dim=0)  # (pos+256+T-pos, dim)
            combined_list.append(combined)

            # labels: IGNORE for visual positions, keep text labels
            lbl_before = labels[b, :pos]
            lbl_visual = torch.full((num_visual,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            lbl_after = labels[b, pos:]
            full_labels_list.append(torch.cat([lbl_before, lbl_visual, lbl_after], dim=0))

            # attention mask: 1 for visual positions
            mask_before = attention_mask[b, :pos]
            mask_visual = torch.ones(num_visual, dtype=attention_mask.dtype, device=attention_mask.device)
            mask_after = attention_mask[b, pos:]
            full_mask_list.append(torch.cat([mask_before, mask_visual, mask_after], dim=0))

        combined = torch.stack(combined_list, dim=0)       # (B, pos+256+T-pos, dim)
        full_labels = torch.stack(full_labels_list, dim=0)
        full_mask = torch.stack(full_mask_list, dim=0)

        outputs = self.llm(
            inputs_embeds=combined,
            attention_mask=full_mask
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX
        )
        return loss


if __name__ == '__main__':
    device = torch.device('cuda:0')

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.bfloat16
    ).to(device)
    llm = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    model = LLaVA(vision_model, llm, tokenizer).to(device)
    model.W_proj = model.W_proj.to(torch.bfloat16).to(device)

    model.W_proj.load_state_dict(torch.load('../checkpoints/W_proj_final.pt', map_location=device))
    print("loaded W_proj from stage 1")

    model.llm.gradient_checkpointing_enable()

    dataset = LLaVAInstructDataset(
        json_path='../data/LLaVA-Instruct-150K/llava_instruct_150k.json',
        image_folder='../data/COCO/images/train2017',
        processor=processor,
        tokenizer=tokenizer
    )
    dataset.data = dataset.data[:110000]
    


    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            num_workers=8, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        list(model.W_proj.parameters()) + list(model.llm.parameters()),
        lr=2e-5
    )

    for epoch in range(1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (images, input_ids, labels, attention_mask, image_positions) in enumerate(pbar):
            images = images.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            image_positions = image_positions.to(device)

            optimizer.zero_grad()
            loss = model(images, input_ids, labels, attention_mask, image_positions)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if i % 5500 == 0 and i > 0:
                torch.save(model.W_proj.state_dict(), f'../checkpoints/W_proj_ft_step_{i}.pt')
                torch.save(model.llm.state_dict(), f'../checkpoints/llm_ft_step_{i}.pt')

    torch.save(model.W_proj.state_dict(), '../checkpoints/W_proj_ft_final.pt')
    torch.save(model.llm.state_dict(), '../checkpoints/llm_ft_final.pt')