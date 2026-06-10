import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
BEGIN_SIGNAL = "### "
END_SIGNAL = "\n"
HUMAN_ROLE = "Human"
GPT_ROLE = "Assistant"


class LLaVAInstructDataset(Dataset):
    def __init__(self, json_path, image_folder, processor, tokenizer):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # load image
        image = Image.open(f"{self.image_folder}/{sample['image']}").convert('RGB')
        image_tensor = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

        # build full conversation string
        # track which character ranges are assistant turns for masking
        full_text = ""
        assistant_ranges = []  # list of (start_char, end_char) for assistant answer text only

        for turn in sample['conversations']:
            text = turn['value'].replace(IMAGE_TOKEN, '').strip()

            if turn['from'] == 'human':
                full_text += f"{BEGIN_SIGNAL}{HUMAN_ROLE}: {text}{END_SIGNAL}"
            else:
                prefix = f"{BEGIN_SIGNAL}{GPT_ROLE}: "
                answer_start = len(full_text) + len(prefix)
                answer_end = answer_start + len(text) + len(END_SIGNAL)
                full_text += f"{prefix}{text}{END_SIGNAL}"
                assistant_ranges.append((answer_start, answer_end))

        # tokenize full conversation at once
        tokenized = self.tokenizer(
            full_text,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = tokenized['input_ids'][0]

        # build labels — start all as IGNORE
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # unmask only assistant answer tokens using char to token mapping
        # use char_to_token to find token positions of assistant ranges
        encoding = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        offset_mapping = encoding['offset_mapping']  # list of (char_start, char_end) per token

        for ans_start, ans_end in assistant_ranges:
            for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
                if char_start >= ans_start and char_end <= ans_end and char_end > char_start:
                    labels[tok_idx] = input_ids[tok_idx]

        # find image position — first human turn had <image>, insert after "### Human: "
        first_human_prefix = f"{BEGIN_SIGNAL}{HUMAN_ROLE}: "
        image_position = len(self.tokenizer(
            first_human_prefix,
            add_special_tokens=False
        )['input_ids'])

        return image_tensor, input_ids, labels, image_position


def collate_fn(batch):
    images, input_ids, labels, image_positions = zip(*batch)

    images = torch.stack(images)
    image_positions = torch.tensor(image_positions, dtype=torch.long)

    max_len = max(x.size(0) for x in input_ids)

    padded_input_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(len(input_ids), max_len, dtype=torch.long)

    for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
        padded_input_ids[i, :ids.size(0)] = ids
        padded_labels[i, :lbl.size(0)] = lbl
        attention_mask[i, :ids.size(0)] = 1

    return images, padded_input_ids, padded_labels, attention_mask, image_positions


if __name__ == '__main__':
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    dataset = LLaVAInstructDataset(
        json_path='../data/LLaVA-Instruct-150K/llava_instruct_150k.json',
        image_folder='../data/COCO/images/train2017',
        processor=processor,
        tokenizer=tokenizer
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            num_workers=4, collate_fn=collate_fn)

    images, input_ids, labels, attention_mask, image_positions = next(iter(dataloader))
    print(f"images: {images.shape}")
    print(f"input_ids: {input_ids.shape}")
    print(f"labels: {labels.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"image_positions: {image_positions}")
    print(f"non-ignored label tokens: {(labels != -100).sum().item()}")