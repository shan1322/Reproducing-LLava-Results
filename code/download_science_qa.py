import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv('../.env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

ds = load_dataset('derek-thomas/ScienceQA', split='test')
ds.save_to_disk('../data/scienceqa')

print(f"Downloaded {len(ds)} samples")
print(f"Saved to ../data/scienceqa")
print(f"Sample: {ds[0]}")