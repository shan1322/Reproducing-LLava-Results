import os
import zipfile
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv('../.env')
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# download llava instruct 150k json
hf_hub_download(
    repo_id="liuhaotian/LLaVA-Instruct-150K",
    filename="llava_instruct_150k.json",
    repo_type="dataset",
    local_dir="./data/LLaVA-Instruct-150K"
)

# download COCO 2017 train images
import urllib.request

os.makedirs('../data/COCO/images', exist_ok=True)

print("Downloading COCO train2017 images (~18GB)...")
urllib.request.urlretrieve(
    "http://images.cocodataset.org/zips/train2017.zip",
    "./data/COCO/train2017.zip"
)

print("Extracting...")
with zipfile.ZipFile('../data/COCO/train2017.zip', 'r') as zip_ref:
    zip_ref.extractall('../data/COCO/images')

os.remove('./data/COCO/train2017.zip')
print("Done.")