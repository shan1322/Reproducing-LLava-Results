import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import zipfile

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# download chat.json
hf_hub_download(
    repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
    filename="chat.json",
    repo_type="dataset",
    local_dir="./data/CC3M"
)

# download images.zip
hf_hub_download(
    repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
    filename="images.zip",
    repo_type="dataset",
    local_dir="./data/CC3M"
)

# extract images
with zipfile.ZipFile('./data/CC3M/images.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/CC3M/images')

# delete zip
os.remove('./data/CC3M/images.zip')