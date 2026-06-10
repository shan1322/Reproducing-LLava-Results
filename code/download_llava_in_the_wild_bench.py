import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv('../.env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

snapshot_download(
    repo_id='liuhaotian/llava-bench-in-the-wild',
    repo_type='dataset',
    local_dir='../data/eval/llava-bench-in-the-wild'
)

print("Done.")