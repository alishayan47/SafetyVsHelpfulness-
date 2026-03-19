"""
download_models.py
------------------
Run this ONCE on Wolpertinger to download both fine-tuned models from
HuggingFace to local disk before starting model_api.py.

Usage:
    python3 download_models.py
"""

from huggingface_hub import snapshot_download
import os

HF_TOKEN  = os.getenv("HF_TOKEN", "")   # export HF_TOKEN=hf_your_token_here
BASE_DIR  = os.path.expanduser("~/models")

MODELS = {
    "qwen2.5-7b-safetywolf":  "ea4034/qwen2.5-7b-safetywolf",
    "llama3.1-8b-safetywolf": "ea4034/llama3.1-8b-safetywolf",
}

os.makedirs(BASE_DIR, exist_ok=True)

for folder, repo_id in MODELS.items():
    local_path = os.path.join(BASE_DIR, folder)
    if os.path.isdir(local_path) and os.listdir(local_path):
        print(f"[SKIP] {folder} already exists at {local_path}")
        continue

    print(f"\n[↓] Downloading {repo_id}  →  {local_path}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=HF_TOKEN,
    )
    print(f"[✓] {folder} downloaded")

print("\nAll models ready.")
print(f"Models stored in: {BASE_DIR}")
