
from huggingface_hub import list_repo_files
import os

repo_id = "Srikanth22MH1A42C6/cataract-classification"
try:
    files = list_repo_files(repo_id)
    print(f"FILES_IN_REPO: {files}")
except Exception as e:
    print(f"ERROR: {e}")

# Check Space
space_id = "Srikanth22MH1A42C6/model-api"
slug = space_id.replace('/', '-').lower()
url = f"https://{slug}.hf.space/run/predict"
print(f"SPACE_URL: {url}")
