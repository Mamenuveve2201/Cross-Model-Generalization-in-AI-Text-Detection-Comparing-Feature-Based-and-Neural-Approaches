from huggingface_hub import hf_hub_download
import os

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)

print("Downloading datasets from Hugging Face...")

repo_id = "amenuvevemelissa/ai-text-detection-dataset"

files = [
    'master_dataset.csv',
    'features_normalized.csv',
    'persuade_sampled.csv',
    'persuade_with_topics.csv',
]

for filename in files:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir="data/raw"
    )

print("\n✅ All files downloaded successfully!")
print("You can now run the scripts in order starting from 03_Feature_Extraction.")
