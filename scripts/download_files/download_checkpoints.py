import requests
import os

# List of (filename, download URL)
files_to_download = [
    (
        "step-5000-checkpoint-point-force.pt",
        "https://huggingface.co/brown-palm/force-prompting/resolve/main/step-5000-checkpoint-point-force.pt"
    ),
    (
        "step-5000-checkpoint-wind-force.pt",
        "https://huggingface.co/brown-palm/force-prompting/resolve/main/step-5000-checkpoint-wind-force.pt"
    ),
]

# Directory to save the files (you can customize this)
save_dir = "checkpoints"

os.makedirs(save_dir, exist_ok=True)

# Download each file
for filename, url in files_to_download:
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(save_dir, filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {os.path.join(save_dir, filename)}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")
