import os
import shutil
import zipfile
from huggingface_hub import snapshot_download

# datasets/
# ├── point-force/
# │   └── train/
# │       ├── point_force_23000/
# │       └── point_force_23000.csv
# └── wind-force/
#     └── train/
#         ├── wind_force_15359/
#         └── wind_force_15359.csv

# === Configuration ===
repo_id = "brown-palm/force-prompting-training-datasets"
download_root = "datasets"  # Top-level output directory
local_dir="hf_temp_datasets"

# === Step 1: Download entire repo snapshot ===
repo_snapshot_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

# === Step 2: Define paths ===
zip_path = os.path.join(repo_snapshot_path, "dataset_videos.zip")
csv1_src = os.path.join(repo_snapshot_path, "point_force_23000.csv")
csv2_src = os.path.join(repo_snapshot_path, "wind_force_15359.csv")

# === Step 3: Extract ZIP to a temp location ===
temp_extract_dir = os.path.join(download_root, "_extracted_temp")
os.makedirs(temp_extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(temp_extract_dir)

# === Step 4: Move folders and CSVs into nested structure ===
# Define target paths
target1_dir = os.path.join(download_root, "point-force", "train", "point_force_23000")
target2_dir = os.path.join(download_root, "wind-force", "train", "wind_force_15359")
target1_csv = os.path.join(download_root, "point-force", "train", "point_force_23000.csv")
target2_csv = os.path.join(download_root, "wind-force", "train", "wind_force_15359.csv")

# Move video folders
shutil.move(os.path.join(temp_extract_dir, "point_force_23000"), target1_dir)
shutil.move(os.path.join(temp_extract_dir, "wind_force_15359"), target2_dir)

# Move CSVs
os.makedirs(os.path.dirname(target1_csv), exist_ok=True)
os.makedirs(os.path.dirname(target2_csv), exist_ok=True)
shutil.copy(csv1_src, target1_csv)
shutil.copy(csv2_src, target2_csv)

# Clean up temporary extract folder
shutil.rmtree(temp_extract_dir)

# Clean up temporary local dir
shutil.rmtree(local_dir)

print("✅ Downloaded and organized dataset successfully.")
