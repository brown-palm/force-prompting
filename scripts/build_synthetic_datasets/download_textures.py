import os
import shutil
from huggingface_hub import snapshot_download

# === Configuration ===
repo_id = "nate-gillman/textures-for-blender"
repo_type = "dataset"
snapshot_dir = "hf_temp_textures_snapshot"

# Map of directory names in the repo to local destination paths
directories_to_download = {
    "HDRIs":                "scripts/build_synthetic_datasets/wind_model_waving_flags/HDRIs",
    "football_textures":    "scripts/build_synthetic_datasets/poke_model_rolling_balls/football_textures",
    "ground_textures":      "scripts/build_synthetic_datasets/poke_model_rolling_balls/ground_textures",
}

# === Step 1: Download full repo snapshot to temp dir ===
snapshot_path = snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=snapshot_dir,
    local_dir_use_symlinks=False
)

# === Step 2: Copy specified directories to target locations ===
for dir_in_repo, target_path in directories_to_download.items():
    src_path = os.path.join(snapshot_path, dir_in_repo)
    
    if not os.path.isdir(src_path):
        print(f"⚠️ Skipping: '{dir_in_repo}' not found in the repo.")
        continue

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(src_path, target_path)

    print(f"✅ Downloaded '{dir_in_repo}' to '{target_path}'")

# === Step 3: Cleanup temporary snapshot ===
shutil.rmtree(snapshot_dir)
print(f"🧹 Cleaned up temp directory '{snapshot_dir}'")

print("🎉 All specified directories downloaded successfully.")
