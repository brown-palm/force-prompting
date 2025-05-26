from huggingface_hub import snapshot_download
import os
import shutil

# === Step 1: Download the public dataset snapshot ===
REPO_ID = "brown-palm/force-prompting-dataset-creation"
tmp_checkout = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
)

# === Step 2: Map the three folders → your local .cache paths ===
FOLDER_MAP = {
    "HDRIs":             os.path.expanduser(".cache/HDRIs"),
    "football_textures": os.path.expanduser(".cache/football_textures"),
    "ground_textures":   os.path.expanduser(".cache/ground_textures"),
}

# === Step 3: Copy each directory out of the snapshot ===
for subfolder, dest in FOLDER_MAP.items():
    src = os.path.join(tmp_checkout, subfolder)
    if not os.path.isdir(src):
        print(f"⚠️  Skipped `{subfolder}`: not found in snapshot.")
        continue

    # Ensure parent folder exists
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Clean out any existing copy
    if os.path.exists(dest):
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    print(f"✅  `{subfolder}` → `{dest}`")

print("🎉 Download & copy complete!")
