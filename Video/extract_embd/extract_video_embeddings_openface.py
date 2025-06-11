import os
import subprocess
import torch
import pandas as pd
from tqdm import tqdm

# ========== CONFIG ==========
openface_bin = "/scratch/s6028608/OpenFace/build/bin/FeatureExtraction"  # Modify this
video_dir = "/scratch/s6028608/MUStARD_Plus_Plus/all_videos/final_utterance_videos"
openface_output_dir = "/scratch/s6028608/MUStARD_Plus_Plus/video/Phase1_video/openface_output"
output_pt_path = "/scratch/s6028608/MUStARD_Plus_Plus/video/Phase1_video/output_video/video_features_openface.pt"

os.makedirs(openface_output_dir, exist_ok=True)

# ========== FUNCTION TO RUN OPENFACE ==========
def run_openface(video_path, out_dir):
    cmd = [
        openface_bin,
        "-f", video_path,
        "-out_dir", out_dir,
        "-quiet",
        "-2Dfp",
        "-pose",
        "-aus"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ========== AGGREGATE FEATURES FROM CSV ==========
def extract_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return torch.zeros(100)  # fallback
    # Drop NaNs and take mean across all frames
    df_clean = df.dropna()
    features = df_clean.iloc[:, 4:]  # Drop frame info columns
    return torch.tensor(features.mean().values, dtype=torch.float)

# ========== MAIN LOOP ==========


print("Extracting OpenFace features...")
features = {}
missing = []

for fname in tqdm(sorted(os.listdir(video_dir))):
    if fname.endswith(".mp4"):
        video_path = os.path.join(video_dir, fname)
        key = fname.replace(".mp4", "")
        csv_output = os.path.join(openface_output_dir, key + ".csv")

        if not os.path.exists(csv_output):
            run_openface(video_path, openface_output_dir)

        if os.path.exists(csv_output):
            try:
                emb = extract_features_from_csv(csv_output)
                features[key] = emb
            except Exception as e:
                print(f"Failed to extract from {key}: {e}")
                missing.append(key)
        else:
            print(f"Missing CSV for {key}")
            missing.append(key)


# ========== SAVE ==========
print(f"Saving to {output_pt_path} ...")
os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
torch.save(features, output_pt_path)
print(f"Done. Total videos processed: {len(features)}, Missing: {len(missing)}")