import os
import pandas as pd
import torch
import numpy as np

# Paths
OPENFACE_DIR = "/scratch/s6028608/MUStARD_Plus_Plus/video/Phase1_video/openface_output/"
LABEL_CSV = "/scratch/s6028608/MUStARD_Plus_Plus/audio/labels.csv"
OUTPUT_PT = "/scratch/s6028608/MUStARD_Plus_Plus/Phase2/output/video_features_openface.pt"

# Load label CSV to get video clip order
label_df = pd.read_csv(LABEL_CSV)
video_ids = label_df["filename"].tolist()

# Function to extract features from each OpenFace .csv file
def extract_features(csv_path, video_id):
    try:
        df = pd.read_csv(csv_path)

        if df.empty or len(df.columns) == 0:
            print(f"[WARNING] Empty or invalid file: {video_id}.csv — Skipping.")
            return None

        df = df.dropna()

        useful_columns = [
            col for col in df.columns
            if (col.startswith('AU') and col.endswith('_r')) or
               col.startswith('pose_') or
               col.startswith('gaze_')
        ]

        if not useful_columns:
            print(f"[WARNING] No valid features in {video_id}.csv — Skipping.")
            return None

        feature_df = df[useful_columns]
        aggregated = feature_df.mean().values.astype(np.float32)

        return torch.tensor(aggregated)

    except pd.errors.EmptyDataError:
        print(f"[WARNING] No data in file: {video_id}.csv — Skipping.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process {video_id}.csv: {e}")
        return None

# Process all clips
all_features = []
processed_ids = []
skipped_ids = []

for vid in video_ids:
    csv_file = os.path.join(OPENFACE_DIR, f"{vid}.csv")

    if not os.path.exists(csv_file):
        print(f"[WARNING] Missing OpenFace CSV: {vid}.csv — Skipping.")
        skipped_ids.append(vid)
        continue

    features = extract_features(csv_file, vid)
    if features is None:
        skipped_ids.append(vid)
        continue

    all_features.append(features)
    processed_ids.append(vid)

# Final stacking and saving
if not all_features:
    raise RuntimeError("❌ No valid video features could be extracted. Aborting.")

video_tensor = torch.stack(all_features)
print(f"✅ Final video tensor shape: {video_tensor.shape}")

# Save with clip IDs as a dictionary (optional for traceability)
feature_dict = {vid: feat for vid, feat in zip(processed_ids, all_features)}
torch.save(feature_dict, OUTPUT_PT)
print(f"✅ Saved features for {len(processed_ids)} clips to: {OUTPUT_PT}")

# Optional: Report skipped clips
if skipped_ids:
    print(f"\n⚠️ Skipped {len(skipped_ids)} clips due to missing or invalid data.")
    print(f"Examples: {skipped_ids[:5]}")