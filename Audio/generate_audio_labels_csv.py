import os
import torch
import pandas as pd

# === Updated Paths ===
LABEL_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/text/processed/Preprocessed_with_BERT/labels.pt"
FEATURE_DIR = "/scratch/s6028608/MUStARD_Plus_Plus/audio/wav2vec_features"
OUTPUT_CSV = "/scratch/s6028608/MUStARD_Plus_Plus/audio/labels.csv"

# 1. Load labels
labels = torch.load(LABEL_PATH).tolist()

# 2. Get feature filenames (skip hidden/temp/duplicate files)
feature_files = sorted(f for f in os.listdir(FEATURE_DIR)
                       if f.endswith('.pt') and not f.startswith('.') and not f.startswith('Copy of'))
feature_ids = [os.path.splitext(f)[0] for f in feature_files]

# 3. Sanity check
if len(feature_ids) != len(labels):
    raise ValueError(f"Mismatch: {len(feature_ids)} features vs {len(labels)} labels")

# 4. Create DataFrame
df = pd.DataFrame({
    "filename": feature_ids,
    "label": labels
})

# 5. Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved labels CSV: {OUTPUT_CSV}")