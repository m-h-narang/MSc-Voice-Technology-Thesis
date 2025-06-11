import os
import torch
import numpy as np

# Define paths
wav2vec_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/wav2vec_features"
prosody_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/other_features"
output_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/combined_features"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(prosody_dir):
    if not file.endswith(".npy"):
        continue

    base_name = file.replace(".npy", "")
    wav2vec_file = base_name + ".pt"

    wav2vec_path = os.path.join(wav2vec_dir, wav2vec_file)
    prosody_path = os.path.join(prosody_dir, file)
    output_path = os.path.join(output_dir, base_name + ".npy")

    if not os.path.exists(wav2vec_path):
        print(f"⚠️ Skipping {file} — Wav2Vec2 feature not found.")
        continue

    try:
        # Load Wav2Vec2 features
        wav2vec_feat = torch.load(wav2vec_path)

        if isinstance(wav2vec_feat, tuple):
            wav2vec_feat = wav2vec_feat[0]

        # Ensure we use only first channel if shape is (2, T, D)
        if wav2vec_feat.ndim == 3 and wav2vec_feat.shape[0] == 2:
            wav2vec_feat = wav2vec_feat[0]  # Use only one channel

        wav2vec_feat = wav2vec_feat.cpu().numpy()
        prosody_feat = np.load(prosody_path)

        # Align lengths
        min_len = min(wav2vec_feat.shape[0], prosody_feat.shape[0])
        wav2vec_feat = wav2vec_feat[:min_len, :]
        prosody_feat = prosody_feat[:min_len, :]

        # Concatenate features
        combined = np.concatenate([wav2vec_feat, prosody_feat], axis=1)

        # Save
        np.save(output_path, combined)
        print(f"✅ Saved: {output_path} | shape: {combined.shape}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")