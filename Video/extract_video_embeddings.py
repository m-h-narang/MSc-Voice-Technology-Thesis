# extract_video_embeddings.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from Facial_Dataset_Model import FacialExpressionDataset, FaceMLP

# ---------- Paths ----------
FEATURE_DIR = "/scratch/s6028608/MUStARD_Plus_Plus/video/openface_features"
LABEL_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/video/labels.csv"
CHECKPOINT_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/video/face_mlp_checkpoint.pth"
OUTPUT_FEATURES_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/video/video_embeddings.pt"
BATCH_SIZE = 32

# ---------- Step 1: Load Labels ----------
label_df = pd.read_csv(LABEL_PATH)
label_dict = dict(zip(label_df['filename'], label_df['label']))

# ---------- Step 2: Dataset (no reshape!) ----------
dataset = FacialExpressionDataset(FEATURE_DIR, label_dict, reshape=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Step 3: Load Model ----------
input_dim = next(iter(dataloader))[0].shape[1]
model = FaceMLP(input_dim=input_dim, hidden_dim=512, num_classes=2)

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# ---------- Step 4: Extract Features ----------
feature_extractor = nn.Sequential(*list(model.model.children())[:-1])

all_features = []

with torch.no_grad():
    for inputs, _ in dataloader:
        features = feature_extractor(inputs)
        all_features.append(features)

all_features = torch.cat(all_features, dim=0)  # Shape: (N, 512)

# ---------- Step 5: Save Only Features ----------
torch.save(all_features, OUTPUT_FEATURES_PATH)

print(f"✅ Saved video embeddings to: {OUTPUT_FEATURES_PATH}")
print(f"🔢 Video embedding shape: {all_features.shape}")