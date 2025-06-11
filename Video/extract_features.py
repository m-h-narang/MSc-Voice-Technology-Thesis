# extract_features.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Facial_Dataset_Model import FacialExpressionDataset, FaceMLP

# ---------- Configuration ----------
FEATURE_DIR = "openface_features/"
LABEL_PATH = "labels.csv"  # Adjust path if needed
CHECKPOINT_PATH = "face_mlp_checkpoint.pth"
OUTPUT_FEATURES_PATH = "face_features_extracted.pt"
BATCH_SIZE = 32

# ---------- Step 1: Load Labels ----------
label_df = pd.read_csv(LABEL_PATH)
label_dict = dict(zip(label_df['filename'], label_df['label']))

# ---------- Step 2: Load Dataset (no reshape!) ----------
dataset = FacialExpressionDataset(FEATURE_DIR, label_dict, reshape=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Step 3: Load Model and Weights ----------
input_dim = next(iter(dataloader))[0].shape[1]  # Automatically determine input size
model = FaceMLP(input_dim=input_dim, hidden_dim=512, num_classes=2)

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# ---------- Step 4: Extract Features (from penultimate layer) ----------
# Remove last classification layer (keep up to 2nd Linear)
feature_extractor = nn.Sequential(*list(model.model.children())[:-1])

all_features = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        features = feature_extractor(inputs)
        all_features.append(features)
        all_labels.append(labels)

# ---------- Step 5: Save Features ----------
all_features = torch.cat(all_features, dim=0)  # Shape: (N, 512)
all_labels = torch.cat(all_labels, dim=0)

torch.save({
    'features': all_features,
    'labels': all_labels
}, OUTPUT_FEATURES_PATH)

print(f"✅ Saved extracted features to: {OUTPUT_FEATURES_PATH}")
print(f"🔢 Feature shape: {all_features.shape}")