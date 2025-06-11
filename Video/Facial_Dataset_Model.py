import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def features_to_image(features):
    # Split features by type (adjust indices as per your feature order)
    au_features = features[:17]           # AUs
    pose_features = features[17:23]       # 6 pose features
    gaze_features = features[23:25]       # 2 gaze features

    # Reshape each to 1D
    au_map = np.array(au_features).reshape(1, -1)
    pose_map = np.array(pose_features).reshape(1, -1)
    gaze_map = np.array(gaze_features).reshape(1, -1)

    # Pad each to same length (32 for uniformity)
    def pad_to_length(arr, length=32):
        return np.pad(arr, ((0, 0), (0, length - arr.shape[1])), mode='constant')

    au_map = pad_to_length(au_map)
    pose_map = pad_to_length(pose_map)
    gaze_map = pad_to_length(gaze_map)

    # Stack as 3-channel tensor
    image = np.stack([au_map, pose_map, gaze_map], axis=0)  # shape: (3, 1, 32)
    return torch.tensor(image, dtype=torch.float32)

class FacialExpressionDataset(Dataset):
    def __init__(self, feature_dir, label_dict, reshape=False):
        self.feature_dir = feature_dir
        self.label_dict = label_dict
        self.reshape = reshape

        self.samples = []
        for file in os.listdir(feature_dir):
            if file.endswith(".csv"):
                key = file.replace(".csv", "")
                if key in label_dict:
                    self.samples.append(file)
                else:
                    print(f"⚠️ Warning: '{key}' not found in label_dict")

        if not self.samples:
            raise ValueError("❌ No matching OpenFace samples found.")

        print(f"✅ Found {len(self.samples)} OpenFace feature files.")

        # Fit scaler using all data
        all_features = []
        for filename in self.samples:
            df = pd.read_csv(os.path.join(self.feature_dir, filename))
            df = df.drop(columns=["frame", "timestamp"], errors="ignore")
            all_features.append(df.mean(axis=0).values)
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        file_path = os.path.join(self.feature_dir, filename)
        df = pd.read_csv(file_path)
        df = df.drop(columns=["frame", "timestamp"], errors="ignore")

        # Feature engineering
        features = df.mean(axis=0).values
        features = self.scaler.transform([features])[0]
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if self.reshape:
            features_tensor = features_to_image(features_tensor.numpy())  # shape: (3, 1, 32)

        label = self.label_dict[filename.replace(".csv", "")]
        return features_tensor, torch.tensor(label)

class FaceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=2):
        super(FaceMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)