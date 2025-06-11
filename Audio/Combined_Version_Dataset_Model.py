#Combined_Version_Dataset_Model.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SarcasmAudioDataset(Dataset):
    def __init__(self, feature_dir, label_dict):
        self.feature_dir = feature_dir
        self.label_dict = label_dict

        self.samples = []
        for file in os.listdir(self.feature_dir):
            if file.endswith(".npy"):
                key = file.replace(".npy", "")
                if key in label_dict:
                    self.samples.append(file)
                else:
                    print(f"⚠️ Warning: '{key}' not found in labels.csv, skipping.")

        if not self.samples:
            raise ValueError("No valid samples found. Check feature files and labels.csv for consistency.")

        print(f"✅ {len(self.samples)} valid samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        feature_path = os.path.join(self.feature_dir, filename)
        features = np.load(feature_path)
        features = torch.tensor(features, dtype=torch.float32)

        label_key = filename.replace(".npy", "")
        label = self.label_dict[label_key]
        return features, torch.tensor(label)

class AudioRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(AudioRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(hn)