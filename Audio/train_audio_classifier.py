import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Paths and hyperparams
FEATURE_DIR = "/scratch/s6028608/MUStARD_Plus_Plus/audio/wav2vec_features"
LABEL_FILE = "/scratch/s6028608/MUStARD_Plus_Plus/audio/labels.csv"
MODEL_PATH = os.path.join(FEATURE_DIR, "audio_sarcasm_classifier.pt")

MAX_LEN = 300
BATCH_SIZE = 16
EPOCHS = 25
HIDDEN_SIZE = 256
DROPOUT = 0.5
LEARNING_RATE = 1e-4
INPUT_DIM = 768

# Dataset
class AudioDataset(Dataset):
    def __init__(self, label_csv, feature_dir, max_len=MAX_LEN):
        self.data = pd.read_csv(label_csv)
        self.feature_dir = feature_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feature_path = os.path.join(self.feature_dir, row['filename'] + '.pt')
        features = torch.load(feature_path)  # Ensure features are torch tensors

        # Collapse to (seq_len, feature_dim) if necessary
        if features.dim() == 3:
            features = features.mean(dim=0)
        elif features.dim() == 1:
            features = features.unsqueeze(0)

        # Pad or truncate to max_len
        seq_len = features.size(0)
        if seq_len < self.max_len:
            pad = torch.zeros(self.max_len - seq_len, features.size(1))
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:self.max_len]

        label = torch.tensor(row['label'], dtype=torch.long)
        return features, label

# Attention module
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        weighted = (x * weights.unsqueeze(-1)).sum(dim=1)
        return weighted

# Model
class AudioRNNWithAttention(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(DROPOUT)
        self.attention = Attention(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        attn_out = self.attention(out)
        dropped = self.dropout(attn_out)
        return self.classifier(dropped)

# Prepare dataset
dataset = AudioDataset(LABEL_FILE, FEATURE_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Class weights
labels = dataset.data['label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioRNNWithAttention().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# Training loop
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"], output_dict=True)
    f1 = report['macro avg']['f1-score']

    print(f"\nEpoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f} — F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))

    scheduler.step(f1)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ New best model saved: {MODEL_PATH}")