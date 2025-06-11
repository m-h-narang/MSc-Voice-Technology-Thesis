import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from Combined_Version_Dataset_Model import SarcasmAudioDataset, AudioRNN

# 0. Device configuration (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load label dictionary from CSV
labels_df = pd.read_csv("/scratch/s6028608/MUStARD_Plus_Plus/audio/labels.csv")
label_dict = dict(zip(labels_df['filename'], labels_df['label']))

# 2. Dataset and filtering
feature_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/combined_features"
full_dataset = SarcasmAudioDataset(feature_dir=feature_dir, label_dict=label_dict)

# Ensure there are enough samples
if len(full_dataset) == 0:
    raise ValueError("No valid samples found. Check feature files and labels.csv for consistency.")

# 3. Train-test split
indices = list(range(len(full_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# 3.5 Inspect validation set label distribution
val_labels_list = [full_dataset[i][1].item() for i in val_indices]
from collections import Counter
label_map = {0: "Not Sarcastic", 1: "Sarcastic"}
print("📊 Validation label distribution:")
for label_id, count in Counter(val_labels_list).items():
    print(f"  {label_map[label_id]}: {count}")

# 4. Custom collate function for variable-length sequences
def collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)  # [B, max_T, 770]
    labels = torch.stack(labels)
    return features_padded, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 5. Model, loss, optimizer
model = AudioRNN(input_dim=770, hidden_dim=256, num_layers=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    print(f"Epoch {epoch + 1} starting...")
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 7. Save model checkpoint
torch.save(model.state_dict(), "/scratch/s6028608/MUStARD_Plus_Plus/audio/audio_rnn_checkpoint.pth")