# Evaluate_And_Infer_AudioRNN.py

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, accuracy_score
from Combined_Version_Dataset_Model import AudioRNN

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Collate function
def collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels = torch.stack(labels)
    return features_padded, labels

# Load validation set
val_dataset = torch.load("/scratch/s6028608/MUStARD_Plus_Plus/audio/val_dataset.pt")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"✅ {len(val_dataset)} validation samples loaded.")

# Model
model = AudioRNN(input_dim=770, hidden_dim=256, num_layers=2, num_classes=2).to(device)
model.load_state_dict(torch.load("/scratch/s6028608/MUStARD_Plus_Plus/audio/audio_rnn_checkpoint.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_features, batch_labels in val_loader:
        batch_features = batch_features.to(device)
        outputs = model(batch_features)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())

acc = accuracy_score(all_labels, all_preds)
print(f"\n✅ Accuracy: {acc:.4f}\n")

print("📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))