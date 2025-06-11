'''
#train.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multimodal_dataset import MultimodalDataset
from fusion_model import CrossModalFusionModel
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved features
data_path = "/scratch/s6028608/MUStARD_Plus_Plus/Phase2/output/combined_multimodal_features.pt"
loaded = torch.load(data_path, map_location=device)

# Extract and verify components
text_feats = loaded.get('text')
audio_feats = loaded.get('audio')
visual_feats = loaded.get('video')
labels = loaded.get('labels')

assert text_feats.shape[0] == audio_feats.shape[0] == visual_feats.shape[0] == labels.shape[0], \
    "Mismatch in number of samples across modalities."

if audio_feats.dim() == 2:
    audio_feats = audio_feats.unsqueeze(1)

train_idx, val_idx = train_test_split(
    range(len(labels)),
    test_size=0.2,
    stratify=labels.cpu(),
    random_state=42
)

train_dataset = MultimodalDataset(
    text_feats[train_idx],
    audio_feats[train_idx],
    visual_feats[train_idx],
    labels[train_idx],
    device
)
val_dataset = MultimodalDataset(
    text_feats[val_idx],
    audio_feats[val_idx],
    visual_feats[val_idx],
    labels[val_idx],
    device
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model, optimizer, loss
model = CrossModalFusionModel().to(device)

# Xavier initialization (better for ReLU-based models)
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
loss_fn = torch.nn.CrossEntropyLoss()

# Training with Early Stopping
best_val_acc = 0
patience = 3
counter = 0
model_path = "best_model.pt"

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for text, audio, visual, label in train_loader:
        optimizer.zero_grad()
        output = model(text, audio, visual)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for text, audio, visual, label in val_loader:
            output = model(text, audio, visual)
            val_preds.extend(output.argmax(dim=1).cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Loss: {total_loss:.4f}")
    print(f"Epoch {epoch} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), model_path)
        print(f"✅ New best model saved (Val Acc: {val_acc:.4f})")
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered.")
            break

# Final best model load (optional for evaluation)
model.load_state_dict(torch.load(model_path))
print(f"\n🔁 Best validation accuracy: {best_val_acc:.4f}")
'''

# train.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from fusion_model import CrossModalFusionModel
#from fusion_model import CrossModalAttentionFusionModel
from multimodal_dataset import MultimodalDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved features
data_path = "/scratch/s6028608/MUStARD_Plus_Plus/Phase2/output/combined_multimodal_features.pt"
loaded = torch.load(data_path, map_location=device)

text_feats = loaded.get('text')
audio_feats = loaded.get('audio')
visual_feats = loaded.get('video')
labels = loaded.get('labels')

assert text_feats.shape[0] == audio_feats.shape[0] == visual_feats.shape[0] == labels.shape[0]

# Add channel dim to audio if missing
if audio_feats.dim() == 2:
    audio_feats = audio_feats.unsqueeze(1)

# Split: 70% train, 15% val, 15% test
indices = list(range(len(labels)))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels.cpu(), random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx].cpu(), random_state=42)

train_dataset = MultimodalDataset(text_feats[train_idx], audio_feats[train_idx], visual_feats[train_idx], labels[train_idx], device)
val_dataset = MultimodalDataset(text_feats[val_idx], audio_feats[val_idx], visual_feats[val_idx], labels[val_idx], device)
test_dataset = MultimodalDataset(text_feats[test_idx], audio_feats[test_idx], visual_feats[test_idx], labels[test_idx], device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model setup
model = CrossModalFusionModel().to(device)
#model = CrossModalAttentionFusionModel().to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
loss_fn = torch.nn.CrossEntropyLoss()

# Training
best_val_acc = 0
patience = 3
counter = 0
model_path = "best_model.pt"

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for text, audio, visual, label in train_loader:
        optimizer.zero_grad()
        output = model(text, audio, visual)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for text, audio, visual, label in val_loader:
            output = model(text, audio, visual)
            val_preds.extend(output.argmax(dim=1).cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Loss: {total_loss:.4f}")
    print(f"Epoch {epoch} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), model_path)
        print(f"✅ New best model saved (Val Acc: {val_acc:.4f})")
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered.")
            break

# Load best model for test evaluation
model.load_state_dict(torch.load(model_path))
model.eval()

# Test Evaluation
test_preds, test_labels = [], []
with torch.no_grad():
    for text, audio, visual, label in test_loader:
        output = model(text, audio, visual)
        test_preds.extend(output.argmax(dim=1).cpu().numpy())
        test_labels.extend(label.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='weighted')

print("\n📊 Final Test Evaluation:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("Classification Report:")
print(classification_report(test_labels, test_preds))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds))