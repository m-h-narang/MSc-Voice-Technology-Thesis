import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from Facial_Dataset_Model import FacialExpressionDataset
from torchvision.models import resnet152, ResNet152_Weights

# --------------------------
# Step 1: CNN Model
# --------------------------
class ResNetFace(nn.Module):
    def __init__(self, num_classes):
        super(ResNetFace, self).__init__()
        self.model = resnet152(weights=ResNet152_Weights.DEFAULT)  # Updated for new API

        # Modify input conv to handle 3-channel 8x8 pseudo-image
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze early layers (optional but helps on small data)
        for param in self.model.layer1.parameters():
            param.requires_grad = False

        # Replace FC with dropout + classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --------------------------
# Step 2: Training
# --------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {correct/total:.4f}")

# --------------------------
# Step 3: Main
# --------------------------
if __name__ == "__main__":
    FEATURE_DIR = "openface_features/"
    LABEL_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/video/labels.csv"
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 10

    # Load labels.csv and convert to dict {filename_without_ext: label}
    label_df = pd.read_csv(LABEL_PATH)
    label_dict = dict(zip(label_df['filename'], label_df['label']))

    # Dataset with reshape=True for CNN
    dataset = FacialExpressionDataset(FEATURE_DIR, label_dict, reshape=True)

    labels_for_stratify = [label_dict[sample.replace(".csv", "")] for sample in dataset.samples]
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels_for_stratify, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFace(num_classes=NUM_CLASSES).to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=EPOCHS)