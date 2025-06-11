import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertModel, get_scheduler
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os

# ========== Paths ==========
input_dir = "/scratch/s6028608/MUStARD_Plus_Plus/text/processed/Preprocessed_with_BERT"
model_output_path = os.path.join(input_dir, "bert_sarcasm_classifier.pt")

# ========== Load Encoded Data ==========
input_ids = torch.load(os.path.join(input_dir, 'text_input_ids.pt'))
attention_mask = torch.load(os.path.join(input_dir, 'text_attention_mask.pt'))
labels = torch.load(os.path.join(input_dir, 'labels.pt'))

# ========== Class Weights ==========
class_weights = compute_class_weight('balanced', classes=[0, 1], y=labels.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ========== Dataset & Dataloaders ==========
dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ========== BERT Classifier ==========
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# ========== Initialize ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

optimizer = torch.optim.AdamW([
    {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 5e-6},
    {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 2e-5},
])

total_steps = len(train_loader) * 6
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ========== Training & Validation Loops ==========
def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
            logits = model(b_input_ids, b_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(b_labels.cpu().tolist())
    return all_labels, all_preds

# ========== Training ==========
epochs = 6
best_f1 = 0
patience = 2
no_improve_epochs = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        logits = model(b_input_ids, b_mask)
        loss = criterion(logits, b_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    y_true, y_pred = evaluate()
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\nEpoch {epoch+1}/{epochs} — Loss: {avg_train_loss:.4f} — Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Not Sarcastic", "Sarcastic"]))

    # Early Stopping
    if f1 > best_f1:
        best_f1 = f1
        no_improve_epochs = 0
        torch.save(model.state_dict(), model_output_path)
        print(f"✅ New best model saved to: {model_output_path}")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("⏹️ Early stopping triggered.")
            break

# ========== Final Output ==========
print(f"🏁 Best model F1: {best_f1:.4f}")