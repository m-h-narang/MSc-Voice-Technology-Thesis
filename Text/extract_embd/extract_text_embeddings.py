#extract_text_embeddings.py
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ========== CONFIG ==========
csv_path = "/scratch/s6028608/MUStARD_Plus_Plus/mustard++_text.csv"
output_path = "/scratch/s6028608/MUStARD_Plus_Plus/text/Phase1_text/text_features.pt"
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD CSV ==========
print("Loading CSV...")
df = pd.read_csv(csv_path)

# Filter only utterance rows (those ending in '_u') and with non-empty sentence
utterances = df[df["KEY"].str.endswith("_u")].dropna(subset=["SENTENCE"]).reset_index(drop=True)
sentences = utterances["SENTENCE"].tolist()
labels = utterances["Sarcasm"].fillna(0).astype(int).tolist()
keys = utterances["KEY"].tolist()

# ========== LOAD BERT ==========
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# ========== EXTRACT EMBEDDINGS ==========
print("Extracting embeddings...")
embeddings = []

with torch.no_grad():
    for sentence in tqdm(sentences, desc="Processing"):
        encoded = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
        output = model(**encoded)
        cls_emb = output.last_hidden_state[:, 0, :]  # CLS token
        embeddings.append(cls_emb.squeeze().cpu())

text_features = torch.stack(embeddings)

# ========== SAVE OUTPUT ==========
print(f"Saving features to {output_path} ...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

torch.save({
    "text_features": text_features,
    "labels": torch.tensor(labels),
    "keys": keys
}, output_path)

print("Done.")