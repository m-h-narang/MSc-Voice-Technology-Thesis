import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# Paths
input_path = "/scratch/s6028608/MUStARD_Plus_Plus/text/processed/text_data.csv"
output_path = "/scratch/s6028608/MUStARD_Plus_Plus/text/processed/text_embeddings.pt"

# Load the data
df = pd.read_csv(input_path)
text_column = 'SENTENCE_clean'

# Clean and prepare text
text_data = df[text_column].fillna("").astype(str).tolist()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Store all embeddings
all_embeddings = []

# Batch process for efficiency
batch_size = 32
for i in tqdm(range(0, len(text_data), batch_size), desc="Extracting BERT embeddings"):
    batch = text_data[i:i + batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler_output: [batch_size, 768]
        batch_embeddings = outputs.pooler_output.cpu()
        all_embeddings.append(batch_embeddings)

# Concatenate all batches
final_embeddings = torch.cat(all_embeddings, dim=0)
print(f"✅ Final text embedding shape: {final_embeddings.shape}")  # Should be [1202, 768]

# Save
torch.save(final_embeddings, output_path)
print(f"✅ Saved to: {output_path}")