import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# Paths
input_path = "/scratch/s6028608/MUStARD_Plus_Plus/text/processed/text_data.csv"
output_path = "/scratch/s6028608/MUStARD_Plus_Plus/text/processed/text_embeddings.pt"

# Load CSV
df = pd.read_csv(input_path)
text_column = 'SENTENCE_clean'
text_data = df[text_column].fillna("").astype(str).tolist()

# Load BERT model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize
encoded_inputs = tokenizer.batch_encode_plus(
    text_data,
    padding='max_length',
    truncation=True,
    max_length=64,
    return_tensors='pt'
)

input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

# Dataloader
dataset = TensorDataset(input_ids, attention_mask)
loader = DataLoader(dataset, batch_size=32)

# Inference
all_embeddings = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Generating BERT embeddings"):
        b_input_ids, b_att_mask = [x.to(device) for x in batch]
        outputs = model(input_ids=b_input_ids, attention_mask=b_att_mask)
        cls_embeddings = outputs.pooler_output  # shape: [batch_size, 768]
        all_embeddings.append(cls_embeddings.cpu())

# Concatenate and save
final_embeddings = torch.cat(all_embeddings, dim=0)  # shape: [1202, 768]
torch.save(final_embeddings, output_path)

print("✅ Final text embedding shape:", final_embeddings.shape)
print("✅ Saved to:", output_path)