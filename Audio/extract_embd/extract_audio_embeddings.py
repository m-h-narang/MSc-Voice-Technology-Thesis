import os
import torch
import pandas as pd
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# ======== CONFIG ========
csv_path = "/scratch/s6028608/MUStARD_Plus_Plus/mustard++_text.csv"
audio_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/audio_wavs"
output_path = "/scratch/s6028608/MUStARD_Plus_Plus/audio/Phase1_audio/output_audio/audio_features.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== LOAD CSV ========
print("Loading CSV...")
df = pd.read_csv(csv_path)
utterances = df[df["KEY"].str.endswith("_u")].dropna(subset=["SENTENCE"]).reset_index(drop=True)

keys = utterances["KEY"].tolist()
labels = utterances["Sarcasm"].fillna(0).astype(int).tolist()

# ======== LOAD Wav2Vec2 ========
print("Loading Wav2Vec2 model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

# ======== EXTRACT EMBEDDINGS ========
print("Extracting audio embeddings...")
features = []
missing = []

for key in tqdm(keys, desc="Processing"):
    wav_path = os.path.join(audio_dir, f"{key}.wav")
    
    if not os.path.exists(wav_path):
        missing.append(key)
        features.append(torch.zeros(768))  # Fill missing with zero vector
        continue

    waveform, sr = torchaudio.load(wav_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    with torch.no_grad():
        inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls_embedding = last_hidden.mean(dim=1).squeeze().cpu()  # Mean-pooling

        features.append(cls_embedding)

audio_features = torch.stack(features)
print(f"Missing files: {len(missing)}")

# ======== SAVE ========
print(f"Saving features to {output_path} ...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

torch.save({
    "audio_features": audio_features,
    "labels": torch.tensor(labels),
    "keys": keys,
    "missing": missing
}, output_path)

print("Done.")