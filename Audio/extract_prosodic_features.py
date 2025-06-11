import os
import librosa
import numpy as np
import soundfile as sf

# === Step 1: Feature Extraction Function ===
def extract_pitch_energy(audio_path, sr=16000, frame_length=2048, hop_length=512):
    y, _ = librosa.load(audio_path, sr=sr)

    # 1. Pitch (F0) using PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, frame_length=frame_length, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0)  # Replace NaNs (unvoiced) with 0

    # 2. Energy (RMS)
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Match frame lengths
    min_len = min(len(f0), len(energy))
    f0 = f0[:min_len]
    energy = energy[:min_len]

    # Combine features: shape (n_frames, 2)
    prosodic_features = np.stack([f0, energy], axis=1)
    return prosodic_features

# === Step 2: Process All Audio Files ===
audio_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/audio_wavs"
output_dir = "/scratch/s6028608/MUStARD_Plus_Plus/audio/other_features"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        audio_path = os.path.join(audio_dir, file)
        features = extract_pitch_energy(audio_path)
        
        # Save features as .npy file
        save_path = os.path.join(output_dir, file.replace(".wav", ".npy"))
        np.save(save_path, features)

        print(f"✅ Saved: {save_path}")