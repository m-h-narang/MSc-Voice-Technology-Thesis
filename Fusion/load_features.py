#load_features.py
import torch
import os
import pandas as pd

# Paths to saved .pt feature files
TEXT_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/text/Phase1_text/output_text/text_features.pt"
AUDIO_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/audio/Phase1_audio/output_audio/audio_features.pt"
VIDEO_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/Phase2/output/video_features_openface.pt"
LABEL_CSV_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/audio/labels.csv"
OUTPUT_PATH = "/scratch/s6028608/MUStARD_Plus_Plus/Phase2/output/combined_multimodal_features.pt"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_convert_to_tensor(item):
    if torch.is_tensor(item):
        return item
    elif isinstance(item, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in item):
            return torch.tensor(item, dtype=torch.float)
        else:
            raise TypeError(f"List contains non-numeric items: {item}")
    else:
        raise TypeError(f"Unsupported item type for conversion: {type(item)}")

def dict_or_list_to_tensor(data, modality_name="features", expected_length=None):
    def is_valid_tensor(t):
        return torch.is_tensor(t) and (expected_length is None or t.shape[0] == expected_length)

    if isinstance(data, dict):
        # Special handling for dict of tensors (e.g., video features)
        if all(torch.is_tensor(v) for v in data.values()) and all(len(v.shape) == 1 or len(v.shape) == 2 for v in data.values()):
            print(f"[INFO] Detected dict of per-clip tensors for {modality_name} features.")
            try:
                tensor_list = list(data.values())
                stacked_tensor = torch.stack(tensor_list)
                print(f"{modality_name.capitalize()} features shape (after stacking): {stacked_tensor.shape}")
                return stacked_tensor
            except Exception as e:
                raise ValueError(f"Failed to stack {modality_name} tensors: {e}")

        candidates = []
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                print(f"[INFO] For {modality_name}: Skipping key '{k}' because it contains a list of strings (likely IDs).")
                continue
            try:
                tensor = safe_convert_to_tensor(v)
                if is_valid_tensor(tensor):
                    candidates.append((k, tensor))
            except Exception as e:
                print(f"[WARNING] For {modality_name}: Could not convert key '{k}' to tensor: {e}")

        if not candidates:
            raise ValueError(f"No valid tensors found in {modality_name} with expected length {expected_length}.")

        best_tensor = max(candidates, key=lambda kv: kv[1].shape[1] if kv[1].dim() > 1 else 0)
        print(f"[INFO] For {modality_name}: Using tensor from key '{best_tensor[0]}' with shape {best_tensor[1].shape}")
        return best_tensor[1]

    elif isinstance(data, list):
        tensors = []
        for i, item in enumerate(data):
            try:
                tensor = safe_convert_to_tensor(item)
                if is_valid_tensor(tensor):
                    tensors.append(tensor)
            except Exception as e:
                print(f"[WARNING] For {modality_name}: Could not convert list item at index {i} to tensor: {e}")

        if not tensors:
            raise ValueError(f"No valid tensors in list for {modality_name} with expected length {expected_length}.")

        best_tensor = max(tensors, key=lambda t: t.shape[1] if t.dim() > 1 else 0)
        print(f"[INFO] For {modality_name}: Using tensor with shape {best_tensor.shape}")
        return best_tensor

    elif torch.is_tensor(data):
        if expected_length and data.shape[0] != expected_length:
            raise ValueError(f"{modality_name} tensor has wrong length: {data.shape[0]} != {expected_length}")
        return data

    else:
        raise TypeError(f"Unsupported data type for {modality_name}: {type(data)}")


# Load features
print("Loading text features...")
text_loaded = torch.load(TEXT_PATH, map_location=device, weights_only=False)
text_features = dict_or_list_to_tensor(text_loaded, "text features")
print(f"Text features shape: {text_features.shape}")

print("Loading audio features...")
audio_loaded = torch.load(AUDIO_PATH, map_location=device, weights_only=False)
audio_features = dict_or_list_to_tensor(audio_loaded, "audio features")
print(f"Audio features shape: {audio_features.shape}")

print("Loading video features...")
video_loaded = torch.load(VIDEO_PATH, map_location=device, weights_only=False)
video_features = dict_or_list_to_tensor(video_loaded, "video features")
print(f"Video features shape: {video_features.shape}")

print("Loading labels...")
labels_df = pd.read_csv(LABEL_CSV_PATH)
labels = torch.tensor(labels_df['label'].values, dtype=torch.long)
print(f"Labels shape: {labels.shape}")

# Validations
num_samples = text_features.shape[0]
assert audio_features.shape[0] == num_samples, f"Audio samples ({audio_features.shape[0]}) != text samples ({num_samples})"
assert video_features.shape[0] == num_samples, f"Video samples ({video_features.shape[0]}) != text samples ({num_samples})"
assert labels.shape[0] == num_samples, f"Label samples ({labels.shape[0]}) != text samples ({num_samples})"

# Save final multimodal features
print("Saving combined multimodal features...")
torch.save({
    'text': text_features,
    'audio': audio_features,
    'video': video_features,
    'labels': labels
}, OUTPUT_PATH)

print("✅ Combined features and labels saved to:")
print(OUTPUT_PATH)