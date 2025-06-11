
# Video Modality — Sarcasm Detection

This folder includes code for extracting facial features using OpenFace and training CNN-based classifiers.

## Files

- `extract_features.py`: General script for preprocessing videos.
- `extract_video_embeddings.py`: Legacy script for facial embeddings.
- `Train_Face.py`: Trains a CNN model for sarcasm detection from facial features.
- `Facial_Dataset_Model.py`: Dataset definition for training/testing.
- `extract embd/extract_video_embeddings_openface.py`: Extracts AUs and pose using OpenFace.
- `extract embd/save_video_features.py`: Saves extracted features into serialized files.

## How to Run

```bash
# Make sure OpenFace is installed and configured

# Extract facial features
python3 extract_features.py
python3 extract embd/extract_video_embeddings_openface.py

# Train facial classifier
python3 Train_Face.py

