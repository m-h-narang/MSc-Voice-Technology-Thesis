```markdown
# Audio Modality — Sarcasm Detection

This folder includes scripts for extracting prosodic and wav2vec features, aligning them, and training audio classifiers.

## Files

- `extract_wav2vec_features.py`: Extracts deep audio embeddings using Wav2Vec2.
- `extract_prosodic_features.py`: Extracts MFCCs and pitch features.
- `align_and_concat.py`: Combines prosodic and wav2vec embeddings into a unified format.
- `train_audio_classifier.py`: Trains an RNN on the combined audio features.
- `Evaluate_And_Infer_AudioRNN.py`: Evaluates the trained audio classifier.
- `Train.py`: Alternative training script (used during experimentation).
- `Combined_Version_Dataset_Model.py`: Dataset pipeline for audio models.
- `generate_audio_labels_csv.py`: Prepares label files for audio classification.
- `extract_audio_embeddings.py`: Embedding extractor for Phase 2.
- `extract embd/extract_audio_embeddings.py`: Duplicate extractor used in Phase 1 (kept for completeness).

## How to Run

```bash
# Extract features
python3 extract_wav2vec_features.py
python3 extract_prosodic_features.py
python3 align_and_concat.py

# Train audio model
python3 train_audio_classifier.py

# Evaluate
python3 Evaluate_And_Infer_AudioRNN.py
