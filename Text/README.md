# Text Modality — Sarcasm Detection

This folder contains scripts for preprocessing text data, extracting BERT embeddings, and training text-based classifiers.

## Files

- `prepare_text_data.py`: Cleans and formats raw text from the dataset.
- `extract_bert_embeddings.py`: Extracts `[CLS]` token embeddings using `bert-base-uncased` for standalone text classification.
- `extract_bert_embeddings_for_fusion.py`: Similar to above, but stores data in a format compatible with multimodal fusion.
- `train_bert_classifier.py`: Trains a BERT-based sarcasm detector using the extracted embeddings.
- `run_svm_baseline.py`: Trains an SVM classifier using TF-IDF features as a baseline.
- `extract embd/extract_text_embeddings.py`: Used in Phase 1 for baseline feature extraction.

## How to Run

```bash
# Prepare data
python3 prepare_text_data.py

# Extract embeddings
python3 extract_bert_embeddings.py

# Train classifier
python3 train_bert_classifier.py

# Optional: SVM Baseline
python3 run_svm_baseline.py
