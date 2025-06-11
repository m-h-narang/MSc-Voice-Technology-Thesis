# MSc-Voice-Technology-Thesis

# Multimodal Sarcasm Detection with Transformers

This repository contains the full implementation of a multimodal sarcasm detection pipeline based on text, audio, and video signals, trained and evaluated using the MUStARD++ dataset. Each modality is processed and trained separately, then fused using a cross-attention transformer to enhance classification performance.

---

## Project Overview

This project explores sarcasm detection using three key modalities:

- **Text**: Uses BERT embeddings and trains text classifiers (BERT and SVM baselines).
- **Audio**: Uses Wav2Vec2 and prosodic features (pitch, MFCCs) with an RNN classifier.
- **Video**: Extracts facial features using OpenFace (AUs, pose) and uses CNN-based models.
- **Fusion**: Combines all three modalities using a transformer-based fusion architecture with cross-attention.

---

## Folder Structure

```

MUStARD\_Plus\_Plus/
│
├── Text/                   # Scripts for text processing and classification
├── Audio/                  # Scripts for audio feature extraction and models
├── Video/                  # Scripts for OpenFace-based visual modeling
├── Fusion/                 # Final multimodal fusion model
├── all\_videos/            # Input videos (not included in this repo)
├── README.md               # This file

````

---

## Installation Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MUStARD_Plus_Plus.git
   cd MUStARD_Plus_Plus
````

2. Create and activate a virtual environment (or use conda):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required Python packages:

   ```bash
   pip install numpy pandas scikit-learn tqdm transformers torchaudio soundfile
   ```

4. If you plan to extract video features:

   * Install [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
   * Ensure it is built and accessible in your `$PATH`

---

## How to Run

### 1. Preprocessing and Training - Modality Specific

Each folder contains its own `README.md` with step-by-step instructions:

* [Text README](./Text/README.md)
* [Audio README](./Audio/README.md)
* [Video README](./Video/README.md)

### 2. Fusion Model

After extracting all features, train the fusion model:

```bash
cd Fusion

# Preprocess and load features
python3 load_features.py

# Train fusion model
python3 train.py

# Run inference (optional)
python3 inference.py
```

---

## Dataset

This project uses the **MUStARD++** dataset, an extension of the MUStARD sarcasm dataset with additional samples and annotations.

> Due to licensing, raw video and audio files are **not included**. Please download MUStARD++ dataset manually and place it in the `all_videos/` directory.

---

## Notes

* All modality-specific `extract_*.py` files are used for Phase 1 and Phase 2 separately.
* Some files are retained for completeness even if unused in the final model.

---

## Maintainer

* Mohammadhossein Narang
* mohammadhossein.narang@gmail.com

---

## License

This repository is for academic and non-commercial use only.

