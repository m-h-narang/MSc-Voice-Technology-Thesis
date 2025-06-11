# multimodal_dataset.py
import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, text_feats, audio_feats, visual_feats, labels, device=torch.device('cpu')):
        assert len(text_feats) == len(audio_feats) == len(visual_feats) == len(labels), "Length mismatch"

        self.text = text_feats
        self.audio = audio_feats
        self.visual = visual_feats
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.text[idx].to(self.device),
            self.audio[idx].to(self.device),
            self.visual[idx].to(self.device),
            self.labels[idx].to(self.device),
        )