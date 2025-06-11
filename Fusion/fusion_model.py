# fusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusionModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, video_dim=23, hidden_dim=256, num_classes=2, dropout=0.3):
        super(CrossModalFusionModel, self).__init__()

        # Projection layers to a common dimensionality
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, audio, video):
        """
        Args:
            text: Tensor of shape [batch_size, text_dim]
            audio: Tensor of shape [batch_size, audio_dim] or [batch_size, 1, audio_dim]
            video: Tensor of shape [batch_size, video_dim]
        """
        # Flatten audio input if necessary
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        # Apply modality-specific projections
        text_feat = self.text_proj(text)
        audio_feat = self.audio_proj(audio)
        video_feat = self.video_proj(video)

        # Fuse features
        fused = torch.cat([text_feat, audio_feat, video_feat], dim=1)
        fused = self.fusion(fused)

        # Final classification
        output = self.classifier(fused)
        return output
        
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusionModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, video_dim=23, hidden_dim=256, num_classes=2, dropout=0.3, num_heads=4):
        super(CrossModalAttentionFusionModel, self).__init__()

        # Projection layers to common dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Cross-attention layers
        # Text queries attend to audio and video keys/values
        self.cross_attn_text_audio = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_text_video = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Final fusion layer after attention
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, audio, video):
        """
        Args:
            text: Tensor [batch_size, text_dim]
            audio: Tensor [batch_size, audio_dim] or [batch_size, 1, audio_dim]
            video: Tensor [batch_size, video_dim]
        Returns:
            logits: Tensor [batch_size, num_classes]
        """

        # Flatten audio input if necessary
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        # Project each modality
        text_feat = self.text_proj(text)   # [B, hidden_dim]
        audio_feat = self.audio_proj(audio) # [B, hidden_dim]
        video_feat = self.video_proj(video) # [B, hidden_dim]

        # Add sequence dimension for attention: [B, hidden_dim] -> [B, 1, hidden_dim]
        text_feat = text_feat.unsqueeze(1)  # query shape
        audio_feat = audio_feat.unsqueeze(1) # key & value shape
        video_feat = video_feat.unsqueeze(1) # key & value shape

        # Cross-attention: text queries attend to audio & video modalities
        text_audio_attended, _ = self.cross_attn_text_audio(query=text_feat, key=audio_feat, value=audio_feat)
        text_video_attended, _ = self.cross_attn_text_video(query=text_feat, key=video_feat, value=video_feat)

        # Fuse attended features by summing (or you could concat + linear)
        fused = text_feat + text_audio_attended + text_video_attended  # [B, 1, hidden_dim]

        fused = fused.squeeze(1)  # [B, hidden_dim]

        # Pass through fusion MLP
        fused = self.fusion(fused)  # [B, hidden_dim]

        # Final classification
        logits = self.classifier(fused)  # [B, num_classes]

        return logits
        
'''