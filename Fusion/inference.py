# inference.py
import torch
from fusion_model import CrossModalFusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossModalFusionModel().to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Example input tensors (replace with real data or loader)
text_input = torch.randn(1, 768).to(device)
audio_input = torch.randn(1, 1, 768).to(device)
video_input = torch.randn(1, 23).to(device)

if audio_input.dim() == 3 and audio_input.shape[1] == 1:
    audio_input = audio_input.squeeze(1)

with torch.no_grad():
    output = model(text_input, audio_input, video_input)
    prediction = output.argmax(dim=1).item()

print(f"🧠 Predicted class: {prediction}")