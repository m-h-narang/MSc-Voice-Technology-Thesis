import os
import torch
import torchaudio
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# --- CONFIG ---
VIDEO_DIR = "/scratch/s6028608/MUStARD_Plus_Plus/all_videos/final_utterance_videos"
AUDIO_DIR = os.path.join(VIDEO_DIR, "audio_wavs")
FEATURE_DIR = os.path.join(VIDEO_DIR, "wav2vec_features")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# --- Load pretrained wav2vec 2.0 model ---
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().eval()

def extract_audio(video_path, audio_path):
    """Extract audio from video and save as WAV."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000, verbose=False, logger=None)

def extract_features(audio_path):
    """Extract wav2vec features from audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    with torch.inference_mode():
        features = model(waveform)[0]  # Shape: (1, num_frames, feature_dim)
    return features.squeeze(0)

# --- Main loop ---
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

for video_file in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(VIDEO_DIR, video_file)
    base_name = os.path.splitext(video_file)[0]

    audio_path = os.path.join(AUDIO_DIR, base_name + ".wav")
    feature_path = os.path.join(FEATURE_DIR, base_name + ".pt")

    try:
        extract_audio(video_path, audio_path)
        features = extract_features(audio_path)
        torch.save(features, feature_path)
    except Exception as e:
        print(f"[ERROR] Failed on {video_file}: {e}")