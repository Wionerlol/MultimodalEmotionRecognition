"""Configuration constants for backend."""
import os
from pathlib import Path

# Recording and preprocessing
RECORD_SECONDS = 3
FRAMES = 8
IMG_SIZE = 112
SAMPLE_RATE = 16000
N_MELS = 64
WIN_LENGTH = 400
HOP_LENGTH = 160

# Model checkpoint
CHECKPOINT_PATH = Path("./checkpoints/best.pt")

# Emotion labels (8-class RAVDESS)
EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# Feature dims (match model architecture)
VIDEO_EMBEDDING_DIM = 512  # from ResNet18
AUDIO_EMBEDDING_DIM = 128  # from AudioNet
FUSION_COMMON_DIM = 256

# Inference settings
DEVICE = "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu"
MOCK_MODE = os.environ.get("EMO_MOCK", "0") == "1"

# Paths
TEMP_DIR = Path("./tmp")
TEMP_DIR.mkdir(exist_ok=True)
