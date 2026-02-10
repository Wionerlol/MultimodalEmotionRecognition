"""Configuration constants for backend."""
import os
import platform
from pathlib import Path

import torch

# Recording and preprocessing
RECORD_SECONDS = 3
FRAMES = 8
IMG_SIZE = 112
SAMPLE_RATE = 16000
N_MELS = 64
WIN_LENGTH = 400
HOP_LENGTH = 160

_THIS_FILE = Path(__file__).resolve()
_CANDIDATE_ROOTS = [_THIS_FILE.parents[2], _THIS_FILE.parents[1]]
PROJECT_ROOT = next(
    (p for p in _CANDIDATE_ROOTS if (p / "src").exists() or (p / "checkpoints").exists()),
    _THIS_FILE.parents[2],
)
BACKEND_ROOT = _THIS_FILE.parents[1]

# Model checkpoint
CHECKPOINT_PATH = Path(
    os.environ.get("CHECKPOINT_PATH", str(PROJECT_ROOT / "checkpoints" / "best.pt"))
).expanduser()

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
USE_GPU_ENV = os.environ.get("USE_GPU", "auto").strip().lower()
CUDA_READY = torch.cuda.is_available()
if USE_GPU_ENV in {"1", "true", "yes"}:
    DEVICE = "cuda" if CUDA_READY else "cpu"
elif USE_GPU_ENV in {"0", "false", "no"}:
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if CUDA_READY else "cpu"

IS_WSL = bool(os.environ.get("WSL_DISTRO_NAME")) or ("microsoft" in platform.release().lower())
MOCK_MODE = os.environ.get("EMO_MOCK", "0") == "1"

# Paths
TEMP_DIR = Path(os.environ.get("TEMP_DIR", str(PROJECT_ROOT / "tmp"))).expanduser()
TEMP_DIR.mkdir(parents=True, exist_ok=True)
