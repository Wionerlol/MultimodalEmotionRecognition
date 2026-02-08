"""Model loading and checkpoint handling."""
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch import nn

from .config import CHECKPOINT_PATH, DEVICE


def _build_fusion_model(num_classes: int = 8) -> nn.Module:
    """
    Build a FusionModel for emotion recognition.
    This requires the model definitions from src/models.
    """
    try:
        # Try to import from the main src directory if available
        # For Docker, the models are not available, so we return None
        # and let the backup dummy model handle it
        return None
    except ImportError:
        return None


def load_model_from_checkpoint(checkpoint_path: Path, num_classes: int = 8) -> Optional[nn.Module]:
    """
    Load PyTorch model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        num_classes: Number of emotion classes
    
    Returns:
        Loaded model or None if checkpoint not found
    
    Raises:
        RuntimeError: If checkpoint cannot be loaded
    """
    if not checkpoint_path.exists():
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint
        
        # Try to build the model and load state
        model = _build_fusion_model(num_classes)
        if model is not None:
            model.load_state_dict(model_state)
            return model
        
        # If we can't build the model, just return a flag
        # that the checkpoint exists but couldn't be loaded
        return "checkpoint_exists_but_model_unavailable"
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")


class DummyModel(nn.Module):
    """Placeholder model when checkpoint is not available."""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.dummy = nn.Linear(1, 1)
    
    def forward(self, *args, **kwargs):
        raise RuntimeError("Model not loaded. Please provide checkpoint at ./checkpoints/best.pt or set EMO_MOCK=1")


def get_model(num_classes: int = 8, allow_mock: bool = False) -> nn.Module:
    """
    Get emotion classification model.
    
    Args:
        num_classes: Number of emotion classes (default 8)
        allow_mock: If True and checkpoint missing, return dummy model (for testing)
    
    Returns:
        PyTorch model (loaded from checkpoint or dummy)
    
    Raises:
        RuntimeError: If checkpoint missing and mock not allowed
    """
    model_result = load_model_from_checkpoint(CHECKPOINT_PATH, num_classes)
    
    # Case 1: Checkpoint loaded successfully
    if isinstance(model_result, nn.Module):
        return model_result
    
    # Case 2: Checkpoint exists but couldn't be loaded (model classes unavailable)
    if model_result == "checkpoint_exists_but_model_unavailable":
        if allow_mock:
            print("[INFO] Checkpoint exists but model classes unavailable. Using mock mode.")
            return DummyModel(num_classes=num_classes)
        else:
            raise RuntimeError(
                f"Checkpoint found at {CHECKPOINT_PATH} but model classes are not available in Docker. "
                "Set EMO_MOCK=1 environment variable to run in mock mode for testing."
            )
    
    # Case 3: No checkpoint found
    if model_result is None:
        if allow_mock:
            print(f"[INFO] Checkpoint not found at {CHECKPOINT_PATH}. Running in mock mode.")
            return DummyModel(num_classes=num_classes)
        else:
            raise RuntimeError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. "
                "Please provide a checkpoint, or set EMO_MOCK=1 environment variable for testing."
            )
