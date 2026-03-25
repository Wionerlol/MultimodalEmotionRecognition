"""Inference pipeline for emotion recognition."""
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE, EMOTION_LABELS, MOCK_MODE
from .model_loader import get_model
from .preprocess import preprocess_video_audio, preprocess_stream_window


class EmotionPredictor:
    """Wrapper for emotion prediction."""
    
    def __init__(self, mock_mode: bool = MOCK_MODE):
        self.mock_mode = mock_mode
        self.emotion_labels = EMOTION_LABELS
        self.use_wavlm = False
        if not mock_mode:
            try:
                self.model = get_model(num_classes=len(EMOTION_LABELS), allow_mock=False)
                self.use_wavlm = bool(getattr(self.model, "requires_wavlm", False))
                self.model.to(DEVICE)
                self.model.eval()
            except RuntimeError as e:
                print(f"[WARNING] {e}")
                print("[WARNING] Falling back to mock mode")
                self.mock_mode = True
                self.model = None
        else:
            self.model = None
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict emotion from video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with keys:
            - labels: list of emotion label strings
            - probs: list of probabilities (0-100)
            - top1: dict with "label" and "prob"
        """
        if self.mock_mode:
            return self._predict_mock()
        
        try:
            video, audio = preprocess_video_audio(video_path, use_face_crop=True, use_wavlm=self.use_wavlm)
            return self.predict_tensors(video, audio)
        
        except Exception as e:
            # Fallback to mock on inference error
            return {
                "error": str(e),
                "labels": self.emotion_labels,
                "probs": [1.0 / len(self.emotion_labels) * 100] * len(self.emotion_labels),
                "top1": {"label": self.emotion_labels[0], "prob": 1.0 / len(self.emotion_labels) * 100}
            }
    
    def predict_stream(
        self,
        frames: list[np.ndarray],
        waveform: np.ndarray,
        waveform_sample_rate: int,
        use_face_crop: bool = True,
    ) -> Dict[str, Any]:
        """Predict from an in-memory sliding window."""
        if self.mock_mode:
            return self._predict_mock()

        try:
            video, audio = preprocess_stream_window(
                frames,
                waveform,
                waveform_sample_rate=waveform_sample_rate,
                use_face_crop=use_face_crop,
                use_wavlm=self.use_wavlm,
            )
            return self.predict_tensors(video, audio)
        except Exception as e:
            return {
                "error": str(e),
                "labels": self.emotion_labels,
                "probs": [1.0 / len(self.emotion_labels) * 100] * len(self.emotion_labels),
                "top1": {"label": self.emotion_labels[0], "prob": 1.0 / len(self.emotion_labels) * 100},
            }

    def predict_tensors(self, video: torch.Tensor, audio: torch.Tensor) -> Dict[str, Any]:
        """Run model inference on preprocessed tensors."""
        if self.mock_mode:
            return self._predict_mock()
        video = video.to(DEVICE)
        audio = audio.to(DEVICE)
        with torch.no_grad():
            logits = self.model(video, audio)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        return self._format_output(probs)

    def _predict_mock(self) -> Dict[str, Any]:
        """Generate random predictions for testing."""
        probs = np.random.dirichlet(np.ones(len(self.emotion_labels)))
        return self._format_output(probs)
    
    def _format_output(self, probs: np.ndarray) -> Dict[str, Any]:
        """Format probability output."""
        probs_pct = (probs * 100).tolist()
        top_idx = np.argmax(probs)
        return {
            "labels": self.emotion_labels,
            "probs": probs_pct,
            "top1": {
                "label": self.emotion_labels[top_idx],
                "prob": probs_pct[top_idx],
            }
        }
