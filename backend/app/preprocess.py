"""Video and audio preprocessing for emotion recognition."""
import os
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import librosa
import numpy as np
import torch
import torchaudio

from .config import (
    FRAMES,
    IMG_SIZE,
    N_MELS,
    SAMPLE_RATE,
    HOP_LENGTH,
    WIN_LENGTH,
    RECORD_SECONDS,
)


def _uniform_indices(total: int, num: int) -> list:
    """Uniformly sample frame indices."""
    if total <= 0:
        return [0] * num
    if total >= num:
        return np.linspace(0, total - 1, num=num).round().astype(int).tolist()
    return (list(range(total)) + [total - 1] * (num - total))


def _get_face_detector():
    """Lazy load MediaPipe face detector."""
    try:
        # Try modern API first
        try:
            from mediapipe.tasks import vision
            from mediapipe import core as media_core
            
            if not hasattr(_get_face_detector, '_instance'):
                base_options = media_core.BaseOptions(model_asset_path=None)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=0.5
                )
                _get_face_detector._instance = vision.FaceDetector.create_from_options(options)
                _get_face_detector._api_type = "modern"
            return _get_face_detector._instance
        
        except (ImportError, AttributeError):
            # Fallback to legacy API
            try:
                from mediapipe.python.solutions import face_detection
            except ImportError:
                import mediapipe.solutions.face_detection as face_detection
            
            if not hasattr(_get_face_detector, '_instance'):
                _get_face_detector._instance = face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                _get_face_detector._api_type = "legacy"
            return _get_face_detector._instance
    
    except ImportError:
        return None


def _detect_face_bbox(frame: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Detect face in frame using MediaPipe. Returns (x1, y1, x2, y2) or None."""
    detector = _get_face_detector()
    if detector is None:
        return None
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        api_type = getattr(_get_face_detector, '_api_type', 'unknown')
        
        if api_type == "modern":
            # Modern API
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = detector.detect(mp_image)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.bounding_box
                x1 = max(0, int(bbox.origin_x))
                y1 = max(0, int(bbox.origin_y))
                x2 = min(w, int(bbox.origin_x + bbox.width))
                y2 = min(h, int(bbox.origin_y + bbox.height))
                return (x1, y1, x2, y2)
        
        elif api_type == "legacy":
            # Legacy API
            results = detector.process(frame_rgb)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                return (x1, y1, x2, y2)
    
    except Exception:
        pass
    
    return None


def _crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.3,
) -> np.ndarray:
    """Crop face region with padding."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    pad_x = int(bbox_w * pad_ratio)
    pad_y = int(bbox_h * pad_ratio)
    
    x1_padded = max(0, x1 - pad_x)
    y1_padded = max(0, y1 - pad_y)
    x2_padded = min(w, x2 + pad_x)
    y2_padded = min(h, y2 + pad_y)
    
    return frame[y1_padded:y2_padded, x1_padded:x2_padded]


def load_video_frames(
    video_path: str,
    num_frames: int = FRAMES,
    size: int = IMG_SIZE,
    use_face_crop: bool = True,
) -> torch.Tensor:
    """
    Extract frames from video and preprocess.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default 8)
        size: Image size (default 112)
        use_face_crop: Whether to detect and crop face region (default True)
    
    Returns:
        Tensor of shape [T, 3, H, W] with ImageNet normalization applied
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_indices(total, num_frames)
    frames = []
    idx_set = set(indices)
    current = 0
    grabbed = 0
    
    bbox = None  # Will be detected on first frame if use_face_crop=True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in idx_set:
            # Face detection and cropping (only on first sampled frame)
            if use_face_crop and bbox is None:
                bbox = _detect_face_bbox(frame)
                if bbox is not None:
                    frame = _crop_with_padding(frame, bbox, pad_ratio=0.3)
            elif use_face_crop and bbox is not None:
                # Apply same bbox crop to subsequent frames
                frame = _crop_with_padding(frame, bbox, pad_ratio=0.3)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            grabbed += 1
        current += 1
        if grabbed >= len(indices):
            break
    cap.release()

    if not frames:
        frames = [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(num_frames)]
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    
    frames = np.stack(frames[:num_frames], axis=0).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, 3, H, W]
    return frames


def load_audio_mel(
    audio_path: str,
    sample_rate: int = SAMPLE_RATE,
    duration_sec: float = RECORD_SECONDS,
    n_mels: int = N_MELS,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """
    Load audio and compute log-mel spectrogram.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16000)
        duration_sec: Duration to load (default 3)
        n_mels: Number of mel bands (default 64)
        win_length: Window length for STFT (default 400)
        hop_length: Hop length for STFT (default 160)
    
    Returns:
        Tensor of shape [1, n_mels, T] (log-mel spectrogram)
    """
    wav, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    
    target_len = int(sample_rate * duration_sec)
    if wav.size(1) < target_len:
        pad = target_len - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.size(1) > target_len:
        wav = wav[:, :target_len]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel_db


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video using ffmpeg and save as WAV.
    
    Args:
        video_path: Path to input video
    
    Returns:
        Path to extracted WAV file
    
    Raises:
        RuntimeError: If ffmpeg is not available or extraction fails
    """
    fd, audio_path = tempfile.mkstemp(suffix=".wav", dir="./tmp")
    os.close(fd)
    
    cmd = f'ffmpeg -i "{video_path}" -q:a 9 -n "{audio_path}" -y 2>&1'
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed with code {exit_code}")
    
    return audio_path


def preprocess_video_audio(video_path: str, use_face_crop: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess video and audio from a single file.
    
    Args:
        video_path: Path to video file
        use_face_crop: Whether to detect and crop face region (default True)
    
    Returns:
        Tuple of (video_tensor, audio_tensor)
        - video: [1, T, 3, H, W] (add batch dim)
        - audio: [1, 1, n_mels, Ta] (add batch and channel dims)
    """
    # Extract video frames with optional face cropping
    video = load_video_frames(video_path, num_frames=FRAMES, size=IMG_SIZE, use_face_crop=use_face_crop)
    video = video.unsqueeze(0)  # [1, T, 3, H, W]
    
    # Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    try:
        audio = load_audio_mel(audio_path, sample_rate=SAMPLE_RATE, duration_sec=RECORD_SECONDS)
        audio = audio.unsqueeze(0)  # [1, 1, n_mels, Ta]
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    return video, audio
