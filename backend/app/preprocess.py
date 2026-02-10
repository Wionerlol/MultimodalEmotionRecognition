"""Video and audio preprocessing for emotion recognition."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import librosa
import numpy as np
import torch
import torchaudio

from .config import (
    FRAMES,
    HOP_LENGTH,
    IMG_SIZE,
    N_MELS,
    RECORD_SECONDS,
    SAMPLE_RATE,
    TEMP_DIR,
    WIN_LENGTH,
)


class EmotionPreprocessService:
    """Service object for backend video/audio preprocessing."""

    def __init__(
        self,
        frames: int = FRAMES,
        img_size: int = IMG_SIZE,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        record_seconds: float = RECORD_SECONDS,
        temp_dir: Path = TEMP_DIR,
    ) -> None:
        self.frames = frames
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.record_seconds = record_seconds
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._face_detector = None
        self._face_detector_api_type = "unknown"

    @staticmethod
    def uniform_indices(total: int, num: int) -> list:
        """Uniformly sample frame indices."""
        if total <= 0:
            return [0] * num
        if total >= num:
            return np.linspace(0, total - 1, num=num).round().astype(int).tolist()
        return list(range(total)) + [total - 1] * (num - total)

    def get_face_detector(self):
        """Lazy load MediaPipe face detector."""
        if self._face_detector is not None:
            return self._face_detector

        try:
            try:
                from mediapipe import core as media_core
                from mediapipe.tasks import vision

                base_options = media_core.BaseOptions(model_asset_path=None)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=0.5,
                )
                self._face_detector = vision.FaceDetector.create_from_options(options)
                self._face_detector_api_type = "modern"
                return self._face_detector
            except (ImportError, AttributeError):
                try:
                    from mediapipe.python.solutions import face_detection
                except ImportError:
                    import mediapipe.solutions.face_detection as face_detection

                self._face_detector = face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5,
                )
                self._face_detector_api_type = "legacy"
                return self._face_detector
        except ImportError:
            return None

    def detect_face_bbox(self, frame: np.ndarray) -> Tuple[int, int, int, int] | None:
        """Detect face in frame using MediaPipe. Returns (x1, y1, x2, y2) or None."""
        detector = self.get_face_detector()
        if detector is None:
            return None

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            if self._face_detector_api_type == "modern":
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

            elif self._face_detector_api_type == "legacy":
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

    @staticmethod
    def crop_with_padding(
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
        self,
        video_path: str,
        num_frames: int = FRAMES,
        size: int = IMG_SIZE,
        use_face_crop: bool = True,
    ) -> torch.Tensor:
        """Extract frames from video and preprocess."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.uniform_indices(total, num_frames)
        frames = []
        idx_set = set(indices)
        current = 0
        grabbed = 0
        bbox = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current in idx_set:
                if use_face_crop and bbox is None:
                    bbox = self.detect_face_bbox(frame)
                    if bbox is not None:
                        frame = self.crop_with_padding(frame, bbox, pad_ratio=0.3)
                elif use_face_crop and bbox is not None:
                    frame = self.crop_with_padding(frame, bbox, pad_ratio=0.3)

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

        frames_np = np.stack(frames[:num_frames], axis=0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frames_np = (frames_np - mean) / std
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # [T, 3, H, W]

    def load_audio_mel(
        self,
        audio_path: str,
        sample_rate: int = SAMPLE_RATE,
        duration_sec: float = RECORD_SECONDS,
        n_mels: int = N_MELS,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
    ) -> torch.Tensor:
        """Load audio and compute log-mel spectrogram."""
        wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
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
        return torchaudio.transforms.AmplitudeToDB()(mel)

    def load_audio_wav(
        self,
        audio_path: str,
        sample_rate: int = SAMPLE_RATE,
        duration_sec: float = RECORD_SECONDS,
    ) -> torch.Tensor:
        """Load raw waveform for WavLM-based models."""
        wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        wav = torch.from_numpy(wav).float().unsqueeze(0)

        target_len = int(sample_rate * duration_sec)
        if wav.size(1) < target_len:
            pad = target_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.size(1) > target_len:
            wav = wav[:, :target_len]
        return wav

    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video using ffmpeg and save as WAV.
        """
        fd, audio_path = tempfile.mkstemp(suffix=".wav", dir=str(self.temp_dir))
        os.close(fd)

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg in this environment.")

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-q:a",
            "9",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed with code {result.returncode}: {result.stderr.strip()}"
            )
        return audio_path

    def preprocess_video_audio(
        self,
        video_path: str,
        use_face_crop: bool = True,
        use_wavlm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess video and audio from a single file.
        Returns (video[1,T,3,H,W], audio[1,1,n_mels,T] or [1,1,T]).
        """
        video = self.load_video_frames(video_path, num_frames=self.frames, size=self.img_size, use_face_crop=use_face_crop)
        video = video.unsqueeze(0)

        audio_path = self.extract_audio_from_video(video_path)
        try:
            if use_wavlm:
                audio = self.load_audio_wav(audio_path, sample_rate=self.sample_rate, duration_sec=self.record_seconds)
            else:
                audio = self.load_audio_mel(audio_path, sample_rate=self.sample_rate, duration_sec=self.record_seconds)
            audio = audio.unsqueeze(0)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        return video, audio


PREPROCESS_SERVICE = EmotionPreprocessService()


# Compatibility wrappers (keep old imports working).
def _uniform_indices(total: int, num: int) -> list:
    return PREPROCESS_SERVICE.uniform_indices(total, num)


def _get_face_detector():
    return PREPROCESS_SERVICE.get_face_detector()


def _detect_face_bbox(frame: np.ndarray) -> Tuple[int, int, int, int] | None:
    return PREPROCESS_SERVICE.detect_face_bbox(frame)


def _crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.3,
) -> np.ndarray:
    return PREPROCESS_SERVICE.crop_with_padding(frame, bbox, pad_ratio=pad_ratio)


def load_video_frames(
    video_path: str,
    num_frames: int = FRAMES,
    size: int = IMG_SIZE,
    use_face_crop: bool = True,
) -> torch.Tensor:
    return PREPROCESS_SERVICE.load_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        size=size,
        use_face_crop=use_face_crop,
    )


def load_audio_mel(
    audio_path: str,
    sample_rate: int = SAMPLE_RATE,
    duration_sec: float = RECORD_SECONDS,
    n_mels: int = N_MELS,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    return PREPROCESS_SERVICE.load_audio_mel(
        audio_path=audio_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
    )


def load_audio_wav(
    audio_path: str,
    sample_rate: int = SAMPLE_RATE,
    duration_sec: float = RECORD_SECONDS,
) -> torch.Tensor:
    return PREPROCESS_SERVICE.load_audio_wav(
        audio_path=audio_path,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
    )


def extract_audio_from_video(video_path: str) -> str:
    return PREPROCESS_SERVICE.extract_audio_from_video(video_path)


def preprocess_video_audio(
    video_path: str,
    use_face_crop: bool = True,
    use_wavlm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return PREPROCESS_SERVICE.preprocess_video_audio(
        video_path=video_path,
        use_face_crop=use_face_crop,
        use_wavlm=use_wavlm,
    )
