from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import cv2
import librosa


# Global cache for bar noise background (loaded once)
_bar_noise_cache: Optional[torch.Tensor] = None

def _load_bar_noise(sample_rate: int = 16000) -> Optional[torch.Tensor]:
    """Load and cache bar background noise from data/Noise/noise.wav.
    
    Returns:
        torch.Tensor of shape [1, ...] or None if file not found
    """
    global _bar_noise_cache
    
    if _bar_noise_cache is not None:
        return _bar_noise_cache
    
    noise_path = Path("data") / "Noise" / "noise.wav"
    if not noise_path.exists():
        return None
    
    try:
        noise, sr = librosa.load(str(noise_path), sr=sample_rate, mono=True)
        _bar_noise_cache = torch.from_numpy(noise).float().unsqueeze(0)
        return _bar_noise_cache
    except Exception as e:
        print(f"[WARNING] Failed to load bar noise: {e}")
        return None


EMOTION_ID_TO_NAME = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def parse_ravdess_name(filename: str) -> Dict[str, int]:
    """
    Parse a RAVDESS filename into its 7 identifier fields.
    Expected pattern: 02-01-06-01-02-01-12(.ext)
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS name: {filename}")
    fields = list(map(int, parts))
    return {
        "modality": fields[0],
        "vocal_channel": fields[1],
        "emotion": fields[2],
        "intensity": fields[3],
        "statement": fields[4],
        "repetition": fields[5],
        "actor": fields[6],
    }


def _key_from_fields(fields: Dict[str, int]) -> Tuple[int, int, int, int, int, int]:
    return (
        fields["vocal_channel"],
        fields["emotion"],
        fields["intensity"],
        fields["statement"],
        fields["repetition"],
        fields["actor"],
    )


@dataclass
class PairRecord:
    video_path: Path
    audio_path: Path
    emotion: int
    intensity: int
    statement: int
    repetition: int
    actor: int

    def to_csv_row(self) -> List[str]:
        return [
            str(self.video_path),
            str(self.audio_path),
            str(self.emotion),
            str(self.intensity),
            str(self.statement),
            str(self.repetition),
            str(self.actor),
        ]


def build_pairs(
    data_root: Path,
    vocal_channel: int = 1,
) -> List[PairRecord]:
    """
    Build audio-video pairs from RAVDESS dataset.
    
    Uses modality=02 (video-only) + modality=03 (audio-only) files.
    
    RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    - Modality 02 = video-only (.mp4)
    - Modality 03 = audio-only (.wav)
    """
    pairs: List[PairRecord] = []
    data_root = Path(data_root)
    
    # Maps from (vocal_channel, emotion, intensity, statement, repetition, actor) to file paths
    video_map: Dict[Tuple[int, int, int, int, int, int], Path] = {}
    audio_map: Dict[Tuple[int, int, int, int, int, int], Path] = {}

    # Scan all files
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".wav"}:
            continue
        try:
            fields = parse_ravdess_name(path.name)
        except ValueError:
            continue
        if fields["vocal_channel"] != vocal_channel:
            continue
        key = _key_from_fields(fields)
        
        # Video-only: modality=02
        if fields["modality"] == 2 and path.suffix.lower() == ".mp4":
            video_map[key] = path
        # Audio-only: modality=03
        elif fields["modality"] == 3 and path.suffix.lower() == ".wav":
            audio_map[key] = path

    # Pair video-only + audio-only files
    for key in sorted(set(video_map.keys()) & set(audio_map.keys())):
        v = video_map[key]
        a = audio_map[key]
        # Parse fields from the key to get emotion, intensity, etc.
        fields_from_key = {
            "vocal_channel": key[0],
            "emotion": key[1],
            "intensity": key[2],
            "statement": key[3],
            "repetition": key[4],
            "actor": key[5],
        }
        pairs.append(
            PairRecord(
                video_path=v,
                audio_path=a,
                emotion=fields_from_key["emotion"],
                intensity=fields_from_key["intensity"],
                statement=fields_from_key["statement"],
                repetition=fields_from_key["repetition"],
                actor=fields_from_key["actor"],
            )
        )
    
    return pairs


def save_pairs_csv(pairs: Iterable[PairRecord], csv_path: Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video_path", "audio_path", "emotion", "intensity", "statement", "repetition", "actor"]
        )
        for p in pairs:
            writer.writerow(p.to_csv_row())


def map_emotion_label(emotion_id: int, num_classes: int) -> int:
    if num_classes == 8:
        return emotion_id - 1
    if num_classes != 4:
        raise ValueError("num_classes must be 8 or 4")
    if emotion_id in {1, 2}:
        return 0  # neutral/calm
    if emotion_id == 3:
        return 1  # positive (happy)
    if emotion_id in {4, 5, 6, 7}:
        return 2  # negative
    if emotion_id == 8:
        return 3  # surprise
    raise ValueError(f"Unknown emotion id: {emotion_id}")


def split_pairs_by_actor(
    pairs: List[PairRecord],
    train_actors: Iterable[int],
    val_actors: Iterable[int],
    test_actors: Iterable[int],
) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
    train_set = set(train_actors)
    val_set = set(val_actors)
    test_set = set(test_actors)
    train, val, test = [], [], []
    for p in pairs:
        if p.actor in train_set:
            train.append(p)
        elif p.actor in val_set:
            val.append(p)
        elif p.actor in test_set:
            test.append(p)
    return train, val, test


def split_pairs_stratified(
    pairs: List[PairRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
    """
    Stratified split ensuring balanced emotion distribution across train/val/test.
    
    Args:
        pairs: List of pair records
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    import random
    
    random.seed(seed)
    
    # Group pairs by emotion for stratified sampling
    emotion_groups: Dict[int, List[PairRecord]] = {}
    for p in pairs:
        if p.emotion not in emotion_groups:
            emotion_groups[p.emotion] = []
        emotion_groups[p.emotion].append(p)
    
    train, val, test = [], [], []
    
    # Split each emotion group independently
    for emotion, emotion_pairs in emotion_groups.items():
        random.shuffle(emotion_pairs)
        n = len(emotion_pairs)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train.extend(emotion_pairs[:train_size])
        val.extend(emotion_pairs[train_size:train_size + val_size])
        test.extend(emotion_pairs[train_size + val_size:])
    
    return train, val, test


def _uniform_indices(total: int, num: int) -> List[int]:
    if total <= 0:
        return [0] * num
    if total >= num:
        return np.linspace(0, total - 1, num=num).round().astype(int).tolist()
    return (list(range(total)) + [total - 1] * (num - total))


def load_video_frames(
    video_path: Path,
    num_frames: int = 8,
    size: int = 112,
    augment: bool = False,
    use_face_crop: bool = True,
) -> torch.Tensor:
    """Load and preprocess video frames with optional face detection and cropping.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        size: Target frame size (will be resized to size x size)
        augment: Whether to apply augmentation (blur, darken, noise)
        use_face_crop: Whether to detect and crop face regions
    
    Returns:
        Tensor [T, 3, H, W] normalized to ImageNet mean/std
    """
    # Lazy import face crop to avoid dependency if not needed
    if use_face_crop:
        try:
            from utils.face_crop import get_face_detector, crop_with_padding
        except (ImportError, ModuleNotFoundError):
            use_face_crop = False
    
    cap = cv2.VideoCapture(str(video_path))
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection and cropping (only on first sampled frame)
            if use_face_crop and bbox is None:
                try:
                    detector = get_face_detector()
                    if detector is not None:
                        # Temporarily convert back to BGR for detection
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        bbox = detector.detect_face_bbox(frame_bgr)
                        
                        if bbox is not None:
                            # Crop with padding on first frame; reuse bbox for others
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            frame_bgr = crop_with_padding(frame_bgr, bbox, pad_ratio=0.3)
                            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    # Silently fallback to full frame if detection fails
                    pass
            elif use_face_crop and bbox is not None:
                try:
                    # Apply same bbox crop to subsequent frames
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_bgr = crop_with_padding(frame_bgr, bbox, pad_ratio=0.3)
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    # Silently fallback to full frame if crop fails
                    pass
            
            # Resize to target size
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

    # Augmentation for low-light / noisy venue: blur and reduce brightness
    if augment:
        # Random brightness factor (darker)
        factor = float(np.random.uniform(0.2, 0.6))
        noise_scale = float(np.random.uniform(0.0, 0.0005))
        ksize = int(np.random.choice([3, 5, 7]))
        if ksize % 2 == 0:
            ksize += 1
        for i in range(frames.shape[0]):
            img = (frames[i] * 255.0).astype(np.uint8)
            # Apply Gaussian blur
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            img = img.astype(np.float32) / 255.0
            # Darken
            img = img * factor
            # Add slight Gaussian noise
            if noise_scale > 0:
                img = img + np.random.normal(0, noise_scale, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            frames[i] = img

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, 3, H, W]
    return frames


def load_audio_mel(
    audio_path: Path,
    sample_rate: int = 16000,
    duration_sec: float = 3.0,
    n_mels: int = 64,
    win_length: int = 400,
    hop_length: int = 160,
    augment: bool = False,
) -> torch.Tensor:
    # Load audio from .wav files using librosa
    wav, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    
    target_len = int(sample_rate * duration_sec)
    if wav.size(1) < target_len:
        pad = target_len - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.size(1) > target_len:
        wav = wav[:, :target_len]

    # Augment: mix real bar noise using time-domain fusion
    # Strategy distribution (when augment=True):
    #   50% - original clean audio (no augment)
    #   40% - light/medium noise (SNR: 20/15/10 dB)
    #   10% - heavy noise (SNR: 5 dB)
    if augment:
        augment_level = np.random.uniform(0.0, 1.0)
        
        if augment_level < 0.5:
            # 50% - Keep original clean audio
            pass
        else:
            # 40% light/medium + 10% heavy noise
            bar_noise = _load_bar_noise(sample_rate)
            
            # Determine SNR based on augment_level
            if augment_level < 0.9:
                # 40% - Light/medium noise (SNR: 20, 15, or 10)
                snr_options = [20.0, 15.0, 10.0]
                snr_db = float(np.random.choice(snr_options))
            else:
                # 10% - Heavy noise (SNR: 5)
                snr_db = 5.0
            
            if bar_noise is not None:
                # Time-domain fusion with real bar noise
                # Randomly segment bar noise to match duration
                if bar_noise.size(1) < target_len:
                    # Repeat if needed
                    repeats = (target_len // bar_noise.size(1)) + 1
                    bar_noise = bar_noise.repeat(1, repeats)
                
                # Random start position in noise
                max_start = max(0, bar_noise.size(1) - target_len)
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                noise_segment = bar_noise[:, start_idx:start_idx + target_len]
                
                # Compute SNR-controlled mixing
                # SNR = 10*log10(P_signal / P_noise)
                # => P_noise = P_signal / 10^(SNR/10)
                sig_np = wav.numpy()
                power_sig = np.mean(sig_np ** 2)
                snr_linear = 10 ** (snr_db / 10.0)
                power_noise_target = power_sig / max(snr_linear, 1e-8)
                
                # Scale noise to target power
                noise_np = noise_segment.numpy()
                power_noise_current = np.mean(noise_np ** 2)
                if power_noise_current > 1e-8:
                    scale_factor = np.sqrt(power_noise_target / power_noise_current)
                    noise_segment = noise_segment * scale_factor
                
                # Time-domain mixing: y(t) = s(t) + noise(t)
                wav = wav + noise_segment
                wav = torch.clamp(wav, -1.0, 1.0)
            else:
                # Fallback to Gaussian noise if real noise not available
                sig = wav.numpy()
                power_sig = np.mean(sig ** 2)
                snr = 10 ** (snr_db / 10.0)
                power_noise = power_sig / max(snr, 1e-8)
                noise = np.random.normal(0, np.sqrt(power_noise), sig.shape).astype(np.float32)
                wav = wav + torch.from_numpy(noise)
                wav = torch.clamp(wav, -1.0, 1.0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel_db


def load_audio_wav(
    audio_path: Path,
    sample_rate: int = 16000,
    duration_sec: float = 3.0,
    augment: bool = False,
) -> torch.Tensor:
    """Load raw audio waveform for WavLM finetuning.
    
    Args:
        audio_path: Path to .wav file
        sample_rate: Target sample rate (16kHz for WavLM)
        duration_sec: Duration in seconds (default 3.0)
        augment: Whether to apply real bar noise augmentation (time-domain fusion)
    
    Returns:
        torch.Tensor: Raw waveform [1, target_len]
    """
    wav, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    
    target_len = int(sample_rate * duration_sec)
    if wav.size(1) < target_len:
        pad = target_len - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.size(1) > target_len:
        wav = wav[:, :target_len]

    # Augmentation: mix real bar noise using time-domain fusion
    # Strategy distribution (when augment=True):
    #   50% - original clean audio (no augment)
    #   40% - light/medium noise (SNR: 20/15/10 dB)
    #   10% - heavy noise (SNR: 5 dB)
    if augment:
        augment_level = np.random.uniform(0.0, 1.0)
        
        if augment_level < 0.5:
            # 50% - Keep original clean audio
            pass
        else:
            # 40% light/medium + 10% heavy noise
            bar_noise = _load_bar_noise(sample_rate)
            
            # Determine SNR based on augment_level
            if augment_level < 0.9:
                # 40% - Light/medium noise (SNR: 20, 15, or 10)
                snr_options = [20.0, 15.0, 10.0]
                snr_db = float(np.random.choice(snr_options))
            else:
                # 10% - Heavy noise (SNR: 5)
                snr_db = 5.0
            
            if bar_noise is not None:
                # Time-domain fusion with real bar noise
                # Randomly segment bar noise to match duration
                if bar_noise.size(1) < target_len:
                    # Repeat if needed
                    repeats = (target_len // bar_noise.size(1)) + 1
                    bar_noise = bar_noise.repeat(1, repeats)
                
                # Random start position in noise
                max_start = max(0, bar_noise.size(1) - target_len)
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                noise_segment = bar_noise[:, start_idx:start_idx + target_len]
                
                # Compute SNR-controlled mixing
                sig_np = wav.numpy()
                power_sig = np.mean(sig_np ** 2)
                snr_linear = 10 ** (snr_db / 10.0)
                power_noise_target = power_sig / max(snr_linear, 1e-8)
                
                # Scale noise to target power
                noise_np = noise_segment.numpy()
                power_noise_current = np.mean(noise_np ** 2)
                if power_noise_current > 1e-8:
                    scale_factor = np.sqrt(power_noise_target / power_noise_current)
                    noise_segment = noise_segment * scale_factor
                
                # Time-domain mixing: y(t) = s(t) + noise(t)
                wav = wav + noise_segment
                wav = torch.clamp(wav, -1.0, 1.0)
            else:
                # Fallback to Gaussian noise if real noise not available
                sig = wav.numpy()
                power_sig = np.mean(sig ** 2)
                snr = 10 ** (snr_db / 10.0)
                power_noise = power_sig / max(snr, 1e-8)
                noise = np.random.normal(0, np.sqrt(power_noise), sig.shape).astype(np.float32)
                wav = wav + torch.from_numpy(noise)
                wav = torch.clamp(wav, -1.0, 1.0)

    return wav


class RavdessAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: List[PairRecord],
        num_classes: int = 8,
        num_frames: int = 8,
        augment: bool = False,
        use_face_crop: bool = True,
    ) -> None:
        self.pairs = pairs
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.augment = augment
        self.use_face_crop = use_face_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        video = load_video_frames(
            p.video_path,
            num_frames=self.num_frames,
            augment=self.augment,
            use_face_crop=self.use_face_crop,
        )
        audio = load_audio_mel(p.audio_path, augment=self.augment)
        label = map_emotion_label(p.emotion, self.num_classes)
        meta = {
            "emotion": p.emotion,
            "intensity": p.intensity,
            "statement": p.statement,
            "repetition": p.repetition,
            "actor": p.actor,
        }
        return video, audio, label, meta


class RavdessAVDatasetWavLM(torch.utils.data.Dataset):
    """Dataset for WavLM finetuning - returns raw waveforms instead of mel-spectrograms."""
    def __init__(self,
        pairs: List[PairRecord],
        num_classes: int = 8,
        num_frames: int = 8,
        augment: bool = False,
        use_face_crop: bool = True,
    ) -> None:
        self.pairs = pairs
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.augment = augment
        self.use_face_crop = use_face_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        video = load_video_frames(
            p.video_path,
            num_frames=self.num_frames,
            augment=self.augment,
            use_face_crop=self.use_face_crop,
        )
        audio = load_audio_wav(p.audio_path, augment=self.augment)  # Raw waveform [1, target_len]
        label = map_emotion_label(p.emotion, self.num_classes)
        meta = {
            "emotion": p.emotion,
            "intensity": p.intensity,
            "statement": p.statement,
            "repetition": p.repetition,
            "actor": p.actor,
        }
        return video, audio, label, meta


class RavdessPairService:
    """
    Service layer around pair parsing/building so callers can depend on an object
    instead of module-level functions.
    """

    def parse_name(self, filename: str) -> Dict[str, int]:
        return parse_ravdess_name(filename)

    def build_pairs(self, data_root: Path, vocal_channel: int = 1) -> List[PairRecord]:
        return build_pairs(data_root=data_root, vocal_channel=vocal_channel)

    def save_pairs_csv(self, pairs: Iterable[PairRecord], csv_path: Path) -> None:
        save_pairs_csv(pairs, csv_path)

    def map_emotion_label(self, emotion_id: int, num_classes: int) -> int:
        return map_emotion_label(emotion_id, num_classes)


class RavdessSplitService:
    """Split strategies for train/val/test partitioning."""

    def by_actor(
        self,
        pairs: List[PairRecord],
        train_actors: Iterable[int],
        val_actors: Iterable[int],
        test_actors: Iterable[int],
    ) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
        return split_pairs_by_actor(pairs, train_actors, val_actors, test_actors)

    def stratified(
        self,
        pairs: List[PairRecord],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
        return split_pairs_stratified(
            pairs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )


class RavdessMediaService:
    """Media preprocessing facade used by train/eval/export scripts."""

    def load_video_frames(
        self,
        video_path: Path,
        num_frames: int = 8,
        size: int = 112,
        augment: bool = False,
        use_face_crop: bool = True,
    ) -> torch.Tensor:
        return load_video_frames(
            video_path=video_path,
            num_frames=num_frames,
            size=size,
            augment=augment,
            use_face_crop=use_face_crop,
        )

    def load_audio_mel(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
        duration_sec: float = 3.0,
        n_mels: int = 64,
        win_length: int = 400,
        hop_length: int = 160,
        augment: bool = False,
    ) -> torch.Tensor:
        return load_audio_mel(
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            augment=augment,
        )

    def load_audio_wav(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
        duration_sec: float = 3.0,
        augment: bool = False,
    ) -> torch.Tensor:
        return load_audio_wav(
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            augment=augment,
        )


class RavdessDatasetFactory:
    """
    Dataset factory to centralize dataset class selection and make future
    dataset variants easier to add.
    """

    def __init__(self, media_service: Optional[RavdessMediaService] = None):
        self.media_service = media_service or RavdessMediaService()

    def create(
        self,
        pairs: List[PairRecord],
        num_classes: int = 8,
        num_frames: int = 8,
        augment: bool = False,
        use_face_crop: bool = True,
        use_wavlm: bool = False,
    ) -> torch.utils.data.Dataset:
        dataset_cls = RavdessAVDatasetWavLM if use_wavlm else RavdessAVDataset
        return dataset_cls(
            pairs=pairs,
            num_classes=num_classes,
            num_frames=num_frames,
            augment=augment,
            use_face_crop=use_face_crop,
        )


# Default service instances for object-oriented usage.
PAIR_SERVICE = RavdessPairService()
SPLIT_SERVICE = RavdessSplitService()
MEDIA_SERVICE = RavdessMediaService()
DATASET_FACTORY = RavdessDatasetFactory(media_service=MEDIA_SERVICE)
