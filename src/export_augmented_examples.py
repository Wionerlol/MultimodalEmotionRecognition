from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import soundfile as sf
import torch

from data.ravdess import MEDIA_SERVICE, PAIR_SERVICE


def _save_augmented_video(video_tensor: torch.Tensor, output_path: Path, fps: int = 4) -> Path:
    """
    Save augmented video frames tensor [T, 3, H, W] to a video file.
    Returns the actual saved path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Undo ImageNet normalization from load_video_frames
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = video_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    frames = np.clip(frames * std + mean, 0.0, 1.0)
    frames = (frames * 255.0).astype(np.uint8)

    h, w = frames.shape[1], frames.shape[2]

    # Try mp4 first; fallback to avi if codec unavailable.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    actual_output = output_path
    if not writer.isOpened():
        actual_output = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(
            str(actual_output),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(fps),
            (w, h),
        )

    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer (mp4v/MJPG both unavailable).")

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return actual_output


def _save_augmented_audio(audio_wave: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    """
    Save augmented waveform tensor [1, T] to wav.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wave = audio_wave.detach().cpu().squeeze(0).numpy()
    sf.write(str(output_path), wave, sample_rate)


def _uniform_indices(total: int, num: int) -> List[int]:
    if total <= 0:
        return [0] * num
    if total >= num:
        return np.linspace(0, total - 1, num=num).round().astype(int).tolist()
    return list(range(total)) + [total - 1] * (num - total)


def _load_video_frames_visual(
    video_path: Path,
    num_frames: int,
    target_long_side: int,
    downscale_ratio: float,
    noise_scale: float,
    brightness_factor: float,
) -> np.ndarray:
    """
    Load frames for visual QA (high-resolution) and apply realistic degradation:
    downsample->upsample + optional noise + brightness.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(_uniform_indices(total, num_frames))

    frames = []
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            long_side = max(h, w)
            scale = float(target_long_side) / float(long_side)
            if scale > 0 and scale != 1.0:
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Simulate "1080p looks like 720p": downsample then upsample.
            h2, w2 = frame.shape[:2]
            ds_w = max(1, int(round(w2 * downscale_ratio)))
            ds_h = max(1, int(round(h2 * downscale_ratio)))
            frame = cv2.resize(frame, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_LINEAR)

            img = frame.astype(np.float32) / 255.0
            img = img * brightness_factor
            if noise_scale > 0:
                img = img + np.random.normal(0, noise_scale, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            frames.append((img * 255.0).astype(np.uint8))
        current += 1
        if len(frames) >= num_frames:
            break
    cap.release()

    if not frames:
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(num_frames)]
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return np.stack(frames[:num_frames], axis=0)


def _save_rgb_frames_video(frames_rgb: np.ndarray, output_path: Path, fps: int = 4) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb.shape[1], frames_rgb.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    actual_output = output_path
    if not writer.isOpened():
        actual_output = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(actual_output), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer (mp4v/MJPG both unavailable).")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return actual_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one augmented audio/video example to data/examples.")
    parser.add_argument("--data_root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="data/examples", help="Output directory")
    parser.add_argument("--index", type=int, default=-1, help="Pair index to export (-1 means random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--frames", type=int, default=8, help="Number of sampled frames")
    parser.add_argument("--size", type=int, default=112, help="Frame size")
    parser.add_argument("--fps", type=int, default=4, help="Saved video fps")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration_sec", type=float, default=3.0, help="Audio duration in seconds")
    parser.add_argument("--no_face_crop", action="store_true", help="Disable face crop in video preprocessing")
    parser.add_argument(
        "--visual_mode",
        action="store_true",
        help="Use high-resolution visual QA mode (downsample->upsample) instead of training 112x112 pipeline",
    )
    parser.add_argument("--visual_long_side", type=int, default=1080, help="Long side in visual mode (e.g. 1080)")
    parser.add_argument(
        "--visual_downscale_ratio",
        type=float,
        default=2.0 / 3.0,
        help="Downsample ratio in visual mode (2/3 approximates 1080p->720p)",
    )
    parser.add_argument("--visual_noise_scale", type=float, default=0.0003, help="Noise std in visual mode")
    parser.add_argument("--visual_brightness", type=float, default=1.0, help="Brightness factor in visual mode")
    return parser


class AugmentedExampleExporter:
    """Export one augmented multimodal sample for augmentation QA and tuning."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> None:
        args = self.args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        data_root = Path(args.data_root)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = PAIR_SERVICE.build_pairs(data_root)
        if not pairs:
            raise RuntimeError(f"No audio-video pairs found under: {data_root}")

        if args.index >= 0:
            if args.index >= len(pairs):
                raise IndexError(f"index={args.index} out of range (num_pairs={len(pairs)})")
            idx = args.index
        else:
            idx = random.randrange(len(pairs))

        pair = pairs[idx]

        if args.visual_mode:
            frames_rgb = _load_video_frames_visual(
                pair.video_path,
                num_frames=args.frames,
                target_long_side=args.visual_long_side,
                downscale_ratio=args.visual_downscale_ratio,
                noise_scale=args.visual_noise_scale,
                brightness_factor=args.visual_brightness,
            )
        else:
            aug_video = MEDIA_SERVICE.load_video_frames(
                pair.video_path,
                num_frames=args.frames,
                size=args.size,
                augment=True,
                use_face_crop=(not args.no_face_crop),
            )
        aug_audio = MEDIA_SERVICE.load_audio_wav(
            pair.audio_path,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            augment=True,
        )

        video_out = output_dir / "augmented_video_example.mp4"
        audio_out = output_dir / "augmented_audio_example.wav"
        meta_out = output_dir / "example_meta.json"

        if args.visual_mode:
            actual_video_out = _save_rgb_frames_video(frames_rgb, video_out, fps=args.fps)
        else:
            actual_video_out = _save_augmented_video(aug_video, video_out, fps=args.fps)
        _save_augmented_audio(aug_audio, args.sample_rate, audio_out)

        meta = {
            "source_video": str(pair.video_path),
            "source_audio": str(pair.audio_path),
            "selected_pair_index": idx,
            "num_pairs_total": len(pairs),
            "seed": args.seed,
            "video": {
                "augment": True,
                "num_frames": args.frames,
                "size": args.size,
                "fps": args.fps,
                "use_face_crop": (not args.no_face_crop),
                "visual_mode": args.visual_mode,
                "visual_long_side": args.visual_long_side,
                "visual_downscale_ratio": args.visual_downscale_ratio,
                "visual_noise_scale": args.visual_noise_scale,
                "visual_brightness": args.visual_brightness,
                "output": str(actual_video_out),
            },
            "audio": {
                "augment": True,
                "sample_rate": args.sample_rate,
                "duration_sec": args.duration_sec,
                "output": str(audio_out),
            },
        }
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] Saved augmented video: {actual_video_out}")
        print(f"[OK] Saved augmented audio: {audio_out}")
        print(f"[OK] Saved metadata: {meta_out}")


def main() -> None:
    args = build_arg_parser().parse_args()
    AugmentedExampleExporter(args).run()


if __name__ == "__main__":
    main()
