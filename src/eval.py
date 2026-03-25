from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.ravdess import DATASET_FACTORY, PAIR_SERVICE, SPLIT_SERVICE
from models.audio import AudioNet
from models.video import VideoNet
from models.fusion import FusionModel
from models.wavlm_audio import WavLMAudioEncoder
from utils.metrics import accuracy, macro_f1


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    rel = platform.release().lower()
    return "microsoft" in rel or "wsl" in rel


def _auto_num_workers(data_root: Path, requested: int) -> int:
    if requested >= 0:
        return requested
    if sys.platform == "win32":
        return 0
    if _is_wsl() and str(data_root.expanduser().resolve()).startswith("/mnt/"):
        return 0
    if _is_wsl():
        return 2
    cpu_count = os.cpu_count() or 4
    return min(8, max(2, cpu_count // 2))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, fusion_mode: str) -> None:
    model.eval()
    all_preds = []
    all_targets = []
    for video, audio, labels, _ in loader:
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(video, audio) if fusion_mode in {"late", "concat", "gated", "xattn", "xattn_concat", "xattn_gated"} else model(
            audio if fusion_mode == "audio" else video
        )
        if fusion_mode == "late":
            preds = outputs.argmax(dim=1)
        else:
            preds = outputs.argmax(dim=1)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    print(f"Accuracy: {accuracy(all_preds, all_targets):.4f}")
    print(f"Macro-F1: {macro_f1(all_preds, all_targets):.4f}")


def build_model(
    num_classes: int,
    fusion: str,
    xattn_head: str = "concat",
    xattn_d_model: int = 128,
    xattn_heads: int = 4,
    xattn_attn_dropout: float = 0.1,
    xattn_stochastic_depth: float = 0.1,
    temporal_pooling: str = "mean",
    temporal_num_heads: int = 4,
    temporal_num_layers: int = 1,
    temporal_dropout: float = 0.1,
    audio_n_mels: int = 64,
    use_resnet_audio: bool = True,
    use_wavlm: bool = False,
    fusion_align_mode: str = "none",
    fusion_align_dim: int = 256,
    fusion_align_temperature: float = 0.07,
    xattn_use_emotion_prior: bool = False,
    xattn_emotion_prior_dim: int = 8,
    xattn_emotion_prior_hidden_dim: int = 64,
    xattn_emotion_prior_dropout: float = 0.1,
) -> nn.Module:
    if fusion == "audio":
        if use_wavlm:
            return WavLMAudioEncoder(
                num_classes=num_classes,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        return AudioNet(
            num_classes=num_classes,
            use_resnet=use_resnet_audio,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
            xattn_use_emotion_prior=xattn_use_emotion_prior,
            xattn_emotion_prior_dim=xattn_emotion_prior_dim,
            xattn_emotion_prior_hidden_dim=xattn_emotion_prior_hidden_dim,
            xattn_emotion_prior_dropout=xattn_emotion_prior_dropout,
        )
    if fusion == "video":
        return VideoNet(
            num_classes=num_classes,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
    if fusion in {"late", "concat", "gated"}:
        if use_wavlm:
            audio = WavLMAudioEncoder(
                num_classes=num_classes,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        else:
            audio = AudioNet(
                num_classes=num_classes,
                use_resnet=use_resnet_audio,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        video = VideoNet(
            num_classes=num_classes,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
        return FusionModel(
            audio,
            video,
            num_classes=num_classes,
            mode=fusion,
            fusion_align_mode=fusion_align_mode,
            fusion_align_dim=fusion_align_dim,
            fusion_align_temperature=fusion_align_temperature,
        )
    if fusion in {"xattn", "xattn_concat", "xattn_gated"}:
        if use_wavlm:
            audio = WavLMAudioEncoder(
                num_classes=num_classes,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        else:
            audio = AudioNet(
                num_classes=num_classes,
                use_resnet=use_resnet_audio,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        video = VideoNet(
            num_classes=num_classes,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
        head = xattn_head
        if fusion == "xattn_concat":
            head = "concat"
        if fusion == "xattn_gated":
            head = "gated"
        return FusionModel(
            audio,
            video,
            num_classes=num_classes,
            mode="xattn",
            xattn_head=head,
            d_model=xattn_d_model,
            num_heads=xattn_heads,
            audio_n_mels=audio_n_mels if not use_wavlm else 768,
            xattn_attn_dropout=xattn_attn_dropout,
            xattn_stochastic_depth=xattn_stochastic_depth,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
    raise ValueError(f"Unknown fusion mode: {fusion}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=8, choices=[4, 8])
    parser.add_argument(
        "--fusion",
        type=str,
        default="audio",
        choices=["audio", "video", "late", "concat", "gated", "xattn", "xattn_concat", "xattn_gated"],
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_actors", type=str, default="22,23,24")
    parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers (-1 for auto)")
    return parser


class EmotionEvaluator:
    """Evaluation orchestrator for checkpoint-based model validation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> None:
        args = self.args
        test_actors = [int(x) for x in args.test_actors.split(",")]
        pairs = PAIR_SERVICE.build_pairs(Path(args.data_root))
        _, _, test_pairs = SPLIT_SERVICE.by_actor(pairs, [], [], test_actors)
        test_ds = DATASET_FACTORY.create(
            test_pairs,
            num_classes=args.num_classes,
            num_frames=args.frames,
            augment=False,
            use_face_crop=True,
            use_wavlm=False,
        )
        num_workers = _auto_num_workers(Path(args.data_root), args.num_workers)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, **loader_kwargs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model = build_model(
            args.num_classes,
            args.fusion,
            xattn_head=config.get("xattn_head", "concat"),
            xattn_d_model=config.get("xattn_d_model", 128),
            xattn_heads=config.get("xattn_heads", 4),
            xattn_attn_dropout=config.get("xattn_attn_dropout", 0.1),
            xattn_stochastic_depth=config.get("xattn_stochastic_depth", 0.1),
            xattn_use_emotion_prior=config.get("xattn_use_emotion_prior", False),
            xattn_emotion_prior_dim=config.get("xattn_emotion_prior_dim", 8),
            xattn_emotion_prior_hidden_dim=config.get("xattn_emotion_prior_hidden_dim", 64),
            xattn_emotion_prior_dropout=config.get("xattn_emotion_prior_dropout", 0.1),
            temporal_pooling=config.get("temporal_pooling", "mean"),
            temporal_num_heads=config.get("temporal_num_heads", 4),
            temporal_num_layers=config.get("temporal_num_layers", 1),
            temporal_dropout=config.get("temporal_dropout", 0.1),
            audio_n_mels=config.get("audio_n_mels", 64),
            use_resnet_audio=config.get("use_resnet_audio", True),
            use_wavlm=config.get("use_wavlm", False),
            fusion_align_mode=config.get("fusion_align_mode", "none"),
            fusion_align_dim=config.get("fusion_align_dim", 256),
            fusion_align_temperature=config.get("fusion_align_temperature", 0.07),
        )
        model.load_state_dict(ckpt["model"])
        model.to(device)
        evaluate(model, test_loader, device, args.fusion)


def main() -> None:
    args = build_arg_parser().parse_args()
    EmotionEvaluator(args).run()


if __name__ == "__main__":
    main()
