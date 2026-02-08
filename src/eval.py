from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.ravdess import build_pairs, split_pairs_by_actor, RavdessAVDataset
from models.audio import AudioNet
from models.video import VideoNet
from models.fusion import FusionModel
from utils.metrics import accuracy, macro_f1


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, fusion_mode: str) -> None:
    model.eval()
    all_preds = []
    all_targets = []
    for video, audio, labels, _ in loader:
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(video, audio) if fusion_mode in {"late", "concat", "gated"} else model(
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


def build_model(num_classes: int, fusion: str) -> nn.Module:
    if fusion == "audio":
        return AudioNet(num_classes=num_classes)
    if fusion == "video":
        return VideoNet(num_classes=num_classes)
    if fusion in {"late", "concat", "gated"}:
        audio = AudioNet(num_classes=num_classes)
        video = VideoNet(num_classes=num_classes)
        return FusionModel(audio, video, num_classes=num_classes, mode=fusion)
    raise ValueError(f"Unknown fusion mode: {fusion}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=8, choices=[4, 8])
    parser.add_argument("--fusion", type=str, default="audio", choices=["audio", "video", "late", "concat", "gated"])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_actors", type=str, default="22,23,24")
    args = parser.parse_args()

    test_actors = [int(x) for x in args.test_actors.split(",")]
    pairs = build_pairs(Path(args.data_root))
    _, _, test_pairs = split_pairs_by_actor(pairs, [], [], test_actors)
    test_ds = RavdessAVDataset(test_pairs, num_classes=args.num_classes, num_frames=args.frames)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.num_classes, args.fusion)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    evaluate(model, test_loader, device, args.fusion)


if __name__ == "__main__":
    main()
