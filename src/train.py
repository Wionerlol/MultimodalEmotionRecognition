from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Windows encoding issue with wandb
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from data.ravdess import (
    build_pairs,
    save_pairs_csv,
    split_pairs_by_actor,
    split_pairs_stratified,
    RavdessAVDataset,
    RavdessAVDatasetWavLM,
    map_emotion_label,
)
from models.audio import AudioNet
from models.video import VideoNet
from models.fusion import FusionModel
from models.wavlm_audio import WavLMAudioEncoder
from utils.metrics import accuracy, macro_f1
from utils.seed import set_seed


def parse_actor_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(x) for x in text.split(",")]


def build_dataloaders(
    data_root: Path,
    num_classes: int,
    num_frames: int,
    train_actors: List[int],
    val_actors: List[int],
    test_actors: List[int],
    batch_size: int,
    stratified: bool = False,
    use_wavlm: bool = False,
    train_augment: bool = True,
    use_face_crop: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    pairs = build_pairs(data_root)
    save_pairs_csv(pairs, Path("pairs.csv"))

    if len(pairs) == 0:
        raise RuntimeError("No audio-video pairs found. Check data_root and filenames.")

    print(f"Total pairs: {len(pairs)}")
    for p in pairs[:3]:
        print(
            f"Sample pair: actor={p.actor} emotion={p.emotion} intensity={p.intensity} "
            f"statement={p.statement} repetition={p.repetition}"
        )

    # Use stratified split or actor-based split
    if stratified:
        print("Using stratified split (balanced emotion distribution)")
        train_pairs, val_pairs, test_pairs = split_pairs_stratified(pairs, seed=42)
    else:
        print("Using actor-based split")
        train_pairs, val_pairs, test_pairs = split_pairs_by_actor(
            pairs, train_actors, val_actors, test_actors
        )
    
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs)}")

    def class_dist(ps: List) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for p in ps:
            label = map_emotion_label(p.emotion, num_classes)
            counts[label] = counts.get(label, 0) + 1
        return counts

    print(f"Train class distribution: {class_dist(train_pairs)}")
    print(f"Val class distribution: {class_dist(val_pairs)}")
    print(f"Test class distribution: {class_dist(test_pairs)}")

    # Choose dataset class based on audio encoder type
    DatasetClass = RavdessAVDatasetWavLM if use_wavlm else RavdessAVDataset
    
    # Apply augmentation only to training dataset, face crop to all
    train_ds = DatasetClass(
        train_pairs,
        num_classes=num_classes,
        num_frames=num_frames,
        augment=train_augment,
        use_face_crop=use_face_crop,
    )
    val_ds = DatasetClass(
        val_pairs,
        num_classes=num_classes,
        num_frames=num_frames,
        augment=False,
        use_face_crop=use_face_crop,
    )
    test_ds = DatasetClass(
        test_pairs,
        num_classes=num_classes,
        num_frames=num_frames,
        augment=False,
        use_face_crop=use_face_crop,
    )

    # Use num_workers=0 on Windows to avoid multiprocessing issues
    import sys
    num_workers = 0 if sys.platform == "win32" else 4
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    sizes = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
    return train_loader, val_loader, test_loader, sizes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    fusion_mode: str,
) -> Dict[str, float]:
    model.train()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    for video, audio, labels, _ in tqdm(loader, desc="train", leave=False):
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # For single-modality models (audio/video), pass only that modality
        # For fusion models, pass both
        if fusion_mode in {"audio", "video"}:
            outputs = model(audio if fusion_mode == "audio" else video)
        else:
            outputs = model(video, audio)
        if fusion_mode == "late":
            log_probs = torch.log(outputs + 1e-8)
            loss = loss_fn(log_probs, labels)
            preds = outputs.argmax(dim=1)
        else:
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy(all_preds, all_targets),
        "f1": macro_f1(all_preds, all_targets),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    fusion_mode: str,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    for video, audio, labels, _ in tqdm(loader, desc="eval", leave=False):
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        # For single-modality models (audio/video), pass only that modality
        # For fusion models, pass both
        if fusion_mode in {"audio", "video"}:
            outputs = model(audio if fusion_mode == "audio" else video)
        else:
            outputs = model(video, audio)
        if fusion_mode == "late":
            log_probs = torch.log(outputs + 1e-8)
            loss = loss_fn(log_probs, labels)
            preds = outputs.argmax(dim=1)
        else:
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy(all_preds, all_targets),
        "f1": macro_f1(all_preds, all_targets),
    }


def plot_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """Generate confusion matrix plot."""
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "black")
    
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig


def build_model(
    num_classes: int,
    fusion: str,
    pretrained_video: bool = True,
    xattn_head: str = "concat",
    xattn_d_model: int = 128,
    xattn_heads: int = 4,
    audio_n_mels: int = 64,
    use_resnet_audio: bool = True,
    use_wavlm: bool = False,
) -> nn.Module:
    if fusion == "audio":
        if use_wavlm:
            return WavLMAudioEncoder(num_classes=num_classes)
        else:
            return AudioNet(num_classes=num_classes, use_resnet=use_resnet_audio, spec_augment=True)
    if fusion == "video":
        return VideoNet(num_classes=num_classes, pretrained=pretrained_video)
    if fusion in {"late", "concat", "gated"}:
        if use_wavlm:
            audio = WavLMAudioEncoder(num_classes=num_classes)
        else:
            audio = AudioNet(num_classes=num_classes, use_resnet=use_resnet_audio, spec_augment=True)
        video = VideoNet(num_classes=num_classes, pretrained=pretrained_video)
        return FusionModel(audio, video, num_classes=num_classes, mode=fusion)
    if fusion in {"xattn", "xattn_concat", "xattn_gated"}:
        if use_wavlm:
            audio = WavLMAudioEncoder(num_classes=num_classes)
        else:
            audio = AudioNet(num_classes=num_classes, use_resnet=use_resnet_audio, spec_augment=True)
        video = VideoNet(num_classes=num_classes, pretrained=pretrained_video)
        # map alias fusion names
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
            audio_n_mels=audio_n_mels if not use_wavlm else 768,  # WavLM has 768 hidden dim
        )
    raise ValueError(f"Unknown fusion mode: {fusion}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=8, choices=[4, 8])
    parser.add_argument(
        "--fusion",
        type=str,
        default="audio",
        choices=["audio", "video", "late", "concat", "gated", "xattn", "xattn_concat", "xattn_gated"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default="stratified", choices=["actor", "stratified"],
                        help="Data split mode: 'actor' (by actors) or 'stratified' (balanced emotion distribution)")
    parser.add_argument("--train_actors", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
                        help="Used only with --split_mode actor")
    parser.add_argument("--val_actors", type=str, default="19,20,21",
                        help="Used only with --split_mode actor")
    parser.add_argument("--test_actors", type=str, default="22,23,24",
                        help="Used only with --split_mode actor")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Training set ratio (used only with --split_mode stratified)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation set ratio (used only with --split_mode stratified)")
    parser.add_argument("--no_pretrained_video", action="store_true")
    parser.add_argument("--use_cosine_annealing", action="store_true", help="Use cosine annealing learning rate scheduler")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--xattn_head", type=str, choices=["concat", "gated"], default="concat", help="xattn fusion head when --fusion xattn is used")
    parser.add_argument("--xattn_d_model", type=int, default=128, help="d_model for cross-attention (default 128)")
    parser.add_argument("--xattn_heads", type=int, default=4, help="number of attention heads for xattn (default 4)")
    parser.add_argument("--audio_n_mels", type=int, default=64, help="n_mels used in audio preprocessing (default 64)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay (default: 1e-4)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience in epochs (set 0 to disable)")
    parser.add_argument("--use_resnet_audio", action="store_true", help="Use ResNet18 for audio encoder (default: lightweight CNN)")
    parser.add_argument("--two_stage_training", action="store_true", help="Use two-stage training (freeze encoders, train fusion head first)")
    parser.add_argument("--use_wavlm", action="store_true", help="Use WavLM pretrained model for audio encoder")
    parser.add_argument("--wavlm_stage", type=int, default=1, choices=[1, 2], help="WavLM training stage: 1 (freeze backbone) or 2 (unfreeze last 2-4 layers)")
    parser.add_argument("--backbone_lr", type=float, default=3e-5, help="Learning rate for WavLM backbone in stage 2 (default: 3e-5)")
    parser.add_argument("--use_face_crop", action="store_true", default=True, help="Use face detection and cropping (default: True)")
    parser.add_argument("--no_face_crop", dest="use_face_crop", action="store_false", help="Disable face detection and cropping")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project="multimodal-emotion-recognition",
            name=f"{args.fusion}_epochs{args.epochs}_bs{args.batch_size}_{args.split_mode}",
            config={
                "fusion": args.fusion,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_classes": args.num_classes,
                "split_mode": args.split_mode,
                "cosine_annealing": args.use_cosine_annealing,
            }
        )

    train_actors = parse_actor_list(args.train_actors)
    val_actors = parse_actor_list(args.val_actors)
    test_actors = parse_actor_list(args.test_actors)

    train_loader, val_loader, test_loader, sizes = build_dataloaders(
        data_root=Path(args.data_root),
        num_classes=args.num_classes,
        num_frames=args.frames,
        train_actors=train_actors,
        val_actors=val_actors,
        test_actors=test_actors,
        batch_size=args.batch_size,
        stratified=(args.split_mode == "stratified"),
        use_wavlm=args.use_wavlm,
        train_augment=True,
        use_face_crop=args.use_face_crop,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    model = build_model(
        args.num_classes,
        args.fusion,
        pretrained_video=not args.no_pretrained_video,
        xattn_head=getattr(args, "xattn_head", "concat"),
        xattn_d_model=getattr(args, "xattn_d_model", 128),
        xattn_heads=getattr(args, "xattn_heads", 4),
        audio_n_mels=getattr(args, "audio_n_mels", 64),
        use_resnet_audio=getattr(args, "use_resnet_audio", True),
        use_wavlm=args.use_wavlm,
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})\n")

    if args.fusion == "late":
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Setup optimizer based on WavLM stage
    if args.use_wavlm and args.fusion == "audio":
        # Audio-only with WavLM: two-stage training
        if args.wavlm_stage == 1:
            # Stage 1: freeze backbone, only train classifier
            model.unfreeze_backbone(0)  # Keep backbone frozen
            params_to_train = model.get_stage1_params()
            optimizer = torch.optim.Adam(params_to_train, lr=args.lr, weight_decay=args.weight_decay)
            print(f"WavLM Stage 1: Training classifier head only (lr={args.lr})")
        else:
            # Stage 2: unfreeze last 2 layers
            model.unfreeze_backbone(num_last_layers=2)
            param_groups = model.get_stage2_params()
            optimizer = torch.optim.Adam([
                {'params': param_groups['backbone'], 'lr': args.backbone_lr},
                {'params': param_groups['head'], 'lr': args.lr},
            ], weight_decay=args.weight_decay)
            print(f"WavLM Stage 2: Training with backbone lr={args.backbone_lr}, head lr={args.lr}")
    elif args.use_wavlm and args.fusion in {"concat", "gated", "late"}:
        # Fusion with WavLM: same as stage 1 (only train fusion head, freeze both encoders)
        # In this case, we freeze both audio and video encoders
        if hasattr(model, 'audio'):
            if isinstance(model.audio, WavLMAudioEncoder):
                model.audio.unfreeze_backbone(0)  # Keep WavLM backbone frozen
        if hasattr(model, 'video'):
            for param in model.video.parameters():
                param.requires_grad = False
        # Only train fusion head
        fusion_params = [p for name, p in model.named_parameters() if p.requires_grad]
        if not fusion_params:
            # If no params marked as trainable, we need to mark fusion head
            fusion_params = model.fusion_head.parameters() if hasattr(model, 'fusion_head') else model.parameters()
        optimizer = torch.optim.Adam(fusion_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"WavLM Fusion: Training fusion head with frozen encoders (lr={args.lr})")
    else:
        # Standard training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Use cosine annealing learning rate scheduler
    if args.use_cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        scheduler = None
    
    best_f1 = -1.0
    best_path = Path("outputs") / f"best_{args.fusion}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    early_stopping_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, args.fusion
        )
        val_metrics = evaluate(model, val_loader, device, loss_fn, args.fusion)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} f1 {train_metrics['f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} f1 {val_metrics['f1']:.4f} | "
            f"lr {current_lr:.2e}"
        )
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics['loss'],
                "train/acc": train_metrics['acc'],
                "train/f1": train_metrics['f1'],
                "val/loss": val_metrics['loss'],
                "val/acc": val_metrics['acc'],
                "val/f1": val_metrics['f1'],
                "lr": current_lr,
            })
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            early_stopping_counter = 0
            torch.save({"model": model.state_dict(), "val_f1": best_f1}, best_path)
        else:
            early_stopping_counter += 1
            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
                print(f"Best val F1: {best_f1:.4f}")
                break

    if sizes["test"] > 0:
        test_metrics = evaluate(model, test_loader, device, loss_fn, args.fusion)
        print(
            f"Test | loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f} f1 {test_metrics['f1']:.4f}"
        )
        
        # Log test confusion matrix to wandb
        if args.wandb:
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for video, audio, labels, _ in test_loader:
                    video = video.to(device)
                    audio = audio.to(device)
                    if args.fusion == "audio":
                        outputs = model(audio)
                    elif args.fusion == "video":
                        outputs = model(video)
                    else:
                        outputs = model(video, audio)
                    preds = outputs.argmax(dim=1)
                    test_preds.append(preds)
                    test_targets.append(labels)
            
            test_preds = torch.cat(test_preds)
            test_targets = torch.cat(test_targets)
            cm_fig = plot_confusion_matrix(test_preds, test_targets, args.num_classes)
            wandb.log({"test/confusion_matrix": wandb.Image(cm_fig)})
            plt.close(cm_fig)
            wandb.log({
                "test/loss": test_metrics['loss'],
                "test/acc": test_metrics['acc'],
                "test/f1": test_metrics['f1'],
            })
    
    print(f"Best val macro-F1: {best_f1:.4f} | checkpoint: {best_path}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
