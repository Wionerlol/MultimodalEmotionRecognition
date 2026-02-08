from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    return (preds == targets).float().mean().item()


def macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return float(f1_score(targets, preds, average="macro"))
