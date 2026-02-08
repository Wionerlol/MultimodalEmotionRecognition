from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class VideoNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B, 512, 1, 1]
        self.embedding_dim = 512
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x).view(b, t, self.embedding_dim)
        emb = feat.mean(dim=1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.classifier(emb)
