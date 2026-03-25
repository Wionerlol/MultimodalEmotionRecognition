from __future__ import annotations

import math

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    """Learnable attention pooling over the temporal dimension."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(1, dim // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.score(x).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        return torch.sum(x * attn, dim=1)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for variable-length sequences."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / max(1, dim)))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)


class TemporalTransformerPooling(nn.Module):
    """Temporal transformer encoder followed by learnable attention pooling."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        ffn_dim = max(dim * 2, int(dim * mlp_ratio))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pos_encoding = SinusoidalPositionalEncoding(dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = TemporalAttentionPooling(dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.pool(x)


class TemporalPooler(nn.Module):
    """Configurable temporal aggregation: mean, attention, or transformer."""

    def __init__(
        self,
        dim: int,
        mode: str = "mean",
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "mean":
            self.pool = None
        elif mode == "attn":
            self.pool = TemporalAttentionPooling(dim=dim, dropout=dropout)
        elif mode == "transformer":
            self.pool = TemporalTransformerPooling(
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported temporal pooling mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TemporalPooler expects [B, T, D], got shape={tuple(x.shape)}")
        if self.pool is None:
            return x.mean(dim=1)
        return self.pool(x)
