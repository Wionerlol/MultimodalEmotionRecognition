from __future__ import annotations

import torch
from torch import nn


class StochasticDepth(nn.Module):
    """Per-sample drop-path used on residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(max(0.0, min(1.0, drop_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        if keep_prob <= 0.0:
            return torch.zeros_like(x)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class ModalityDropout(nn.Module):
    """Randomly drop audio or video modality during training."""
    def __init__(self, audio_dropout_p: float = 0.2, video_dropout_p: float = 0.2):
        super().__init__()
        self.audio_dropout_p = audio_dropout_p
        self.video_dropout_p = video_dropout_p

    def forward(self, audio_emb: torch.Tensor, video_emb: torch.Tensor) -> tuple:
        """
        Args:
            audio_emb: [B, D_a]
            video_emb: [B, D_v]
        Returns:
            (audio_emb, video_emb) with optional dropout applied
        """
        if not self.training:
            return audio_emb, video_emb
        
        # Randomly drop audio (set to zero)
        if torch.rand(1).item() < self.audio_dropout_p:
            audio_emb = torch.zeros_like(audio_emb)
        
        # Randomly drop video (set to zero)
        if torch.rand(1).item() < self.video_dropout_p:
            video_emb = torch.zeros_like(video_emb)
        
        return audio_emb, video_emb


class GatedFusion(nn.Module):
    """Improved gated fusion with better initialization and dropout."""
    def __init__(self, audio_dim: int, video_dim: int, hidden_dim: int, 
                 num_classes: int, dropout_p: float = 0.2, 
                 modality_dropout_audio: float = 0.2, 
                 modality_dropout_video: float = 0.2):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        
        # Projections to common dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # Modality dropout
        self.modality_dropout = ModalityDropout(modality_dropout_audio, modality_dropout_video)
        
        # Gate network: [a_emb; v_emb] -> scalar in [0, 1]
        # This gate controls how much to trust audio vs video
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Classifier on fused representation
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize gate to favor video initially
        self._init_gate_bias()
    
    def _init_gate_bias(self):
        """Initialize gate bias to favor video (gate=0.3~0.4 initially)."""
        with torch.no_grad():
            # The last layer before sigmoid: if output = -1, sigmoid(-1) ≈ 0.27
            if len(list(self.gate.children())) >= 2:
                last_layer = list(self.gate.children())[-2]
                if isinstance(last_layer, nn.Linear):
                    # Initialize bias to negative value to make gate favor video
                    last_layer.bias.fill_(-1.0)
    
    def forward(self, audio_emb: torch.Tensor, video_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_emb: [B, audio_dim]
            video_emb: [B, video_dim]
        Returns:
            logits: [B, num_classes]
        """
        # Apply modality dropout
        audio_emb, video_emb = self.modality_dropout(audio_emb, video_emb)
        
        # Project to common dimension
        a = self.audio_proj(audio_emb)
        v = self.video_proj(video_emb)
        
        # Compute gate
        gate_in = torch.cat([a, v], dim=1)
        gate = self.gate(gate_in)  # [B, 1]
        
        # Gated fusion: g * a + (1 - g) * v
        fused = gate * a + (1 - gate) * v  # [B, hidden_dim]
        
        return self.classifier(fused)


class FusionModel(nn.Module):
    def __init__(
        self,
        audio_model: nn.Module,
        video_model: nn.Module,
        num_classes: int,
        mode: str = "late",
        common_dim: int = 256,
        # cross-attention params
        xattn_head: str = "concat",
        d_model: int = 128,
        num_heads: int = 4,
        audio_n_mels: int = 64,
        xattn_attn_dropout: float = 0.1,
        xattn_stochastic_depth: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.mode = mode
        self.d_model = d_model
        self.num_heads = num_heads
        self.audio_n_mels = audio_n_mels

        if mode in {"concat", "gated"}:
            self.audio_proj = nn.Linear(audio_model.embedding_dim, common_dim)
            self.video_proj = nn.Linear(video_model.embedding_dim, common_dim)
            if mode == "concat":
                self.fusion = nn.Sequential(
                    nn.Linear(common_dim * 2, common_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(common_dim, num_classes),
                )
            else:
                # Improved gated fusion with modality dropout
                self.modality_dropout = ModalityDropout(audio_dropout_p=0.2, video_dropout_p=0.2)
                self.gate = nn.Sequential(
                    nn.Linear(common_dim * 2, common_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(common_dim, 1),
                    nn.Sigmoid(),
                )
                self.classifier = nn.Linear(common_dim, num_classes)
                self._init_gated_fusion_bias()
        
        # Cross-attention fusion
        if mode in {"xattn", "xattn_concat", "xattn_gated"}:
            # video per-frame dim (from video_model)
            self.v_dim = getattr(video_model, "embedding_dim", 512)
            self.audio_sequence_dim = getattr(audio_model, "sequence_dim", d_model)
            self.a_dim = d_model
            # projectors
            self.v_in_proj = nn.Linear(self.v_dim, d_model)
            self.a_in_proj = nn.Linear(self.a_dim, d_model)
            # Audio conv fallback for mel-based encoders that do not provide sequence features.
            # [B,1,n_mels,Ta] -> [B,Ta,a_dim]
            self.audio_time_conv = nn.Conv1d(in_channels=audio_n_mels, out_channels=self.a_dim, kernel_size=3, padding=1)
            self.audio_seq_proj = nn.Linear(self.audio_sequence_dim, self.a_dim)
            # multihead attention
            self.v2a_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=xattn_attn_dropout, batch_first=True
            )
            self.a2v_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=xattn_attn_dropout, batch_first=True
            )
            self.v_drop_path = StochasticDepth(drop_prob=xattn_stochastic_depth)
            self.a_drop_path = StochasticDepth(drop_prob=xattn_stochastic_depth)
            self.v_norm = nn.LayerNorm(d_model)
            self.a_norm = nn.LayerNorm(d_model)
            # fusion head
            self.xattn_head = xattn_head
            if xattn_head == "concat":
                self.xattn_mlp = nn.Sequential(
                    nn.Linear(d_model * 2, common_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(common_dim, num_classes),
                )
            elif xattn_head == "gated":
                self.xattn_gate = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(d_model, 1),
                    nn.Sigmoid(),
                )
                self.xattn_classifier = nn.Linear(d_model, num_classes)
                self._init_xattn_gated_bias()
    
    def _init_gated_fusion_bias(self):
        """Initialize gate bias to favor video initially."""
        with torch.no_grad():
            for layer in self.gate:
                if isinstance(layer, nn.Linear):
                    # Find the layer before sigmoid (the last one)
                    if layer != self.gate[-1]:
                        layer.bias.fill_(-1.0)
    
    def _init_xattn_gated_bias(self):
        """Initialize xattn gate bias to favor video initially."""
        with torch.no_grad():
            for layer in self.xattn_gate:
                if isinstance(layer, nn.Linear):
                    if layer != self.xattn_gate[-1]:
                        layer.bias.fill_(-1.0)

    def forward(self, video: torch.Tensor, audio: torch.Tensor):
        if self.mode == "late":
            a_logits = self.audio_model(audio)
            v_logits = self.video_model(video)
            a_probs = torch.softmax(a_logits, dim=1)
            v_probs = torch.softmax(v_logits, dim=1)
            return (a_probs + v_probs) / 2.0

        # Cross-attention fusion branch
        if self.mode in {"xattn", "xattn_concat", "xattn_gated"}:
            # video: [B, T, 3, H, W] -> per-frame features [B, T, v_dim]
            b, t, c, h, w = video.shape
            v_in = video.view(b * t, c, h, w)
            v_feat = self.video_model.backbone(v_in).view(b, t, self.v_dim)
            # project video
            v = self.v_in_proj(v_feat)  # [B, T, d_model]

            if hasattr(self.audio_model, "encode_sequence"):
                # Warm-start friendly path: use sequence states from audio encoder.
                # WavLM returns [B, Ta, hidden], then project to attention dim.
                a_seq = self.audio_model.encode_sequence(audio)
                a_seq = self.audio_seq_proj(a_seq)
            else:
                # Mel fallback for encoders without sequence interface.
                # audio: [B, 1, n_mels, Ta] -> [B, n_mels, Ta]
                a_in = audio.squeeze(1)
                a_time = self.audio_time_conv(a_in)  # [B, a_dim, Ta]
                a_seq = a_time.permute(0, 2, 1).contiguous()  # [B, Ta, a_dim]
            # project audio (a_dim -> d_model)
            a = self.a_in_proj(a_seq)

            # cross-attention: v2 = MHA(query=v, key=a, value=a)
            v2, _ = self.v2a_attn(query=v, key=a, value=a)
            v = self.v_norm(v + self.v_drop_path(v2))

            # a2 = MHA(query=a, key=v, value=v)
            a2, _ = self.a2v_attn(query=a, key=v, value=v)
            a = self.a_norm(a + self.a_drop_path(a2))

            # pool
            v_emb = v.mean(dim=1)
            a_emb = a.mean(dim=1)

            # fusion head
            if self.xattn_head == "concat":
                fused = torch.cat([v_emb, a_emb], dim=1)
                return self.xattn_mlp(fused)
            elif self.xattn_head == "gated":
                gate = self.xattn_gate(torch.cat([v_emb, a_emb], dim=1))
                fused = gate * v_emb + (1 - gate) * a_emb
                return self.xattn_classifier(fused)

        # default (non-xattn) behavior: use existing audio/video encoders + simple fusion
        a_emb = self.audio_model.encode(audio)
        v_emb = self.video_model.encode(video)

        if self.mode in {"concat", "gated"}:
            a_emb = self.audio_proj(a_emb)
            v_emb = self.video_proj(v_emb)

        if self.mode == "concat":
            fused = torch.cat([a_emb, v_emb], dim=1)
            return self.fusion(fused)

        if self.mode == "gated":
            # Apply modality dropout
            a_emb, v_emb = self.modality_dropout(a_emb, v_emb)
            
            gate_in = torch.cat([a_emb, v_emb], dim=1)
            g = self.gate(gate_in)
            fused = g * a_emb + (1 - g) * v_emb
            return self.classifier(fused)

        raise ValueError(f"Unknown fusion mode: {self.mode}")
