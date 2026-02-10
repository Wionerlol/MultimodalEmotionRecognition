"""WavLM-based audio encoder with two-stage finetuning support."""
from __future__ import annotations

import os

import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel


class WavLMAudioEncoder(nn.Module):
    """WavLM audio encoder with two-stage finetuning strategy.
    
    Stage 1: Freeze backbone, only train classifier head (epochs 5-10, lr=1e-3)
    Stage 2: Unfreeze last 2-4 layers, continue training (epochs 5-10, backbone lr=1e-5~3e-5, head lr=1e-4~3e-4)
    """
    def __init__(self, num_classes: int, embedding_dim: int = 768, model_name: str = "microsoft/wavlm-base"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        
        # Load WavLM pretrained model (with offline fallback for WSL/air-gapped envs)
        local_files_only = os.environ.get("HF_LOCAL_ONLY", "0") == "1"
        try:
            self.wavlm = WavLMModel.from_pretrained(model_name, local_files_only=local_files_only)
        except Exception as exc:
            print(f"[WARNING] Failed to load pretrained {model_name}: {exc}")
            print("[WARNING] Falling back to WavLM base config init; checkpoint weights will be loaded afterward.")
            self.wavlm = WavLMModel(WavLMConfig())
        
        # Get actual hidden size from model config
        actual_hidden_size = self.wavlm.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_classes),
        )
        
        # Initially freeze backbone
        self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all WavLM parameters (Stage 1)."""
        for param in self.wavlm.parameters():
            param.requires_grad = False
    
    def _unfreeze_last_n_layers(self, n: int = 2):
        """Unfreeze last n transformer layers (Stage 2).
        
        Args:
            n: Number of last layers to unfreeze (default 2-4)
        """
        if n == 0:
            # Keep everything frozen, only train classifier
            return
        
        # WavLM has encoder.layers, typically 12 layers
        num_layers = len(self.wavlm.encoder.layers)
        
        # Unfreeze last n layers
        for i in range(max(0, num_layers - n), num_layers):
            for param in self.wavlm.encoder.layers[i].parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self, num_last_layers: int = 2):
        """Transition from Stage 1 to Stage 2: unfreeze last num_last_layers layers."""
        self._unfreeze_last_n_layers(num_last_layers)
    
    def get_stage1_params(self):
        """Get parameters for Stage 1 training (only classifier head).
        
        Returns:
            List of parameters that should be trained in Stage 1
        """
        return list(self.classifier.parameters())
    
    def get_stage2_params(self):
        """Get parameters for Stage 2 training (unfrozen layers + classifier).
        
        Returns:
            Dict with 'backbone' and 'head' parameter groups for different learning rates
        """
        backbone_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name or 'head' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        return {
            'backbone': backbone_params,
            'head': head_params,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Raw waveform [B, 1, T] or [B, T]
        
        Returns:
            Logits [B, num_classes]
        """
        # x shape: [B, 1, T] or [B, T]
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, T]
        
        # WavLM forward
        outputs = self.wavlm(x)
        
        # outputs.last_hidden_state: [B, T, hidden_size]
        # Mean pooling over time
        hidden = outputs.last_hidden_state  # [B, T, H]
        a_emb = hidden.mean(dim=1)  # [B, H]
        
        # Classification
        logits = self.classifier(a_emb)  # [B, num_classes]
        
        return logits
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding representation (for multi-modal fusion).
        
        Args:
            x: Raw waveform [B, 1, T] or [B, T]
        
        Returns:
            Embedding [B, embedding_dim]
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, T]

        # Preserve efficiency when backbone is frozen, but allow gradients
        # in fusion stage-2 when selected WavLM layers are unfrozen.
        wavlm_trainable = any(param.requires_grad for param in self.wavlm.parameters())
        if self.training and wavlm_trainable:
            outputs = self.wavlm(x)
        else:
            with torch.no_grad():
                outputs = self.wavlm(x)
        
        hidden = outputs.last_hidden_state  # [B, T, hidden_size]
        a_emb = hidden.mean(dim=1)  # [B, hidden_size]
        
        # Project to embedding_dim if needed
        if a_emb.size(-1) != self.embedding_dim:
            # Use first layer of classifier for projection
            a_emb = self.classifier[0](a_emb)
        
        return a_emb
