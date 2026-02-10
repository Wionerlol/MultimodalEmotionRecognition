"""Model loading and checkpoint handling."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .config import CHECKPOINT_PATH, DEVICE, PROJECT_ROOT


class _InferenceModelAdapter(nn.Module):
    """Unify inference call signature: forward(video, audio) -> logits."""

    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        self.model = model
        self.mode = mode
        self.requires_wavlm = False

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if self.mode == "audio":
            return self.model(audio)
        if self.mode == "video":
            return self.model(video)
        return self.model(video, audio)


class DummyModel(nn.Module):
    """Placeholder model when checkpoint is not available."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.dummy = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Model not loaded. Provide checkpoint at ./checkpoints/best.pt "
            "or set EMO_MOCK=1 for mock mode."
        )


class ModelLoaderService:
    """Service object for checkpoint-driven model loading."""

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        device: str = DEVICE,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.project_root = Path(project_root)

    def ensure_src_import_path(self) -> None:
        src_dir = Path(os.environ.get("MODEL_SRC_DIR", str(self.project_root / "src"))).expanduser()
        if src_dir.exists():
            src_dir_text = str(src_dir)
            if src_dir_text not in sys.path:
                sys.path.insert(0, src_dir_text)

    @staticmethod
    def infer_model_signature(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, str]:
        if any(k.startswith("audio_model.") for k in state_dict) and any(
            k.startswith("video_model.") for k in state_dict
        ):
            if any(k.startswith("xattn_gate.") for k in state_dict):
                return "xattn", "gated"
            if any(k.startswith("xattn_mlp.") for k in state_dict):
                return "xattn", "concat"
            if any(k.startswith("fusion.") for k in state_dict):
                return "concat", "concat"
            if any(k.startswith("gate.") for k in state_dict):
                return "gated", "gated"
            return "late", "concat"

        if any(k.startswith("encoder.") for k in state_dict):
            return "audio", "concat"
        if any(k.startswith("backbone.") for k in state_dict):
            return "video", "concat"
        raise RuntimeError("Unable to infer model type from checkpoint state_dict keys.")

    @staticmethod
    def checkpoint_uses_wavlm(state_dict: Dict[str, torch.Tensor]) -> bool:
        return any(k.startswith("audio_model.wavlm.") for k in state_dict) or any(
            k.startswith("wavlm.") for k in state_dict
        )

    def build_model(self, num_classes: int, fusion_mode: str, xattn_head: str, use_wavlm: bool) -> nn.Module:
        self.ensure_src_import_path()
        try:
            from models.audio import AudioNet
            from models.fusion import FusionModel
            from models.video import VideoNet
            from models.wavlm_audio import WavLMAudioEncoder
        except Exception as exc:
            raise RuntimeError(
                "Failed to import model classes from src/models. "
                "Make sure project src/ is mounted and PYTHONPATH includes it."
            ) from exc

        if fusion_mode == "audio":
            if use_wavlm:
                return WavLMAudioEncoder(num_classes=num_classes)
            return AudioNet(num_classes=num_classes, use_resnet=True, spec_augment=False)
        if fusion_mode == "video":
            return VideoNet(num_classes=num_classes, pretrained=False)

        if use_wavlm:
            audio = WavLMAudioEncoder(num_classes=num_classes)
            audio_n_mels = 768
        else:
            audio = AudioNet(num_classes=num_classes, use_resnet=True, spec_augment=False)
            audio_n_mels = 64
        video = VideoNet(num_classes=num_classes, pretrained=False)
        return FusionModel(
            audio_model=audio,
            video_model=video,
            num_classes=num_classes,
            mode=fusion_mode,
            xattn_head=xattn_head,
            audio_n_mels=audio_n_mels,
        )

    def load_model_from_checkpoint(self, checkpoint_path: Path, num_classes: int = 8) -> Optional[nn.Module]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            env_mode = os.environ.get("MODEL_FUSION", "").strip().lower()
            env_head = os.environ.get("MODEL_XATTN_HEAD", "concat").strip().lower()
            use_wavlm = self.checkpoint_uses_wavlm(state_dict)

            if env_mode:
                fusion_mode, xattn_head = env_mode, env_head
            else:
                fusion_mode, xattn_head = self.infer_model_signature(state_dict)

            model = self.build_model(
                num_classes=num_classes,
                fusion_mode=fusion_mode,
                xattn_head=xattn_head,
                use_wavlm=use_wavlm,
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected checkpoint keys ({len(unexpected)}): {unexpected[:8]}")
            if len(missing) > 32:
                raise RuntimeError(
                    f"Too many missing keys when loading checkpoint ({len(missing)}). "
                    "Set MODEL_FUSION/MODEL_XATTN_HEAD to match training config."
                )
            print(
                f"[INFO] Loaded checkpoint with fusion_mode={fusion_mode}, "
                f"xattn_head={xattn_head}, use_wavlm={use_wavlm}"
            )
            adapted_model = _InferenceModelAdapter(model, fusion_mode)
            adapted_model.requires_wavlm = use_wavlm
            return adapted_model
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc

    def get_model(self, num_classes: int = 8, allow_mock: bool = False) -> nn.Module:
        model_result = self.load_model_from_checkpoint(self.checkpoint_path, num_classes=num_classes)

        if isinstance(model_result, nn.Module):
            return model_result

        if allow_mock:
            print(f"[INFO] Checkpoint not found at {self.checkpoint_path}. Running in mock mode.")
            return DummyModel(num_classes=num_classes)

        raise RuntimeError(
            f"Checkpoint not found at {self.checkpoint_path}. "
            "Please provide a checkpoint or set EMO_MOCK=1 for testing."
        )


MODEL_LOADER_SERVICE = ModelLoaderService()


# Compatibility wrappers (keep old imports working).
def _ensure_src_import_path() -> None:
    MODEL_LOADER_SERVICE.ensure_src_import_path()


def _infer_model_signature(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, str]:
    return MODEL_LOADER_SERVICE.infer_model_signature(state_dict)


def _checkpoint_uses_wavlm(state_dict: Dict[str, torch.Tensor]) -> bool:
    return MODEL_LOADER_SERVICE.checkpoint_uses_wavlm(state_dict)


def _build_model(num_classes: int, fusion_mode: str, xattn_head: str, use_wavlm: bool) -> nn.Module:
    return MODEL_LOADER_SERVICE.build_model(
        num_classes=num_classes,
        fusion_mode=fusion_mode,
        xattn_head=xattn_head,
        use_wavlm=use_wavlm,
    )


def load_model_from_checkpoint(checkpoint_path: Path, num_classes: int = 8) -> Optional[nn.Module]:
    return MODEL_LOADER_SERVICE.load_model_from_checkpoint(checkpoint_path=checkpoint_path, num_classes=num_classes)


def get_model(num_classes: int = 8, allow_mock: bool = False) -> nn.Module:
    return MODEL_LOADER_SERVICE.get_model(num_classes=num_classes, allow_mock=allow_mock)
