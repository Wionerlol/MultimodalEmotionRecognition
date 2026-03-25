from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eval import build_model


FOUR_CLASS_LABELS = ["neutral_calm", "happy", "negative", "surprised"]
EIGHT_CLASS_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
FUSION_MODES = {"audio", "video", "late", "concat", "gated", "xattn", "xattn_concat", "xattn_gated"}


def labels_for_num_classes(num_classes: int) -> list[str]:
    return EIGHT_CLASS_LABELS if num_classes == 8 else FOUR_CLASS_LABELS


class TorchModelRunner:
    def __init__(self, checkpoint_path: str, device: str, fallback_fusion: str = "xattn", enable_dynamic_quant: bool = False):
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError("Checkpoint format not supported. Expected {'model': state_dict, 'config': ...}.")

        self.config = checkpoint.get("config", {})
        self.fusion_mode = str(self.config.get("fusion", fallback_fusion))
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        self.num_classes = int(self.config.get("num_classes", 8))
        self.use_wavlm = bool(self.config.get("use_wavlm", False))
        self.labels = labels_for_num_classes(self.num_classes)
        model = build_model(
            num_classes=self.num_classes,
            fusion=self.fusion_mode,
            xattn_head=self.config.get("xattn_head", "concat"),
            xattn_d_model=self.config.get("xattn_d_model", 128),
            xattn_heads=self.config.get("xattn_heads", 4),
            xattn_attn_dropout=self.config.get("xattn_attn_dropout", 0.1),
            xattn_stochastic_depth=self.config.get("xattn_stochastic_depth", 0.1),
            xattn_use_emotion_prior=self.config.get("xattn_use_emotion_prior", False),
            xattn_emotion_prior_dim=self.config.get("xattn_emotion_prior_dim", 8),
            xattn_emotion_prior_hidden_dim=self.config.get("xattn_emotion_prior_hidden_dim", 64),
            xattn_emotion_prior_dropout=self.config.get("xattn_emotion_prior_dropout", 0.1),
            temporal_pooling=self.config.get("temporal_pooling", "mean"),
            temporal_num_heads=self.config.get("temporal_num_heads", 4),
            temporal_num_layers=self.config.get("temporal_num_layers", 1),
            temporal_dropout=self.config.get("temporal_dropout", 0.1),
            audio_n_mels=self.config.get("audio_n_mels", 64),
            use_resnet_audio=self.config.get("use_resnet_audio", True),
            use_wavlm=self.use_wavlm,
            fusion_align_mode=self.config.get("fusion_align_mode", "none"),
            fusion_align_dim=self.config.get("fusion_align_dim", 256),
            fusion_align_temperature=self.config.get("fusion_align_temperature", 0.07),
        )
        model.load_state_dict(checkpoint["model"])
        if enable_dynamic_quant and self.device.type == "cpu":
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        self.model = model.to(self.device).eval()

    def predict_probs(self, videos: torch.Tensor, audios: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.fusion_mode == "audio":
                outputs = self.model(audios)
            elif self.fusion_mode == "video":
                outputs = self.model(videos)
            else:
                outputs = self.model(videos, audios)
            probs = outputs if self.fusion_mode == "late" else torch.softmax(outputs, dim=1)
        return probs.detach().cpu()


class OnnxModelRunner:
    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX inference.") from exc

        self.onnx_path = Path(onnx_path).expanduser()
        meta_path = self.onnx_path.with_suffix(self.onnx_path.suffix + ".meta.json")
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"ONNX metadata file not found: {meta_path}")

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.fusion_mode = str(self.meta["fusion"])
        self.num_classes = int(self.meta["num_classes"])
        self.labels = labels_for_num_classes(self.num_classes)
        available = ort.get_available_providers()
        chosen = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(self.onnx_path), providers=chosen)

    def predict_probs(self, videos: torch.Tensor, audios: torch.Tensor) -> torch.Tensor:
        inputs: dict[str, Any] = {}
        if self.fusion_mode == "audio":
            inputs["audio"] = audios.detach().cpu().numpy()
        elif self.fusion_mode == "video":
            inputs["video"] = videos.detach().cpu().numpy()
        else:
            inputs["video"] = videos.detach().cpu().numpy()
            inputs["audio"] = audios.detach().cpu().numpy()
        outputs = self.session.run(None, inputs)[0]
        return torch.from_numpy(outputs)
