from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from optimized_runtime import TorchModelRunner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export optimized inference models.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path.")
    parser.add_argument("--device", type=str, default="cpu", help="Export device, usually cpu.")
    parser.add_argument("--frames", type=int, default=8, help="Dummy video frame count used for ONNX tracing.")
    parser.add_argument("--image_size", type=int, default=112, help="Dummy video spatial size.")
    parser.add_argument("--audio_samples", type=int, default=48000, help="Dummy raw waveform length for WavLM exports.")
    parser.add_argument("--audio_time_steps", type=int, default=301, help="Dummy mel time steps for mel-based exports.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--quantize_int8", action="store_true", help="Export an additional INT8 dynamic-quantized ONNX model.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runner = TorchModelRunner(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_dynamic_quant=False,
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = runner.model
    fusion_mode = runner.fusion_mode
    use_wavlm = runner.use_wavlm
    audio_n_mels = int(runner.config.get("audio_n_mels", 64))

    batch = 1
    video = torch.randn(batch, args.frames, 3, args.image_size, args.image_size, device=runner.device)
    if use_wavlm:
        audio = torch.randn(batch, 1, args.audio_samples, device=runner.device)
    else:
        audio = torch.randn(batch, 1, audio_n_mels, args.audio_time_steps, device=runner.device)

    if fusion_mode == "audio":
        model_inputs = (audio,)
        input_names = ["audio"]
        dynamic_axes = {"audio": {0: "batch"}, "output": {0: "batch"}}
    elif fusion_mode == "video":
        model_inputs = (video,)
        input_names = ["video"]
        dynamic_axes = {"video": {0: "batch"}, "output": {0: "batch"}}
    else:
        model_inputs = (video, audio)
        input_names = ["video", "audio"]
        dynamic_axes = {
            "video": {0: "batch"},
            "audio": {0: "batch"},
            "output": {0: "batch"},
        }

    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            model_inputs,
            str(output_path),
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
        )

    meta = {
        "fusion": fusion_mode,
        "num_classes": runner.num_classes,
        "use_wavlm": use_wavlm,
        "labels": runner.labels,
        "source_checkpoint": str(Path(args.checkpoint).expanduser()),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] Exported ONNX model: {output_path}")
    print(f"[INFO] Exported metadata: {meta_path}")

    if args.quantize_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX INT8 quantization.") from exc

        int8_path = output_path.with_name(output_path.stem + "_int8" + output_path.suffix)
        quantize_dynamic(
            model_input=str(output_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        int8_meta_path = int8_path.with_suffix(int8_path.suffix + ".meta.json")
        int8_meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[INFO] Exported INT8 ONNX model: {int8_path}")
        print(f"[INFO] Exported metadata: {int8_meta_path}")


if __name__ == "__main__":
    main()
