from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.model_loader import DummyModel, ModelLoaderService, _InferenceModelAdapter
from backend.app.preprocess import EmotionPreprocessService
from backend.app.streaming import StreamingEmotionSession


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_state_dict = None

    def forward(self, *args, **kwargs):
        return torch.zeros((1, 8))

    def load_state_dict(self, state_dict, strict: bool = False):
        self.loaded_state_dict = state_dict
        return [], []


class _FakePredictor:
    def __init__(self) -> None:
        self.calls = []

    def predict_stream(self, frames, waveform, waveform_sample_rate: int, use_face_crop: bool = True):
        self.calls.append((len(frames), waveform.shape[0], waveform_sample_rate, use_face_crop))
        return {
            "labels": ["neutral"],
            "probs": [100.0],
            "top1": {"label": "neutral", "prob": 100.0},
        }


class TestEmotionPreprocessService(unittest.TestCase):
    def test_uniform_indices_and_crop(self) -> None:
        service = EmotionPreprocessService()
        self.assertEqual(service.uniform_indices(0, 4), [0, 0, 0, 0])
        self.assertEqual(service.uniform_indices(2, 4), [0, 1, 1, 1])

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = service.crop_with_padding(frame, (40, 40, 60, 60), pad_ratio=0.5)
        self.assertEqual(cropped.shape, (40, 40, 3))

    def test_preprocess_video_audio_mel_cleanup(self) -> None:
        service = EmotionPreprocessService()
        fake_video = torch.zeros((8, 3, 112, 112))
        fake_mel = torch.zeros((1, 64, 32))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        with mock.patch.object(service, "load_video_frames", return_value=fake_video), mock.patch.object(
            service, "extract_audio_from_video", return_value=temp_audio_path
        ), mock.patch.object(service, "load_audio_mel", return_value=fake_mel):
            video, audio = service.preprocess_video_audio("dummy.mp4", use_wavlm=False)

        self.assertEqual(tuple(video.shape), (1, 8, 3, 112, 112))
        self.assertEqual(tuple(audio.shape), (1, 1, 64, 32))
        self.assertFalse(Path(temp_audio_path).exists())

    def test_preprocess_video_audio_wavlm_path(self) -> None:
        service = EmotionPreprocessService()
        fake_video = torch.zeros((8, 3, 112, 112))
        fake_wav = torch.zeros((1, 16000))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        with mock.patch.object(service, "load_video_frames", return_value=fake_video), mock.patch.object(
            service, "extract_audio_from_video", return_value=temp_audio_path
        ), mock.patch.object(service, "load_audio_wav", return_value=fake_wav):
            video, audio = service.preprocess_video_audio("dummy.mp4", use_wavlm=True)

        self.assertEqual(tuple(video.shape), (1, 8, 3, 112, 112))
        self.assertEqual(tuple(audio.shape), (1, 1, 16000))
        self.assertFalse(Path(temp_audio_path).exists())


class TestModelLoaderService(unittest.TestCase):
    def test_signature_and_wavlm_detection(self) -> None:
        state_dict = {
            "audio_model.wavlm.encoder.layer_norm.weight": torch.zeros(1),
            "video_model.backbone.conv1.weight": torch.zeros(1),
            "xattn_mlp.proj.weight": torch.zeros(1),
        }
        fusion_mode, xattn_head = ModelLoaderService.infer_model_signature(state_dict)
        self.assertEqual((fusion_mode, xattn_head), ("xattn", "concat"))
        self.assertTrue(ModelLoaderService.checkpoint_uses_wavlm(state_dict))

    def test_get_model_mock_and_error_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_ckpt = Path(temp_dir) / "missing.pt"
            service = ModelLoaderService(checkpoint_path=missing_ckpt)

            model = service.get_model(num_classes=8, allow_mock=True)
            self.assertIsInstance(model, DummyModel)

            with self.assertRaisesRegex(RuntimeError, "Checkpoint not found"):
                service.get_model(num_classes=8, allow_mock=False)

    def test_load_model_from_checkpoint_builds_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = Path(temp_dir) / "best.pt"
            ckpt_path.touch()
            service = ModelLoaderService(checkpoint_path=ckpt_path)
            state_dict = {
                "audio_model.wavlm.encoder.layer_norm.weight": torch.zeros(1),
                "video_model.backbone.conv1.weight": torch.zeros(1),
                "xattn_mlp.proj.weight": torch.zeros(1),
            }
            fake_model = _FakeModel()

            with mock.patch.dict(os.environ, {"MODEL_FUSION": "", "MODEL_XATTN_HEAD": "concat"}, clear=False), mock.patch(
                "backend.app.model_loader.torch.load", return_value={"model": state_dict}
            ), mock.patch.object(service, "build_model", return_value=fake_model) as build_mock:
                loaded = service.load_model_from_checkpoint(ckpt_path, num_classes=8)

            self.assertIsInstance(loaded, _InferenceModelAdapter)
            self.assertEqual(loaded.mode, "xattn")
            self.assertTrue(loaded.requires_wavlm)
            self.assertIs(loaded.model, fake_model)
            build_mock.assert_called_once_with(
                num_classes=8,
                fusion_mode="xattn",
                xattn_head="concat",
                use_wavlm=True,
                config={},
            )


class TestStreamingEmotionSession(unittest.TestCase):
    def test_session_requires_enough_audio_and_frames(self) -> None:
        predictor = _FakePredictor()
        session = StreamingEmotionSession(predictor=predictor, window_seconds=3.0, step_seconds=0.5)

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        session.add_frame(frame, timestamp=1.0)
        session.add_frame(frame, timestamp=2.0)
        self.assertFalse(session.ready_for_inference(now=2.0))

        audio = np.zeros(48000, dtype=np.float32)
        session.add_audio_chunk(audio, sample_rate=16000, timestamp=2.0)
        self.assertTrue(session.ready_for_inference(now=2.0))

    def test_session_builds_sliding_window_and_updates_cadence(self) -> None:
        predictor = _FakePredictor()
        session = StreamingEmotionSession(
            predictor=predictor,
            window_seconds=3.0,
            step_seconds=0.5,
            max_buffer_seconds=6.0,
        )
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        for ts in (1.0, 2.0, 3.5, 4.0):
            session.add_frame(frame, timestamp=ts)
        session.add_audio_chunk(np.zeros(32000, dtype=np.float32), sample_rate=16000, timestamp=2.0)
        session.add_audio_chunk(np.zeros(32000, dtype=np.float32), sample_rate=16000, timestamp=4.0)

        result = session.infer(now=4.0)
        self.assertEqual(result["top1"]["label"], "neutral")
        self.assertEqual(result["num_buffered_frames"], 2)
        self.assertFalse(session.ready_for_inference(now=4.2))
        self.assertTrue(session.ready_for_inference(now=4.6))


if __name__ == "__main__":
    unittest.main()
