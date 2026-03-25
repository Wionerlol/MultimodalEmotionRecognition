from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.models.fusion import FusionModel
from src.models.audio import AudioNet
from src.models.temporal import TemporalPooler
from src.train import build_dataloaders


class _DummyBackbone(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = torch.nn.Conv2d(3, embedding_dim, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class _DummyVideoModel(torch.nn.Module):
    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = _DummyBackbone(embedding_dim)


class _DummyAudioModel(torch.nn.Module):
    def __init__(self, sequence_dim: int = 8) -> None:
        super().__init__()
        self.sequence_dim = sequence_dim
        self.called = False
        self.proj = torch.nn.Linear(1, sequence_dim)

    def encode_sequence(self, audio: torch.Tensor) -> torch.Tensor:
        self.called = True
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        seq = audio.unsqueeze(-1)  # [B, T, 1]
        return self.proj(seq)  # [B, T, D]


class _ToyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        video = torch.zeros((8, 3, 112, 112), dtype=torch.float32)
        audio = torch.zeros((1, 64, 301), dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)
        meta = {"idx": index}
        return video, audio, label, meta


class TestAttentionIntegration(unittest.TestCase):
    def test_temporal_pooler_modes_preserve_batch_and_channel_dims(self) -> None:
        x = torch.randn(2, 5, 8)
        for mode in ("mean", "attn", "transformer"):
            with self.subTest(mode=mode):
                pooler = TemporalPooler(dim=8, mode=mode, num_heads=2, num_layers=1, dropout=0.0)
                pooled = pooler(x)
                self.assertEqual(tuple(pooled.shape), (2, 8))

    def test_xattn_uses_audio_encode_sequence(self) -> None:
        audio_model = _DummyAudioModel(sequence_dim=8)
        video_model = _DummyVideoModel(embedding_dim=16)
        model = FusionModel(
            audio_model=audio_model,
            video_model=video_model,
            num_classes=8,
            mode="xattn",
            d_model=8,
            num_heads=2,
            audio_n_mels=64,
        )
        model.eval()

        video = torch.randn(2, 4, 3, 16, 16)
        audio = torch.randn(2, 1, 12)
        with torch.no_grad():
            logits = model(video, audio)

        self.assertTrue(audio_model.called, "xAttn should consume audio_model.encode_sequence()")
        self.assertEqual(tuple(logits.shape), (2, 8))

    def test_xattn_supports_temporal_transformer_pooling(self) -> None:
        audio_model = _DummyAudioModel(sequence_dim=8)
        video_model = _DummyVideoModel(embedding_dim=16)
        model = FusionModel(
            audio_model=audio_model,
            video_model=video_model,
            num_classes=8,
            mode="xattn",
            d_model=8,
            num_heads=2,
            audio_n_mels=64,
            temporal_pooling="transformer",
            temporal_num_heads=2,
            temporal_num_layers=1,
            temporal_dropout=0.0,
        )
        model.eval()

        video = torch.randn(2, 4, 3, 16, 16)
        audio = torch.randn(2, 1, 12)
        with torch.no_grad():
            logits = model(video, audio)

        self.assertEqual(tuple(logits.shape), (2, 8))

    def test_mel_audio_net_exposes_temporal_sequence(self) -> None:
        model = AudioNet(
            num_classes=8,
            use_resnet=False,
            spec_augment=False,
            temporal_pooling="attn",
            temporal_num_heads=2,
            temporal_num_layers=1,
            temporal_dropout=0.0,
        )
        mel = torch.randn(2, 1, 64, 301)
        with torch.no_grad():
            seq = model.encode_sequence(mel)
            emb = model.encode(mel)

        self.assertGreater(seq.shape[1], 1)
        self.assertEqual(tuple(seq.shape[:1] + seq.shape[2:]), (2, 128))
        self.assertEqual(tuple(emb.shape), (2, 128))

    def test_build_dataloaders_passes_stratified_ratios(self) -> None:
        fake_pairs = [
            SimpleNamespace(actor=1, emotion=3, intensity=1, statement=1, repetition=1),
            SimpleNamespace(actor=2, emotion=4, intensity=1, statement=1, repetition=1),
            SimpleNamespace(actor=3, emotion=5, intensity=1, statement=1, repetition=1),
            SimpleNamespace(actor=4, emotion=6, intensity=1, statement=1, repetition=1),
        ]
        split_result = (fake_pairs[:2], fake_pairs[2:3], fake_pairs[3:])

        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            with mock.patch("src.train.PAIR_SERVICE.build_pairs", return_value=fake_pairs), mock.patch(
                "src.train.PAIR_SERVICE.save_pairs_csv"
            ), mock.patch("src.train.DATASET_FACTORY.create", return_value=_ToyDataset()), mock.patch(
                "src.train.PAIR_SERVICE.map_emotion_label", return_value=0
            ), mock.patch("src.train.SPLIT_SERVICE.stratified", return_value=split_result) as stratified_mock:
                build_dataloaders(
                    data_root=data_root,
                    num_classes=8,
                    num_frames=8,
                    train_actors=[],
                    val_actors=[],
                    test_actors=[],
                    batch_size=2,
                    device=torch.device("cpu"),
                    requested_num_workers=0,
                    stratified=True,
                    train_ratio=0.6,
                    val_ratio=0.2,
                    use_wavlm=False,
                    train_augment=True,
                    use_face_crop=False,
                )

        stratified_mock.assert_called_once()
        self.assertAlmostEqual(stratified_mock.call_args.kwargs["train_ratio"], 0.6)
        self.assertAlmostEqual(stratified_mock.call_args.kwargs["val_ratio"], 0.2)
        self.assertAlmostEqual(stratified_mock.call_args.kwargs["test_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
