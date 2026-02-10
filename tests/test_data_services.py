from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ravdess import (
    PairRecord,
    RavdessAVDataset,
    RavdessAVDatasetWavLM,
    RavdessDatasetFactory,
    RavdessPairService,
    RavdessSplitService,
)


class TestRavdessDataServices(unittest.TestCase):
    def setUp(self) -> None:
        self.pair_service = RavdessPairService()
        self.split_service = RavdessSplitService()
        self.dataset_factory = RavdessDatasetFactory()

    @staticmethod
    def _touch_ravdess_file(
        root: Path,
        modality: int,
        emotion: int,
        actor: int,
        vocal_channel: int = 1,
        intensity: int = 1,
        statement: int = 1,
        repetition: int = 1,
    ) -> Path:
        stem = (
            f"{modality:02d}-{vocal_channel:02d}-{emotion:02d}-"
            f"{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}"
        )
        suffix = ".mp4" if modality == 2 else ".wav"
        file_path = root / f"Actor_{actor:02d}" / f"{stem}{suffix}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        return file_path

    def test_build_pairs_and_parse_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)

            self._touch_ravdess_file(data_root, modality=2, emotion=3, actor=1)
            self._touch_ravdess_file(data_root, modality=3, emotion=3, actor=1)
            self._touch_ravdess_file(data_root, modality=2, emotion=4, actor=2)
            self._touch_ravdess_file(data_root, modality=3, emotion=4, actor=2)
            self._touch_ravdess_file(data_root, modality=3, emotion=5, actor=3)  # Unpaired audio.

            pairs = self.pair_service.build_pairs(data_root=data_root, vocal_channel=1)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(sorted([p.emotion for p in pairs]), [3, 4])
            self.assertEqual(sorted([p.actor for p in pairs]), [1, 2])

            parsed = self.pair_service.parse_name("02-01-06-01-02-01-12.mp4")
            self.assertEqual(parsed["emotion"], 6)
            self.assertEqual(parsed["actor"], 12)

    def test_split_service_by_actor(self) -> None:
        pairs = [
            PairRecord(Path("a.mp4"), Path("a.wav"), emotion=3, intensity=1, statement=1, repetition=1, actor=1),
            PairRecord(Path("b.mp4"), Path("b.wav"), emotion=4, intensity=1, statement=1, repetition=1, actor=2),
            PairRecord(Path("c.mp4"), Path("c.wav"), emotion=5, intensity=1, statement=1, repetition=1, actor=3),
        ]

        train, val, test = self.split_service.by_actor(
            pairs=pairs,
            train_actors=[1],
            val_actors=[2],
            test_actors=[3],
        )
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)
        self.assertEqual(train[0].actor, 1)
        self.assertEqual(val[0].actor, 2)
        self.assertEqual(test[0].actor, 3)

    def test_dataset_factory_selects_variant(self) -> None:
        pairs = [
            PairRecord(Path("a.mp4"), Path("a.wav"), emotion=3, intensity=1, statement=1, repetition=1, actor=1)
        ]

        mel_dataset = self.dataset_factory.create(pairs=pairs, use_wavlm=False)
        wav_dataset = self.dataset_factory.create(pairs=pairs, use_wavlm=True)

        self.assertIsInstance(mel_dataset, RavdessAVDataset)
        self.assertIsInstance(wav_dataset, RavdessAVDatasetWavLM)
        self.assertEqual(len(mel_dataset), 1)
        self.assertEqual(len(wav_dataset), 1)


if __name__ == "__main__":
    unittest.main()
