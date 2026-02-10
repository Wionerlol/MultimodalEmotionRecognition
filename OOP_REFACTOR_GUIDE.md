# OOP Refactor Guide

This project keeps all existing functionality, while organizing core workflows around service/manager classes for easier extension.

## Data Layer (`src/data/ravdess.py`)

- `RavdessPairService`
  - Parse filenames, build audio-video pairs, save pair index CSV, map labels.
- `RavdessSplitService`
  - Actor-based split and stratified split strategies.
- `RavdessMediaService`
  - Video frame loading + augmentation, audio mel/wav loading + augmentation.
- `RavdessDatasetFactory`
  - Centralized dataset type selection (`RavdessAVDataset` vs `RavdessAVDatasetWavLM`).

Default instances:

- `PAIR_SERVICE`
- `SPLIT_SERVICE`
- `MEDIA_SERVICE`
- `DATASET_FACTORY`

## Training Layer (`src/train.py`)

- `EmotionTrainer`
  - Encapsulates end-to-end training orchestration.
  - Keeps existing CLI arguments and behavior.
  - Uses service objects from `src/data/ravdess.py` for lower coupling.

## Evaluation Layer (`src/eval.py`)

- `EmotionEvaluator`
  - Encapsulates checkpoint evaluation flow.
  - Uses the same service-oriented data access pattern as training.

## Export/QA Layer (`src/export_augmented_examples.py`)

- `AugmentedExampleExporter`
  - Encapsulates augmented sample export for quick visual/audio QA.

## Backend Layer (`backend/app/main.py`)

- `EmotionAPIService`
  - Owns predictor lifecycle and prediction request flow.
  - FastAPI routes are thin adapters delegating into this service.

## Frontend Layer (`frontend/app.js`)

- `EmotionRecognitionApp`
  - Encapsulates DOM state, recording flow, upload flow, and rendering.
  - Removes global mutable state and keeps behavior unchanged.

## Backward Compatibility

- Existing CLI entrypoints are preserved:
  - `python src/train.py ...`
  - `python src/eval.py ...`
  - `python src/export_augmented_examples.py ...`
- Existing API endpoints are preserved:
  - `GET /health`
  - `POST /predict`
- Existing dataset/model paths and file formats are unchanged.
