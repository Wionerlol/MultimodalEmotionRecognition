from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import redis
import torch

from data.ravdess import RavdessMediaService
from optimized_runtime import OnnxModelRunner, TorchModelRunner


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WorkerSettings:
    redis_url: str = os.environ.get("EMO_REDIS_URL", "redis://localhost:6379/0")
    queue_key: str = os.environ.get("EMO_REDIS_QUEUE_KEY", "emo:inference:queue")
    checkpoint_path: str = os.environ.get("EMO_CHECKPOINT", "outputs/best_xattn.pt")
    onnx_model_path: str = os.environ.get("EMO_ONNX_MODEL_PATH", "")
    inference_backend: str = os.environ.get("EMO_INFERENCE_BACKEND", "torch")
    fusion: str = os.environ.get("EMO_FUSION", "xattn")
    num_classes: int = int(os.environ.get("EMO_NUM_CLASSES", "8"))
    frames: int = int(os.environ.get("EMO_NUM_FRAMES", "8"))
    batch_size: int = int(os.environ.get("EMO_BATCH_SIZE", "8"))
    batch_timeout_ms: int = int(os.environ.get("EMO_BATCH_TIMEOUT_MS", "20"))
    preprocess_workers: int = int(os.environ.get("EMO_PREPROCESS_WORKERS", "4"))
    use_face_crop: bool = _env_flag("EMO_USE_FACE_CROP", True)
    use_wavlm: bool = _env_flag("EMO_USE_WAVLM", False)
    audio_n_mels: int = int(os.environ.get("EMO_AUDIO_N_MELS", "64"))
    device: str = os.environ.get("EMO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    enable_dynamic_quant: bool = _env_flag("EMO_ENABLE_DYNAMIC_QUANT", False)
    result_ttl_sec: int = int(os.environ.get("EMO_RESULT_TTL_SEC", "3600"))
    idle_timeout_sec: int = int(os.environ.get("EMO_WORKER_IDLE_TIMEOUT_SEC", "1"))
    worker_name: str = os.environ.get("EMO_WORKER_NAME", f"worker-{os.getpid()}")


class RedisBatchWorker:
    def __init__(self, settings: WorkerSettings):
        self.settings = settings
        self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        self.device = torch.device(settings.device)
        self.media = RavdessMediaService()
        self.preprocess_pool = ThreadPoolExecutor(max_workers=max(1, settings.preprocess_workers))
        self.runner, self.fusion_mode, self.labels, self.use_wavlm = self._load_runner()

    def run(self) -> None:
        print(
            f"[INFO] Redis inference worker started: name={self.settings.worker_name}, "
            f"queue={self.settings.queue_key}, batch_size={self.settings.batch_size}, "
            f"device={self.settings.device}, backend={self.settings.inference_backend}"
        )
        while True:
            batch = self._pop_batch()
            if not batch:
                continue
            self._process_batch(batch)

    def shutdown(self) -> None:
        self.preprocess_pool.shutdown(wait=False, cancel_futures=True)
        self.redis.close()

    def _load_runner(self) -> tuple[object, str, list[str], bool]:
        backend = self.settings.inference_backend.lower()
        if backend == "onnx":
            if not self.settings.onnx_model_path:
                raise ValueError("EMO_ONNX_MODEL_PATH is required when EMO_INFERENCE_BACKEND=onnx.")
            runner = OnnxModelRunner(self.settings.onnx_model_path)
            use_wavlm = bool(runner.meta.get("use_wavlm", False))
            return runner, runner.fusion_mode, runner.labels, use_wavlm
        if backend == "torch":
            runner = TorchModelRunner(
                checkpoint_path=self.settings.checkpoint_path,
                device=self.settings.device,
                fallback_fusion=self.settings.fusion,
                enable_dynamic_quant=self.settings.enable_dynamic_quant,
            )
            return runner, runner.fusion_mode, runner.labels, runner.use_wavlm
        raise ValueError(f"Unsupported inference backend: {self.settings.inference_backend}")

    def _pop_batch(self) -> list[str]:
        first = self.redis.blpop(self.settings.queue_key, timeout=max(1, self.settings.idle_timeout_sec))
        if first is None:
            return []

        task_ids = [self._decode(first[1])]
        deadline = time.monotonic() + (self.settings.batch_timeout_ms / 1000.0)
        while len(task_ids) < self.settings.batch_size:
            task_raw = self.redis.lpop(self.settings.queue_key)
            if task_raw is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.001)
                continue
            task_ids.append(self._decode(task_raw))
        return task_ids

    def _process_batch(self, task_ids: list[str]) -> None:
        task_infos = []
        for task_id in task_ids:
            task_key = self._task_key(task_id)
            payload_key = self._payload_key(task_id)
            task_hash = self.redis.hgetall(task_key)
            payload = self.redis.get(payload_key)
            if not task_hash or payload is None:
                self._mark_failed(task_id, "Task payload missing or expired.")
                continue
            info = {self._decode(k): self._decode(v) for k, v in task_hash.items()}
            task_infos.append(
                {
                    "task_id": task_id,
                    "filename": info.get("filename", "upload.mp4"),
                    "submitted_at": float(info.get("submitted_at", str(time.time()))),
                    "payload": payload,
                }
            )

        if not task_infos:
            return

        try:
            prepared = list(self.preprocess_pool.map(self._preprocess_item, task_infos))
            videos = torch.stack([item["video"] for item in prepared], dim=0).to(self.device)
            audios = torch.stack([item["audio"] for item in prepared], dim=0).to(self.device)

            probs = self.runner.predict_probs(videos, audios)

            for row, item in zip(probs, prepared):
                top_idx = int(row.argmax().item())
                result = {
                    "task_id": item["task_id"],
                    "worker_name": self.settings.worker_name,
                    "labels": self.labels,
                    "probs": [round(float(x), 6) for x in row.tolist()],
                    "top1": {"label": self.labels[top_idx], "prob": round(float(row[top_idx].item()), 6)},
                    "queue_delay_ms": round((time.time() - item["submitted_at"]) * 1000.0, 2),
                    "processed_at": time.time(),
                }
                self._mark_completed(item["task_id"], result)
        except Exception as exc:
            for item in task_infos:
                self._mark_failed(item["task_id"], str(exc))

    def _preprocess_item(self, item: dict[str, Any]) -> dict[str, Any]:
        suffix = Path(item["filename"]).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(item["payload"])
            media_path = Path(tmp.name)
        try:
            video = self.media.load_video_frames(
                video_path=media_path,
                num_frames=self.settings.frames,
                augment=False,
                use_face_crop=self.settings.use_face_crop,
            )
            if self.use_wavlm:
                audio = self.media.load_audio_wav(audio_path=media_path, augment=False)
            else:
                audio = self.media.load_audio_mel(
                    audio_path=media_path,
                    n_mels=self.settings.audio_n_mels,
                    augment=False,
                )
            return {
                "task_id": item["task_id"],
                "submitted_at": item["submitted_at"],
                "video": video,
                "audio": audio,
            }
        finally:
            media_path.unlink(missing_ok=True)

    def _mark_completed(self, task_id: str, result: dict[str, Any]) -> None:
        task_key = self._task_key(task_id)
        payload_key = self._payload_key(task_id)
        self.redis.hset(
            task_key,
            mapping={
                "status": "completed",
                "completed_at": str(time.time()),
                "result": json.dumps(result, ensure_ascii=True),
            },
        )
        self.redis.expire(task_key, self.settings.result_ttl_sec)
        self.redis.delete(payload_key)

    def _mark_failed(self, task_id: str, error: str) -> None:
        task_key = self._task_key(task_id)
        payload_key = self._payload_key(task_id)
        self.redis.hset(
            task_key,
            mapping={
                "status": "failed",
                "failed_at": str(time.time()),
                "error": error,
            },
        )
        self.redis.expire(task_key, self.settings.result_ttl_sec)
        self.redis.delete(payload_key)

    @staticmethod
    def _decode(value: bytes | str) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"emo:task:{task_id}"

    @staticmethod
    def _payload_key(task_id: str) -> str:
        return f"emo:task:{task_id}:payload"


def main() -> None:
    worker = RedisBatchWorker(WorkerSettings())
    try:
        worker.run()
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
