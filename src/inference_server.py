from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as redis_async

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
for candidate in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    candidate_text = str(candidate)
    if candidate.exists() and candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from app.infer import EmotionPredictor
    from app.streaming import StreamingSessionManager, decode_frame_b64, decode_pcm16_b64
except ModuleNotFoundError as exc:
    if exc.name != "app":
        raise
    from backend.app.infer import EmotionPredictor
    from backend.app.streaming import StreamingSessionManager, decode_frame_b64, decode_pcm16_b64


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerSettings:
    redis_url: str = os.environ.get("EMO_REDIS_URL", "redis://localhost:6379/0")
    queue_key: str = os.environ.get("EMO_REDIS_QUEUE_KEY", "emo:inference:queue")
    result_ttl_sec: int = int(os.environ.get("EMO_RESULT_TTL_SEC", "3600"))
    payload_ttl_sec: int = int(os.environ.get("EMO_PAYLOAD_TTL_SEC", "600"))
    predict_timeout_sec: float = float(os.environ.get("EMO_PREDICT_TIMEOUT_SEC", "60"))
    poll_interval_ms: int = int(os.environ.get("EMO_POLL_INTERVAL_MS", "50"))
    batch_size: int = int(os.environ.get("EMO_BATCH_SIZE", "8"))
    batch_timeout_ms: int = int(os.environ.get("EMO_BATCH_TIMEOUT_MS", "20"))
    worker_count: int = int(os.environ.get("EMO_WORKER_COUNT", "1"))
    redis_healthcheck_enabled: bool = _env_flag("EMO_REDIS_HEALTHCHECK", True)


class RedisInferenceGateway:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.redis: Optional[redis_async.Redis] = None
        self.started_at = time.time()

    async def start(self) -> None:
        self.redis = redis_async.from_url(self.settings.redis_url, decode_responses=False)
        if self.settings.redis_healthcheck_enabled:
            await self.redis.ping()

    async def shutdown(self) -> None:
        if self.redis is not None:
            await self.redis.close()
            self.redis = None

    async def submit(self, filename: str, payload: bytes) -> str:
        redis = self._client()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        task_id = str(uuid.uuid4())
        now = str(time.time())
        task_key = self._task_key(task_id)
        payload_key = self._payload_key(task_id)
        await redis.hset(
            task_key,
            mapping={
                "status": "queued",
                "filename": filename or "upload.mp4",
                "submitted_at": now,
            },
        )
        await redis.expire(task_key, self.settings.result_ttl_sec)
        await redis.set(payload_key, payload, ex=self.settings.payload_ttl_sec)
        await redis.rpush(self.settings.queue_key, task_id)
        return task_id

    async def submit_many(self, items: list[tuple[str, bytes]]) -> list[str]:
        task_ids = []
        for filename, payload in items:
            task_ids.append(await self.submit(filename, payload))
        return task_ids

    async def get_result(self, task_id: str) -> dict[str, Any]:
        redis = self._client()
        payload = await redis.hgetall(self._task_key(task_id))
        if not payload:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        decoded = {self._decode(k): self._decode(v) for k, v in payload.items()}
        result_text = decoded.get("result")
        if result_text:
            decoded["result"] = json.loads(result_text)
        return decoded

    async def wait_for_result(self, task_id: str, timeout_sec: Optional[float] = None) -> dict[str, Any]:
        timeout = self.settings.predict_timeout_sec if timeout_sec is None else float(timeout_sec)
        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            result = await self.get_result(task_id)
            status = result.get("status")
            if status == "completed":
                return result["result"]
            if status == "failed":
                detail = result.get("error", "Inference failed.")
                raise HTTPException(status_code=500, detail=detail)
            if time.monotonic() >= deadline:
                raise HTTPException(status_code=202, detail={"task_id": task_id, "status": status})
            await asyncio.sleep(self.settings.poll_interval_ms / 1000.0)

    async def queue_stats(self) -> dict[str, Any]:
        redis = self._client()
        queue_size = await redis.llen(self.settings.queue_key)
        return {
            "redis_url": self.settings.redis_url,
            "queue_key": self.settings.queue_key,
            "queue_size": queue_size,
            "batch_size": self.settings.batch_size,
            "batch_timeout_ms": self.settings.batch_timeout_ms,
            "worker_count_hint": self.settings.worker_count,
            "uptime_sec": round(time.time() - self.started_at, 2),
        }

    def _client(self) -> redis_async.Redis:
        if self.redis is None:
            raise HTTPException(status_code=503, detail="Redis gateway not ready.")
        return self.redis

    @staticmethod
    def _decode(value: bytes | str) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"emo:task:{task_id}"

    @staticmethod
    def _payload_key(task_id: str) -> str:
        return f"emo:task:{task_id}:payload"


class StreamingInferenceService:
    def __init__(self) -> None:
        mock = os.environ.get("EMO_MOCK", "0") == "1"
        self.predictor = EmotionPredictor(mock_mode=mock)
        self.streaming = StreamingSessionManager(self.predictor)

    async def handle_stream(self, websocket: WebSocket) -> None:
        await websocket.accept()
        session = self.streaming.create_session(use_face_crop=True)
        await websocket.send_json({"type": "session_started", "session_id": session.session_id})

        try:
            while True:
                payload = await websocket.receive_json()
                msg_type = str(payload.get("type", "")).lower()

                if msg_type == "start":
                    await websocket.send_json({"type": "ack", "session_id": session.session_id})
                    continue

                if msg_type == "frame":
                    frame = decode_frame_b64(str(payload["image_b64"]))
                    session.add_frame(frame, timestamp=payload.get("timestamp"))
                    if session.ready_for_inference():
                        result = session.infer()
                        await websocket.send_json({"type": "prediction", "payload": result})
                    continue

                if msg_type == "audio":
                    audio = decode_pcm16_b64(str(payload["pcm_b64"]))
                    session.add_audio_chunk(
                        audio,
                        sample_rate=int(payload.get("sample_rate", 16000)),
                        timestamp=payload.get("timestamp"),
                    )
                    if session.ready_for_inference():
                        result = session.infer()
                        await websocket.send_json({"type": "prediction", "payload": result})
                    continue

                if msg_type == "flush":
                    if session.audio_sample_count > 0 and session.frames:
                        result = session.infer()
                        await websocket.send_json({"type": "prediction", "payload": result})
                    continue

                if msg_type == "stop":
                    await websocket.send_json({"type": "session_stopped", "session_id": session.session_id})
                    return

                await websocket.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})
        except WebSocketDisconnect:
            pass
        finally:
            self.streaming.close_session(session.session_id)


settings = ServerSettings()
gateway: Optional[RedisInferenceGateway] = None
streaming_service: Optional[StreamingInferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gateway, streaming_service
    gateway = RedisInferenceGateway(settings)
    await gateway.start()
    try:
        streaming_service = StreamingInferenceService()
        print("[INFO] Streaming inference service initialized.")
    except Exception as exc:
        streaming_service = None
        print(f"[ERROR] Failed to initialize streaming inference service: {exc}")
    try:
        yield
    finally:
        if gateway is not None:
            await gateway.shutdown()


app = FastAPI(title="Multimodal Emotion Redis Inference", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    if gateway is None:
        raise HTTPException(status_code=503, detail="Redis gateway not ready.")
    stats = await gateway.queue_stats()
    return {"status": "ok", "streaming_ready": streaming_service is not None, **stats}


@app.get("/queue/status")
async def queue_status() -> dict[str, Any]:
    if gateway is None:
        raise HTTPException(status_code=503, detail="Redis gateway not ready.")
    return await gateway.queue_stats()


@app.post("/submit")
async def submit(file: UploadFile = File(...)) -> dict[str, Any]:
    if gateway is None:
        raise HTTPException(status_code=503, detail="Redis gateway not ready.")
    payload = await file.read()
    task_id = await gateway.submit(file.filename or "upload.mp4", payload)
    return {"task_id": task_id, "status": "queued"}


@app.get("/result/{task_id}")
async def result(task_id: str) -> dict[str, Any]:
    if gateway is None:
        raise HTTPException(status_code=503, detail="Redis gateway not ready.")
    return await gateway.get_result(task_id)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if gateway is None:
        raise HTTPException(status_code=503, detail="Redis gateway not ready.")
    payload = await file.read()
    task_id = await gateway.submit(file.filename or "upload.mp4", payload)
    result_payload = await gateway.wait_for_result(task_id)
    result_payload["task_id"] = task_id
    return result_payload


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if gateway is None:
        raise HTTPException(status_code=503, detail="Redis gateway not ready.")
    items = []
    for upload in files:
        payload = await upload.read()
        items.append((upload.filename or "upload.mp4", payload))
    task_ids = await gateway.submit_many(items)
    results = await asyncio.gather(*(gateway.wait_for_result(task_id) for task_id in task_ids))
    for task_id, result_payload in zip(task_ids, results):
        result_payload["task_id"] = task_id
    return {"count": len(results), "results": results}


@app.websocket("/ws/stream")
async def stream_emotion(websocket: WebSocket) -> None:
    if streaming_service is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "detail": "Streaming service not ready."})
        await websocket.close(code=1011)
        return
    await streaming_service.handle_stream(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000, reload=False, app_dir="src")
