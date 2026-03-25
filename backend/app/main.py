"""FastAPI backend for emotion recognition."""
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import CHECKPOINT_PATH, DEVICE, EMOTION_LABELS, IS_WSL, MOCK_MODE, TEMP_DIR
from .infer import EmotionPredictor
from .streaming import StreamingSessionManager, decode_frame_b64, decode_pcm16_b64

# Initialize FastAPI app
app = FastAPI(title="Emotion Recognition API", version="0.1.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionAPIService:
    """Service state holder for FastAPI lifecycle and prediction handling."""

    def __init__(self) -> None:
        self.predictor: Optional[EmotionPredictor] = None
        self.streaming: Optional[StreamingSessionManager] = None

    async def startup(self) -> None:
        """Initialize model on app startup."""
        mock = os.environ.get("EMO_MOCK", "0") == "1"
        self.predictor = EmotionPredictor(mock_mode=mock)
        self.streaming = StreamingSessionManager(self.predictor)
        print(
            "[INFO] Emotion predictor initialized "
            f"(mock_mode={self.predictor.mock_mode}, device={DEVICE}, wsl={IS_WSL}, checkpoint={CHECKPOINT_PATH})"
        )
        
    def health_payload(self) -> Dict[str, Any]:
        current_mock_mode = self.predictor.mock_mode if self.predictor is not None else MOCK_MODE
        return {
            "status": "ok",
            "mock_mode": current_mock_mode,
            "device": DEVICE,
            "is_wsl": IS_WSL,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_exists": CHECKPOINT_PATH.exists(),
            "num_emotions": len(EMOTION_LABELS),
            "emotion_labels": EMOTION_LABELS,
        }

    async def predict_from_upload(self, file: UploadFile) -> Dict[str, Any]:
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized")

        fd, temp_path = tempfile.mkstemp(suffix=".webm", dir=str(TEMP_DIR))
        os.close(fd)
        try:
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            return self.predictor.predict(temp_path)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    async def handle_stream(self, websocket: WebSocket) -> None:
        if self.predictor is None or self.streaming is None:
            raise RuntimeError("Predictor not initialized")

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


service = EmotionAPIService()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await service.startup()
    except Exception as e:
        print(f"[ERROR] Failed to initialize predictor: {e}")
        service.predictor = EmotionPredictor(mock_mode=True)
        service.streaming = StreamingSessionManager(service.predictor)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return service.health_payload()


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        return await service.predict_from_upload(file)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.websocket("/ws/stream")
async def stream_emotion(websocket: WebSocket) -> None:
    try:
        await service.handle_stream(websocket)
    except RuntimeError as e:
        await websocket.accept()
        await websocket.send_json({"type": "error", "detail": str(e)})
        await websocket.close(code=1011)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Emotion Recognition API",
        "version": "0.1.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict emotion from video",
            "WS /ws/stream": "Streaming emotion inference with sliding window",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
