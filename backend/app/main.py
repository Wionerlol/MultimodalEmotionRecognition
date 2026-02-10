"""FastAPI backend for emotion recognition."""
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import CHECKPOINT_PATH, DEVICE, EMOTION_LABELS, IS_WSL, MOCK_MODE, TEMP_DIR
from .infer import EmotionPredictor

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

    async def startup(self) -> None:
        """Initialize model on app startup."""
        mock = os.environ.get("EMO_MOCK", "0") == "1"
        self.predictor = EmotionPredictor(mock_mode=mock)
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


service = EmotionAPIService()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await service.startup()
    except Exception as e:
        print(f"[ERROR] Failed to initialize predictor: {e}")
        service.predictor = EmotionPredictor(mock_mode=True)


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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Emotion Recognition API",
        "version": "0.1.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict emotion from video",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
