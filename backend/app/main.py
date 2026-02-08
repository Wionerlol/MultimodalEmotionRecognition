"""FastAPI backend for emotion recognition."""
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

from .config import EMOTION_LABELS, MOCK_MODE, TEMP_DIR
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

# Global predictor instance
predictor: Optional[EmotionPredictor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor
    try:
        mock = os.environ.get("EMO_MOCK", "0") == "1"
        predictor = EmotionPredictor(mock_mode=mock)
        print(f"[INFO] Emotion predictor initialized (mock_mode={mock})")
    except Exception as e:
        print(f"[ERROR] Failed to initialize predictor: {e}")
        # Still allow mock mode for testing
        predictor = EmotionPredictor(mock_mode=True)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "num_emotions": len(EMOTION_LABELS),
        "emotion_labels": EMOTION_LABELS,
    }


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded video.
    
    Args:
        file: Video file (webm, mp4, etc.)
    
    Returns:
        JSON with emotion predictions
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    
    # Save uploaded file to temp directory
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".webm", dir=str(TEMP_DIR))
        os.close(fd)
        
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Run inference
        result = predictor.predict(temp_path)
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        return result
    
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
