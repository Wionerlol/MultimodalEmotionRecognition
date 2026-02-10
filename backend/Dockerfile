FROM python:3.10-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + model source code
COPY backend/app/ ./app/
COPY src/ ./src/

# Create temp directory for uploaded files
RUN mkdir -p /app/tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_GPU=0
ENV EMO_MOCK=0
ENV HF_LOCAL_ONLY=1
ENV PYTHONPATH=/app:/app/src

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
