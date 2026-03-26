FROM python:3.10-slim

# Install runtime system dependencies for OpenCV / ffmpeg.
# Use non-dev runtime libs to reduce image size and avoid package-resolution issues.
ENV DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi && \
    if [ -d /etc/apt/sources.list.d ]; then \
        find /etc/apt/sources.list.d -type f -name '*.sources' -o -name '*.list' | while read -r file; do \
            sed -i "s|http://deb.debian.org/debian|https://${APT_MIRROR}/debian|g; s|http://security.debian.org/debian-security|https://${APT_MIRROR}/debian-security|g; s|https://deb.debian.org/debian|https://${APT_MIRROR}/debian|g; s|https://security.debian.org/debian-security|https://${APT_MIRROR}/debian-security|g" "$file"; \
        done; \
    fi
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    -i ${PIP_INDEX_URL} \
    --trusted-host ${PIP_TRUSTED_HOST} \
    -r requirements.txt

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
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONPATH=/app:/app/src

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
