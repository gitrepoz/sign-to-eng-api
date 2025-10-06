# Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# System libs needed by OpenCV/MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better build cache)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY server.py ./server.py
COPY model/ ./model/

# Default model path; you can override at runtime
ENV MODEL_WEIGHTS=/app/model/updated15words.h5

EXPOSE 8765

CMD ["python", "server.py"]
