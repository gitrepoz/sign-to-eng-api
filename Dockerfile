# ========= Base =========
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_WEIGHTS=/app/model_30.h5

# ========= System Dependencies =========
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ========= App =========
WORKDIR /app

# Install Python dependencies first (better build cache)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY app.py ./app.py
COPY model/ ./model/

ENV MODEL_WEIGHTS=/app/model/model_30.h5

EXPOSE 8765

CMD ["python", "app.py"]
