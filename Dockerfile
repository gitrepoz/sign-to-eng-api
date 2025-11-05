# ========= Base =========
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_WEIGHTS=/app/model/4426_model.pt

# ========= System Dependencies =========
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ========= Working Directory =========
WORKDIR /app

# ========= Python Dependencies =========
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ========= Copy Application Code =========
COPY . /app

# Ensure Python can find all modules (like utils/)
ENV PYTHONPATH=/app

# ========= Expose Port & Run =========
EXPOSE 8765
CMD ["python", "app.py"]
