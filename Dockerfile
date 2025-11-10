# ===============================
#        Base Image
# ===============================
FROM python:3.10-slim

# Prevent interactive prompts & speed up Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# ===============================
#    System Dependencies
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
#        Working Directory
# ===============================
WORKDIR /app

# ===============================
#   Install Python Dependencies
# ===============================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

<<<<<<< HEAD
# ===============================
#        Copy Application
# ===============================
# Copy everything including app.py, model/, etc.
COPY . .
=======
# Copy app code and model
COPY . /app
>>>>>>> 1228975db138ab8cde19c30b4147b09203e70e7a

# ===============================
#     Environment Variables
# ===============================
# Make model path configurable
ENV MODEL_WEIGHTS=/app/model/4426_model.pt

# Optional: debug print
RUN echo "Model path set to: $MODEL_WEIGHTS"

# ===============================
#          Expose Port
# ===============================
EXPOSE 8765

# ===============================
#          Entrypoint
# ===============================
CMD ["python", "app.py"]
