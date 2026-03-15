# BeanHealth CLR Tool — FastAPI Backend
# Optimised for Railway deployment with OpenCV + MediaPipe

FROM python:3.11-slim

# System dependencies required by OpenCV (headless) and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
# mediapipe pulls opencv-contrib-python; forcibly remove any conflicting
# opencv-python-headless afterwards to guarantee a single cv2 installation.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python-headless 2>/dev/null || true

# Copy application code
COPY . .

# Railway injects $PORT automatically
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
