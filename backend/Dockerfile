# Production container for Precision Aerosol Spray Can Sound Detector Backend
FROM python:3.11-slim

# Install system audio dependencies (libsndfile for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and datasets
COPY . .

# Runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
EXPOSE 5000

# Start Gunicorn server with threaded workers for real-time audio analysis
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 app:app"]
