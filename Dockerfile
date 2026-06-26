FROM python:3.11-slim

# System deps: libsndfile for soundfile, ffmpeg for audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HuggingFace downloads model weights here at runtime — mounted as a volume.
# The model is NEVER baked into the image.
ENV HF_HOME=/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
