# ============================================================
# Eloquium — F5-TTS inference service
# 2-stage build: Python deps → lean runtime
# Python 3.11 (matches voicebox; F5-TTS deps unhappy on 3.12+)
# ============================================================

# === Stage 1: Build Python dependencies ===
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install torch from the CPU index by default. The default PyPI torch wheel
# bundles ~1.5GB of CUDA libraries (cudnn, cusparselt, etc.) we don't need for
# CPU inference. For a GPU build:
#   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --prefix=/install \
        --index-url ${TORCH_INDEX} \
        torch torchaudio

# Make the torch we just installed visible to the next pip run so f5-tts
# resolves it as satisfied instead of pulling the CUDA build from PyPI.
ENV PYTHONPATH=/install/lib/python3.11/site-packages

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# === Stage 2: Runtime ===
FROM python:3.11-slim

# Non-root user for security
RUN groupadd -r eloquium && \
    useradd -r -g eloquium -m -s /bin/bash eloquium

WORKDIR /app

# Runtime system deps only (no compilers): libsndfile for soundfile,
# ffmpeg for audio handling, curl for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# HuggingFace downloads model weights here at runtime — mounted as a volume.
# The model is NEVER baked into the image.
ENV HF_HOME=/cache
# librosa/numba needs a writable cache dir when running non-root
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create the cache dir owned by the non-root user; the named volume mounted
# here inherits this ownership on first creation.
RUN mkdir -p /cache && chown -R eloquium:eloquium /cache

COPY --chown=eloquium:eloquium *.py ./

USER eloquium

EXPOSE 8000

# Health check — auto-restart if the server hangs. Long start period because
# the first run downloads the model into the cache volume.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=300s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
