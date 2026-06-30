# TTS Service

A self-hosted, multilingual Text-to-Speech service targeting production-grade audio quality on par with ElevenLabs. Designed for personal use first, with a clear path toward a commercial SaaS product.

## What This Is

This service converts written text into natural-sounding speech in **Russian** and **English**. Each language can use a dedicated voice. The core model is fine-tuned from an open-source TTS base and extended with Russian phonetic preprocessing (accent/stress normalization) to address the quality gap that plagues most open-source Russian TTS solutions.

The system is built to run on modest local hardware for inference, while training and fine-tuning are offloaded to rented cloud GPUs.

## Quickstart

Requires Docker (with Compose). The Makefile auto-detects an NVIDIA GPU and
selects the matching torch build (CUDA vs CPU) and compose config — the same
commands work on a GPU box or a CPU-only machine.

```bash
git clone <repo> && cd eloquium

make gpu-setup     # only if the machine has an NVIDIA GPU (installs the NVIDIA
                   # Container Toolkit on the host; uses sudo, one-time)

make run           # build the image, start the service, wait until it's healthy
                   # (the first run downloads the model weights — can take a few
                   #  minutes; they persist in a Docker volume afterwards)
```

Generate audio — `LNG` is `en` or `ru` (defaults to `en`). It prints the path to
the resulting `.wav`:

```bash
make gen TEXT="Привет, мир. Как дела?" LNG=ru
# -> /path/to/eloquium/out/tts_20260630_121115.wav
```

Or call the API directly:

```bash
curl -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, world.", "language": "en"}' \
  --output hello.wav
```

**API contract** — `POST /tts` takes `{text, language}` (`language` ∈ `en`, `ru`)
and returns `audio/wav`. `GET /health` returns `{status, device}` (device is
`cuda` or `cpu`).

Other useful targets: `make logs`, `make health`, `make gpu-check`, `make down`.

## Goals

- Achieve near-ElevenLabs quality for both Russian and English
- Support voice cloning from short audio samples
- Run inference locally on CPU / low-VRAM GPU
- Expose a simple API suitable for a future SaaS layer
- Keep the model architecture open and fine-tunable

## Architecture Overview

```
Input Text
    │
    ▼
Text Preprocessor
  (accent normalization for RU, G2P, punctuation cleanup)
    │
    ▼
TTS Core Model
  (fine-tuned base model, bilingual: RU + EN)
    │
    ▼
Post-Processor
  (denoiser, normalizer)
    │
    ▼
Audio Output (.wav / .mp3)
```

## Development Phases

| Phase | Description | Infrastructure |
|---|---|---|
| 1. Research | Model selection, dataset collection | Local PC (inference only) |
| 2. Fine-tuning | Train on RU+EN datasets | Cloud GPU (RunPod / Vast.ai / Colab Pro) |
| 3. Personal use | Self-hosted inference | Local PC (CPU / GTX 1050) |
| 4. SaaS | Public API + web UI | Rented GPU server |

## Training Data

- **English** — Common Voice EN, LibriTTS, LibriHeavy
- **Russian** — Common Voice RU, RUSLAN, Balalaika (2,000h studio-quality RU), Sova (RuDevices + RuAudiobooks)
- Custom voice recordings for unique proprietary voice identities

## Tech Stack

### Core Model
| Component | Choice | Notes |
|---|---|---|
| Base TTS model | **Fish Speech S2 / F5-TTS** | F5-TTS for fine-tuning flexibility; S2 as quality benchmark. Both open-source, Apache 2.0 / Fish Audio Research License |
| Architecture | Dual-AR (Slow AR 4B + Fast AR 400M) | Structurally isomorphic to LLMs; inherits all LLM serving optimizations |

### Russian Language Pipeline
| Component | Choice | Notes |
|---|---|---|
| Stress / accent normalization | **RUAccent** | 0.97 accuracy on standard words, 0.96 on homographs (COLING 2025) |
| Text normalization | **sova-tts-tps** | Modular RU+EN text preprocessor, handles numbers, abbreviations, punctuation |

### Training Infrastructure
| Component | Choice | Notes |
|---|---|---|
| Framework | **PyTorch** | — |
| Training platform | **Google Colab Pro / RunPod / Vast.ai** | Rented GPU (A100 / RTX 4090); ~$0.3–1/hr |
| Local hardware | GTX 1050 3GB | Inference testing only; not used for training |

### Inference & Serving
| Component | Choice | Notes |
|---|---|---|
| Inference engine | **SGLang** | Production-ready streaming; RTF 0.195 on H200; continuous batching, paged KV cache |
| Min VRAM (inference) | 12 GB | 24 GB recommended for production |
| API layer | **FastAPI** (Python) | — |
| Audio output | WAV / MP3 | — |

### Production (SaaS phase)
| Component | Choice | Notes |
|---|---|---|
| Hosting | Rented GPU server | Min 12 GB VRAM |
| Frontend | — | TBD |
| Auth / billing | — | TBD |

---

*This README is a living document and will be updated as the stack is validated.*
