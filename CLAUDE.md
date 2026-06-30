# Eloquium — Project Context for Claude Code

## What This Is

A self-hosted, multilingual TTS service targeting near-ElevenLabs quality for
**Russian and English**. Built for personal use first, with a path to SaaS. The
model is fine-tuned from an open-source base on cloud GPUs; inference runs
locally on modest hardware.

## Current Development Phase

**Phase 1 — Bootstrap**: an inference service around the **base F5-TTS model**.
Training has not started. When the cloud-trained checkpoint is ready, it replaces
the base model by setting `F5_CHECKPOINT` (see "Swapping the trained model").

## Architecture

```
POST /tts {text, language}
        │
        ▼
  Text Preprocessor       ← RUAccent stress norm for 'ru' (TODO, hook in place)
        │
        ▼
  F5-TTS inference        ← weights download to /cache volume on first start
        │
        ▼
  WAV response
```

## Tech Stack (decided — do not change without explicit instruction)

| Layer | Choice | Reason |
|---|---|---|
| Base TTS model | **F5-TTS** (`SWivid/F5-TTS`, HuggingFace) | Fine-tuning flexibility, Apache 2.0 |
| API | **FastAPI** + uvicorn | Async, self-documenting |
| Container | **Docker** + docker-compose | Model lives in a volume, not the image |
| RU preprocessing | **RUAccent** + sova-tts-tps | TODO; `language` param already threaded through |
| Training infra | Cloud GPU (RunPod / Vast.ai / Colab Pro) | Not local — GTX 1050 is inference-only |

## Repository Layout

```
eloquium/
├── CLAUDE.md            ← this file
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py             ← FastAPI app + lifespan loader
└── model.py            ← TTSEngine wrapper around F5TTS
```

Training/dataset/preprocessing tooling goes in sibling dirs (`training/`,
`preprocessing/`) when created — keep the inference service at the root.

## Service Design Rules

1. **Model is never baked into the image.** Weights download to `/cache` (named
   Docker volume) via `HF_HOME=/cache` on first startup. The Dockerfile must
   never download the model at build time.
2. **API contract is stable**: `POST /tts` takes `{text, language}` and returns
   `audio/wav`. Don't change this shape without updating this file.
3. **`language` is a hook, not dead code.** Already accepted and passed through
   `TTSEngine.synthesize(text, language)`. RU preprocessing plugs in there — no
   API change needed.
4. **Reference audio ≠ model.** `F5_REF_AUDIO` defines the default voice;
   `F5_CHECKPOINT` defines quality/language ability. Different concerns.
5. **GPU is optional.** Service auto-detects CUDA, falls back to CPU. The
   docker-compose GPU block is commented out.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `/cache` | Where HuggingFace downloads weights |
| `NUMBA_CACHE_DIR` | `/tmp/numba_cache` | Writable numba cache (librosa needs it as non-root) |
| `F5_REF_AUDIO` | bundled `basic_ref_en.wav` | Default voice reference WAV |
| `F5_REF_TEXT` | bundled transcript / `""` | Transcript of ref audio (auto-detected via Whisper if empty + custom ref) |
| `F5_CHECKPOINT` | unset | Override checkpoint weights; passed as `ckpt_file` to F5TTS |
| `F5_VOCAB` | `""` | Tokenizer vocab; passed as `vocab_file` (a RU fine-tune extends it) |
| `F5_MODEL` | `F5TTS_v1_Base` | Architecture name; the community RU model is v0 `F5TTS_Base` |
| `F5_RU_STRESS` | unset (off) | Apply RUAccent `+` stress marks to RU text. Only for a checkpoint **trained** on stress-marked data — it corrupts a plain-text-trained one |
| `RUACCENT_WORKDIR` | `$HF_HOME/ruaccent` | Where RUAccent downloads its models (writable, persistent) |

## Docker Conventions (adapted from the sibling `voicebox` project)

- **Python 3.11** (`python:3.11-slim`) — F5-TTS deps are unhappy on 3.12+.
- **Multi-stage build**: a `builder` stage compiles deps; the runtime image
  copies `/install` and carries no compilers.
- **Non-root user** `eloquium`; `/cache` is chowned so the named volume inherits
  writable ownership.
- **HEALTHCHECK** hits `/health` (300s start period for first-run model download).
- **Port bound to `127.0.0.1`** only (personal-use phase). Open it up for SaaS.
- **CPU-only torch by default.** The Dockerfile installs torch/torchaudio from
  `https://download.pytorch.org/whl/cpu` (the default PyPI wheel bundles ~1.5GB
  of CUDA libs). For a GPU build:
  `docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124`
  (and enable the GPU block in docker-compose.yml).
- **The Makefile auto-detects the GPU** (via `nvidia-smi`) and selects both the
  right `TORCH_INDEX` (CUDA vs CPU) **and** the docker-compose GPU override.
  - `make run` — from-scratch entry point: build + start + wait until healthy.
  - `make gen TEXT="..." LNG=ru` — synthesize; prints the path to a `.wav` in `out/`.
  - `make gpu-setup` — install the NVIDIA Container Toolkit on the host (sudo, one-time).
  - `make gpu-check` — verify Docker can see the GPU.
  - `make up` / `logs` / `down` / `health` / `restart` / `ps` as before.
  Plain `docker compose` still works and defaults to CPU torch.
- **GPU is opt-in via a compose override.** The base `docker-compose.yml` is
  CPU-safe (no device reservation, so `docker compose up` works on any machine).
  The `reservations` GPU block lives in `docker-compose.gpu.yml`, which the
  Makefile layers on (`-f docker-compose.yml -f docker-compose.gpu.yml`) only
  when a GPU is detected. Run GPU manually with both `-f` flags.

## Swapping the Trained Model

When cloud training completes:
1. Place the checkpoint in the `model-cache` volume (or mount it into the container).
2. Set `F5_CHECKPOINT=/cache/your_model.pt` in docker-compose `environment`. Also
   set `F5_VOCAB` if the model ships its own vocab and `F5_MODEL` if it isn't the
   v1 architecture. If it was trained on stress-marked text, set `F5_RU_STRESS=1`.
3. `docker compose up -d` — restart picks it up. **No code change required.**

### Current model: pre-trained Russian (interim)

Until our own training completes, the service runs the community
`hotstone228/F5-TTS-Russian` checkpoint (RU+EN, v0 `F5TTS_Base` arch, plain text,
**CC BY-NC-SA 4.0 — non-commercial**, so personal-use only). Files live at
`/cache/ru/` in the volume; docker-compose points `F5_CHECKPOINT`/`F5_VOCAB`/
`F5_MODEL` at it. Russian intelligibility verified via ASR round-trip
(`"Привет, мир."` → audio → `"Привет, мир!"`). `F5_RU_STRESS` is **off** for it.

**Default voice is a native Russian reference** (`assets/ref_ru.wav`, baked into
the image; `F5_REF_AUDIO`/`F5_REF_TEXT` point at it). This matters a lot: F5-TTS
imitates the reference voice, so the old English default (`basic_ref_en.wav`)
injected a foreign accent and even corrupted words in Russian output. Switching
to a native RU reference (clip from `google/fleurs` ru_ru, CC BY 4.0) fixed both,
verified via before/after ASR. Replace the file + `F5_REF_TEXT` to change voices.

## TODO (not done yet)

- [x] Russian preprocessing — hook wired and `ruaccent` installed.
  `TTSEngine.synthesize` calls `preprocessing.normalize_ru` for `language ==
  "ru"` when `F5_RU_STRESS=1`. It is **off by default**: the current RU
  checkpoint is plain-text-trained and the `+` marks corrupt it (verified). The
  same `normalize_ru` is used by `training/prepare_dataset.py`, so a future
  stress-trained checkpoint (set `F5_RU_STRESS=1`) gets identical normalization
  at train and inference time — no drift.
- [~] Training scripts / dataset pipeline — scaffolded in `training/` (runs on
  the cloud GPU box). F5-TTS CLI specifics marked `TODO(gpu)`; no run yet.
- [ ] Voice cloning endpoint (future phase)
