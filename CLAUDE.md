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
| `F5_REF_AUDIO` | bundled `basic_ref_en.wav` | Default voice reference WAV |
| `F5_REF_TEXT` | `""` | Transcript of ref audio (auto-detected if empty) |
| `F5_CHECKPOINT` | unset | Override checkpoint path; passed as `ckpt_file` to F5TTS |

## Swapping the Trained Model

When cloud training completes:
1. Place the checkpoint in the `model-cache` volume (or mount it into the container).
2. Set `F5_CHECKPOINT=/cache/your_model.pt` in docker-compose `environment`.
3. `docker compose up -d` — restart picks it up. **No code change required.**

## TODO (not done yet)

- [ ] Russian preprocessing (RUAccent + sova-tts-tps) in `TTSEngine.synthesize`
- [ ] Training scripts / dataset pipeline (`training/`)
- [ ] Voice cloning endpoint (future phase)
