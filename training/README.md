# Eloquium — Training (Phase 2)

Fine-tune F5-TTS so it can actually speak **Russian** (and stay strong on
English). The base `F5TTS_v1_Base` checkpoint is English/Chinese only — verified
by an ASR round-trip where base-model Russian output transcribes back to
gibberish. This directory produces the checkpoint that replaces it.

> Runs on a **cloud GPU** (RunPod / Vast.ai / Colab Pro). The local GTX 1050 is
> inference-only. Nothing here is meant to run on the inference host.

## Pipeline

```
raw data (wav + transcript)
        │  prepare_dataset.py   ← resample, validate, RUAccent stress-mark (ru)
        ▼
metadata.csv  (F5-TTS "audio|text" format)  +  wavs/
        │  f5-tts prepare_csv_wavs            ← builds arrow dataset + vocab + durations
        ▼
data/<name>_char/  (raw.arrow, duration.json, vocab.txt)
        │  finetune.py          ← wraps F5-TTS finetune CLI with configs/*.toml
        ▼
checkpoints/  (model_last.pt → export model_<name>.pt)
        │  eval_asr.py          ← objective WER gate (ASR round-trip)
        ▼
copy .pt into the model-cache volume, set F5_CHECKPOINT, restart  (see root CLAUDE.md)
```

## Setup (on the GPU box)

```bash
pip install -r training/requirements-train.txt
# torch is installed from the CUDA index here — opposite of the inference image,
# which is CPU-only.
```

## Steps

1. **Collect data.** Russian + English speech with clean transcripts. Target
   short, single-speaker clips (3–15 s) at first. Licensing matters for the
   eventual SaaS path — keep provenance for every source.

2. **Prepare the dataset.**
   ```bash
   python training/prepare_dataset.py \
       --raw-dir /data/ru_raw \
       --out-dir /data/eloquium_ru \
       --language ru          # applies RUAccent stress marks to transcripts
   ```
   Produces `/data/eloquium_ru/metadata.csv` + `wavs/`. Then build the arrow
   dataset with F5-TTS's own tool:
   ```bash
   f5-tts_prepare_csv_wavs /data/eloquium_ru data/eloquium_ru_char
   ```

3. **Fine-tune.**
   ```bash
   python training/finetune.py --config training/configs/ru_finetune.toml
   ```

4. **Evaluate** (objective intelligibility gate):
   ```bash
   python training/eval_asr.py --checkpoint checkpoints/model_eloquium_ru.pt \
       --lines training/eval_lines_ru.txt --language ru
   ```
   Reports WER of `ASR(synth(text))` vs `text`. Base model RU WER is ~100%
   (gibberish); a usable checkpoint should drop this substantially.

5. **Swap in.** Copy the `.pt` into the `model-cache` volume and set
   `F5_CHECKPOINT=/cache/model_eloquium_ru.pt`. No inference code change — see
   the root `CLAUDE.md`.

## Status

Scaffold only. The scripts encode the workflow and wire up RUAccent, but the
F5-TTS CLI invocations and dataset specifics must be confirmed against the
installed F5-TTS version on the GPU box (marked with `TODO(gpu)` in the code).
