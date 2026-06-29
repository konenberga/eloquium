#!/usr/bin/env python3
"""Objective intelligibility gate: ASR round-trip WER.

Synthesize each eval line with a given checkpoint, transcribe the audio back
with Whisper, and report the word error rate of ASR(synth(text)) vs text. This
is the same round-trip used to prove the base model can't speak Russian (base
RU output transcribes to gibberish ~= 100% WER); a usable Russian checkpoint
should drop it sharply.

    python training/eval_asr.py \
        --checkpoint checkpoints/model_eloquium_ru.pt \
        --lines training/eval_lines_ru.txt \
        --language ru

`--lines` is one sentence per line, plain (un-stressed) text — stress marks are
applied internally for synthesis but compared against the plain reference, since
Whisper does not emit them.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_WHISPER = "openai/whisper-large-v3-turbo"


def _normalize_for_wer(s: str) -> str:
    # Lowercase, drop stress marks and punctuation so WER measures words, not
    # formatting. Keep it simple and language-agnostic.
    out = []
    for ch in s.lower():
        if ch in "+":  # RUAccent stress mark
            continue
        out.append(" " if not (ch.isalnum() or ch.isspace()) else ch)
    return " ".join("".join(out).split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="F5-TTS .pt to evaluate; omit to test the base model")
    ap.add_argument("--lines", required=True, type=Path)
    ap.add_argument("--language", choices=["en", "ru"], default="ru")
    ap.add_argument("--ref-audio", default=None, help="reference voice WAV")
    args = ap.parse_args()

    import jiwer
    import soundfile as sf
    import torch
    from f5_tts.api import F5TTS
    from transformers import pipeline

    if args.language == "ru":
        from preprocessing import normalize_ru
    else:
        normalize_ru = lambda t: t  # noqa: E731

    lines = [ln.strip() for ln in args.lines.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        print("no eval lines", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = str(args.checkpoint) if args.checkpoint else None
    model = F5TTS(ckpt_file=ckpt, device=device) if ckpt else F5TTS(device=device)
    asr = pipeline("automatic-speech-recognition", model=_WHISPER, device=device)

    refs, hyps = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, line in enumerate(lines):
            wav, sr, _ = model.infer(
                ref_file=args.ref_audio or model.ref_audio if hasattr(model, "ref_audio") else args.ref_audio,
                ref_text="",
                gen_text=normalize_ru(line),
                remove_silence=True,
            )
            path = Path(tmp) / f"line_{i}.wav"
            sf.write(str(path), wav, sr)
            text = asr(str(path), generate_kwargs={"language": args.language})["text"]
            refs.append(_normalize_for_wer(line))
            hyps.append(_normalize_for_wer(text))
            print(f"[{i}] ref: {refs[-1]!r}\n    asr: {hyps[-1]!r}")

    wer = jiwer.wer(refs, hyps)
    print(f"\nWER = {wer:.3f}  ({len(lines)} lines, lang={args.language}, "
          f"ckpt={'base' if not ckpt else args.checkpoint.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
