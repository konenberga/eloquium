#!/usr/bin/env python3
"""Turn raw {audio, transcript} pairs into the F5-TTS metadata.csv format.

Input layout (``--raw-dir``): one transcript per clip, matched by stem —

    raw/
      clip0001.wav   clip0001.txt
      clip0002.wav   clip0002.txt

Output (``--out-dir``):

    out/
      wavs/clip0001.wav   (resampled to 24 kHz mono)
      metadata.csv        ("audio|text", one row per clip)

For ``--language ru`` the transcript is stress-normalized with RUAccent (shared
preprocessing/ module) so the model trains on stress-marked Cyrillic — the same
normalization applied at inference time.

After this, build the arrow dataset with F5-TTS's own tool:

    f5-tts_prepare_csv_wavs <out-dir> data/<name>_char
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo-root `preprocessing` package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_SR = 24000  # F5-TTS / vocos operate at 24 kHz


def _load_normalizer(language: str):
    if language == "ru":
        from preprocessing import normalize_ru

        return normalize_ru
    return lambda text: text  # en / passthrough


def _convert_audio(src: Path, dst: Path) -> float:
    """Resample to 24 kHz mono, write to dst, return duration in seconds."""
    import librosa
    import soundfile as sf

    wav, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
    sf.write(str(dst), wav, TARGET_SR)
    return len(wav) / TARGET_SR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--language", choices=["en", "ru"], default="ru")
    ap.add_argument("--min-sec", type=float, default=1.0, help="drop clips shorter than this")
    ap.add_argument("--max-sec", type=float, default=20.0, help="drop clips longer than this")
    args = ap.parse_args()

    normalize = _load_normalizer(args.language)
    wav_out = args.out_dir / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    kept = skipped = 0
    for txt in sorted(args.raw_dir.glob("*.txt")):
        wav = txt.with_suffix(".wav")
        if not wav.exists():
            print(f"skip {txt.name}: no matching .wav", file=sys.stderr)
            skipped += 1
            continue
        transcript = txt.read_text(encoding="utf-8").strip()
        if not transcript:
            print(f"skip {txt.name}: empty transcript", file=sys.stderr)
            skipped += 1
            continue

        dst = wav_out / wav.name
        dur = _convert_audio(wav, dst)
        if not (args.min_sec <= dur <= args.max_sec):
            print(f"skip {wav.name}: {dur:.1f}s out of [{args.min_sec},{args.max_sec}]", file=sys.stderr)
            dst.unlink(missing_ok=True)
            skipped += 1
            continue

        text = normalize(transcript)
        # F5-TTS prepare_csv_wavs expects pipe-separated "audio|text".
        rows.append(f"wavs/{wav.name}|{text}")
        kept += 1

    metadata = args.out_dir / "metadata.csv"
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nWrote {metadata} — kept {kept}, skipped {skipped}")
    print(f"Next: f5-tts_prepare_csv_wavs {args.out_dir} data/<name>_char")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
