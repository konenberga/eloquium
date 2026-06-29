#!/usr/bin/env python3
"""Launch an F5-TTS fine-tune from a TOML config.

Thin wrapper around F5-TTS's own training CLI so our hyperparameters live in
version control (configs/*.toml) instead of a shell history. Run on the GPU box
after `f5-tts_prepare_csv_wavs` has built data/<name>_char/.

    python training/finetune.py --config training/configs/ru_finetune.toml

TODO(gpu): the exact CLI name/flags depend on the installed F5-TTS version.
Recent versions expose `f5-tts_finetune-cli`. Confirm with `f5-tts_finetune-cli
--help` on the box and reconcile the flag names below before the first real run.
This wrapper intentionally prints the command and (unless --dry-run) execs it,
so a flag mismatch is visible rather than silently wrong.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def build_command(cfg: dict) -> list[str]:
    ds, base, tr, out = cfg["dataset"], cfg["base"], cfg["train"], cfg["output"]
    cmd = [
        "f5-tts_finetune-cli",
        "--dataset_name", ds["name"],
        "--tokenizer", ds["tokenizer"],
        "--pretrain", base["pretrain_ckpt"] or base["model"],
        "--epochs", str(tr["epochs"]),
        "--batch_size_per_gpu", str(tr["batch_size_per_gpu"]),
        "--batch_size_type", tr["batch_size_type"],
        "--max_samples", str(tr["max_samples"]),
        "--learning_rate", str(tr["learning_rate"]),
        "--num_warmup_updates", str(tr["warmup_updates"]),
        "--grad_accumulation_steps", str(tr["grad_accumulation"]),
        "--max_grad_norm", str(tr["max_grad_norm"]),
        "--save_per_updates", str(tr["save_per_updates"]),
        "--last_per_updates", str(tr["last_per_updates"]),
        "--num_workers", str(tr["num_workers"]),
    ]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    with args.config.open("rb") as fh:
        cfg = tomllib.load(fh)

    cmd = build_command(cfg)
    printable = " \\\n    ".join(cmd)
    print(f"# fine-tune command:\n{printable}\n")

    if args.dry_run:
        return 0
    if shutil.which(cmd[0]) is None:
        print(
            f"error: '{cmd[0]}' not found. Install training/requirements-train.txt "
            "on the GPU box (this is not meant to run on the inference host).",
            file=sys.stderr,
        )
        return 127
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
