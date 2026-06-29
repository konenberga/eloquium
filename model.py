"""F5-TTS inference wrapper.

Model weights are downloaded to HF_HOME (the /cache volume) on first use — never
baked into the Docker image. See CLAUDE.md for the design rules.
"""
import io
import logging
import os

import soundfile as sf
import torch

from preprocessing import normalize_ru

logger = logging.getLogger(__name__)

# Known transcript of the bundled basic_ref_en.wav (from f5_tts basic.toml).
# Supplying it avoids F5-TTS auto-transcribing the reference with Whisper
# (a ~1.6GB download + slow CPU pass) on the first request.
_BUNDLED_REF_TEXT = "Some call me nature, others call me mother nature."


def _resolve_ref_audio() -> tuple[str, bool]:
    """Resolve the default reference audio (defines the default voice).

    Priority: F5_REF_AUDIO env var → bundled example from the f5-tts package.
    This is the voice reference, NOT the model weights.

    Returns (path, is_bundled_default); the flag lets the caller supply the
    known transcript only when using our own bundled reference.
    """
    env_path = os.environ.get("F5_REF_AUDIO")
    if env_path:
        return env_path, False
    try:
        # f5_tts is a namespace package (__file__ is None), so resolve the
        # bundled example via importlib.resources rather than __file__.
        from importlib.resources import files

        candidate = (
            files("f5_tts")
            / "infer" / "examples" / "basic" / "basic_ref_en.wav"
        )
        if candidate.is_file():
            return str(candidate), True
    except Exception:  # pragma: no cover - import/layout fallback
        pass
    raise RuntimeError(
        "Default reference audio not found. "
        "Set F5_REF_AUDIO to point at a WAV file."
    )


class TTSEngine:
    """Wraps F5-TTS. Constructed once at app startup, reused per request."""

    def __init__(self) -> None:
        from f5_tts.api import F5TTS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # A custom checkpoint swaps in with zero code change. Three knobs,
        # because a fine-tune may differ from the base on more than the weights:
        #   F5_CHECKPOINT — weights (.safetensors/.pt)
        #   F5_VOCAB      — tokenizer vocab (a RU fine-tune extends it)
        #   F5_MODEL      — architecture name; e.g. the community RU model is
        #                   "F5TTS_Base" (v0), not the default "F5TTS_v1_Base".
        # ckpt_file/vocab_file default to "" (F5TTS's own "use base" sentinel).
        ckpt = os.environ.get("F5_CHECKPOINT") or ""
        vocab = os.environ.get("F5_VOCAB") or ""
        model_arch = os.environ.get("F5_MODEL") or "F5TTS_v1_Base"
        if ckpt:
            logger.info(
                "Loading checkpoint=%s vocab=%s arch=%s",
                ckpt, vocab or "(default)", model_arch,
            )
        else:
            logger.info("Loading base F5-TTS (downloads to HF_HOME if not cached)...")

        self._model = F5TTS(
            model=model_arch,
            ckpt_file=ckpt,
            vocab_file=vocab,
            device=self.device,
        )
        self._ref_audio, is_bundled = _resolve_ref_audio()

        # An explicit F5_REF_TEXT always wins. Otherwise, use the known
        # transcript for the bundled reference; for a custom F5_REF_AUDIO with
        # no text, leave it empty so F5-TTS auto-transcribes (Whisper).
        ref_text = os.environ.get("F5_REF_TEXT")
        if ref_text is None:
            ref_text = _BUNDLED_REF_TEXT if is_bundled else ""
        self._ref_text = ref_text

        # Whether to apply RUAccent stress marks to Russian text is a property of
        # the loaded checkpoint, not a universal good: a model trained on plain
        # text (e.g. hotstone228/F5-TTS-Russian) is *corrupted* by '+' marks
        # (verified via ASR round-trip), while a model trained on stress-marked
        # data needs them. Off by default; enable F5_RU_STRESS=1 for a
        # stress-trained checkpoint (e.g. one trained via our training/ scaffold).
        self._ru_stress = os.environ.get("F5_RU_STRESS", "").lower() in ("1", "true", "yes")

        logger.info(
            "Ready  device=%s  ref_audio=%s  ru_stress=%s",
            self.device, self._ref_audio, self._ru_stress,
        )

    def synthesize(self, text: str, language: str = "en") -> bytes:
        """Generate speech for `text`, returning WAV bytes.

        For `language == "ru"` AND when the loaded checkpoint was trained on
        stress-marked text (F5_RU_STRESS=1), the text is stress-normalized with
        RUAccent (shared preprocessing/ module) — the same normalization applied
        to training transcripts, so the two never drift. It is off by default
        because a plain-text-trained checkpoint is degraded by the marks.
        """
        if language == "ru" and self._ru_stress:
            text = normalize_ru(text)

        wav, sr, _ = self._model.infer(
            ref_file=self._ref_audio,
            ref_text=self._ref_text,
            gen_text=text,
            remove_silence=True,
        )
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        return buf.read()
