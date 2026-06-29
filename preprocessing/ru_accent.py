"""Russian stress normalization via RUAccent.

Why this exists
---------------
The base F5-TTS checkpoint cannot pronounce Russian (verified: an ASR
round-trip on base-model RU output returns gibberish). Stress placement is
also lexically contrastive in Russian (за́мок "castle" vs замо́к "lock"), so a
Russian-capable checkpoint still needs explicit stress to sound right.

RUAccent annotates each word with its stressed vowel. Feeding stress-marked
text *both* during fine-tuning and at inference is what lets the trained model
place stress correctly. On the base English model this is effectively a no-op
for intelligibility — it only pays off once the Russian checkpoint is swapped
in (see CLAUDE.md "Swapping the Trained Model").

This module lazy-loads the model so importing it is cheap and the heavy
RUAccent weights are only pulled when Russian text is actually processed.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# RUAccent marks the stressed vowel with a trailing '+': "приве+т". F5-TTS keeps
# arbitrary characters in its vocab, so the mark survives tokenization.
_accentizer = None
_load_failed = False


def _get_accentizer():
    """Lazily construct the RUAccent singleton. Returns None if unavailable."""
    global _accentizer, _load_failed
    if _accentizer is not None:
        return _accentizer
    if _load_failed:
        return None
    try:
        from ruaccent import RUAccent

        accentizer = RUAccent()
        # RUAccent downloads its models into `workdir`; default it to a writable,
        # persistent dir in the cache volume (the container runs non-root, so the
        # package's own site-packages/.cache is read-only). Falls back to /tmp.
        workdir = os.environ.get("RUACCENT_WORKDIR") or os.path.join(
            os.environ.get("HF_HOME", "/tmp"), "ruaccent"
        )
        os.makedirs(workdir, exist_ok=True)
        # 'turbo' is the small/fast omograph model — good default for CPU
        # inference. Override with RUACCENT_MODEL_SIZE on the training box.
        accentizer.load(
            omograph_model_size=os.environ.get("RUACCENT_MODEL_SIZE", "turbo"),
            use_dictionary=True,
            workdir=workdir,
        )
        _accentizer = accentizer
        logger.info("RUAccent loaded (omograph=%s)", os.environ.get("RUACCENT_MODEL_SIZE", "turbo"))
        return _accentizer
    except Exception:  # pragma: no cover - optional dependency
        # Not installed (e.g. the CPU inference image hasn't added it yet).
        # Degrade gracefully: callers get the original text back.
        logger.warning(
            "RUAccent unavailable; Russian text passed through unmodified. "
            "Install `ruaccent` to enable stress normalization."
        )
        _load_failed = True
        return None


def normalize_ru(text: str) -> str:
    """Return `text` with Russian stress marks applied.

    Falls back to the original text unchanged if RUAccent is not installed, so
    this is always safe to call. Non-Russian text is left effectively untouched
    by RUAccent.
    """
    accentizer = _get_accentizer()
    if accentizer is None:
        return text
    return accentizer.process_all(text)
