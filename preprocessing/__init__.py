"""Text preprocessing shared by the training pipeline and inference service.

The Russian path here is the hook referenced in CLAUDE.md: RUAccent stress
normalization. It is intentionally usable from two places —

  * `training/prepare_dataset.py` — normalize transcripts before fine-tuning so
    the model learns stress-marked Cyrillic, and
  * `TTSEngine.synthesize` (root `model.py`) — normalize request text the same
    way at inference time.

Keeping it in one module means training and inference never drift.
"""

from .ru_accent import normalize_ru

__all__ = ["normalize_ru"]
