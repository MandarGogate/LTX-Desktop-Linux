from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _variant_weight_files(variant_path: str) -> list[Path]:
    path = Path(variant_path)
    if path.is_dir():
        shard_paths = sorted(path.glob("model-*.safetensors"))
        if shard_paths:
            return shard_paths
        return sorted(path.glob("*.safetensors"))
    return [path]


def variant_uses_wrapped_gemma_text_encoder_keys(variant_path: str) -> bool:
    """Whether a safetensors variant already uses wrapped LTX text-encoder keys.

    Native Gemma checkpoints are expected to use keys like
    ``language_model.layers.*`` / ``lm_head.*`` and therefore still need
    ``GEMMA_MODEL_OPS`` applied by the builder. Some exported text-encoder
    variants already contain wrapped keys like ``model.language_model.*``.
    Applying ``GEMMA_MODEL_OPS`` again to those weights produces an invalid
    double prefix such as ``model.model.language_model.*``.
    """
    files = _variant_weight_files(variant_path)
    if not files:
        return False

    try:
        from safetensors import safe_open

        with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                return key.startswith("model.language_model.") or key.startswith("model.lm_head.")
    except Exception:
        logger.debug(
            "Could not inspect text encoder variant keys for %s",
            variant_path,
            exc_info=True,
        )
    return False
