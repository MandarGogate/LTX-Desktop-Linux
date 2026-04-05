from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from services.text_encoder.gguf_text_encoder_builder import _materialize_meta_module_on_cpu
from services.text_encoder.text_encoder_variant_utils import (
    variant_uses_wrapped_gemma_text_encoder_keys,
)

logger = logging.getLogger(__name__)


def _resolve_variant_checkpoint_sources(checkpoint_sources: Any) -> list[str]:
    if isinstance(checkpoint_sources, (list, tuple)):
        return [str(path) for path in checkpoint_sources]
    return [str(checkpoint_sources)]


def _normalize_text_encoder_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    needs_unwrap = any(
        key.startswith("model.model.language_model.") or key.startswith("model.model.lm_head.")
        for key in state_dict.keys()
    )
    if not needs_unwrap:
        return state_dict

    logger.info("Text encoder variant state dict already wrapped twice; stripping one leading 'model.' prefix")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.model."):
            normalized[key[len("model."):]] = value
        else:
            normalized[key] = value
    return normalized


@dataclass(frozen=True)
class SafetensorsGemmaTextEncoderBuilder:
    """Build a Gemma text encoder from a safetensors file/folder variant.

    This builder preserves existing folder loading behavior while tolerating
    text-encoder variants that already contain wrapped keys such as
    ``model.language_model.*``. In those cases the upstream sd_ops can produce
    ``model.model.language_model.*``; we normalize that before loading.
    """

    base_builder: Any
    checkpoint_sources: Any
    module_ops: tuple[Any, ...]
    variant_path: str

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.nn.Module:
        target_device = torch.device("cuda") if device is None else device
        target_dtype = torch.bfloat16 if dtype is None else dtype

        config = self.base_builder.model_config()
        text_encoder = self.base_builder.meta_model(config, self.module_ops)
        text_encoder = _materialize_meta_module_on_cpu(text_encoder)

        checkpoint_paths = _resolve_variant_checkpoint_sources(self.checkpoint_sources)
        checkpoint_sd = self.base_builder.model_loader.load(
            checkpoint_paths,
            sd_ops=self.base_builder.model_sd_ops,
            device=torch.device("cpu"),
        )
        checkpoint_state = {
            key: value.to(dtype=target_dtype) for key, value in checkpoint_sd.sd.items()
        }
        checkpoint_state = _normalize_text_encoder_state_dict_keys(checkpoint_state)

        text_encoder.load_state_dict(checkpoint_state, strict=False, assign=True)
        return text_encoder.to(target_device).eval()

    @property
    def uses_wrapped_variant_keys(self) -> bool:
        return variant_uses_wrapped_gemma_text_encoder_keys(self.variant_path)
