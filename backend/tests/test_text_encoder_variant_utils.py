from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from services.text_encoder.text_encoder_variant_utils import (
    variant_uses_wrapped_gemma_text_encoder_keys,
)


def test_variant_uses_wrapped_gemma_text_encoder_keys_for_prewrapped_folder(tmp_path: Path) -> None:
    variant_dir = tmp_path / "wrapped"
    variant_dir.mkdir()
    save_file(
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 1)),
            "model.language_model.norm.weight": torch.zeros((1,)),
        },
        str(variant_dir / "model-00001-of-00001.safetensors"),
    )

    assert variant_uses_wrapped_gemma_text_encoder_keys(str(variant_dir)) is True


def test_variant_uses_wrapped_gemma_text_encoder_keys_for_native_gemma_folder(tmp_path: Path) -> None:
    variant_dir = tmp_path / "native"
    variant_dir.mkdir()
    save_file(
        {
            "language_model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 1)),
            "language_model.norm.weight": torch.zeros((1,)),
        },
        str(variant_dir / "model-00001-of-00001.safetensors"),
    )

    assert variant_uses_wrapped_gemma_text_encoder_keys(str(variant_dir)) is False


def test_variant_uses_wrapped_gemma_text_encoder_keys_for_gguf_path_is_false(tmp_path: Path) -> None:
    gguf_path = tmp_path / "gemma.gguf"
    gguf_path.write_bytes(b"GGUF")

    assert variant_uses_wrapped_gemma_text_encoder_keys(str(gguf_path)) is False
