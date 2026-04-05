from __future__ import annotations

import torch

from services.text_encoder.safetensors_text_encoder_builder import (
    _normalize_text_encoder_state_dict_keys,
)


def test_normalize_text_encoder_state_dict_keys_strips_double_model_prefix() -> None:
    state = {
        "model.model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 1)),
        "model.model.language_model.norm.weight": torch.zeros((1,)),
        "other.weight": torch.zeros((1,)),
    }

    normalized = _normalize_text_encoder_state_dict_keys(state)

    assert "model.language_model.layers.0.self_attn.q_proj.weight" in normalized
    assert "model.language_model.norm.weight" in normalized
    assert "model.model.language_model.layers.0.self_attn.q_proj.weight" not in normalized
    assert "other.weight" in normalized


def test_normalize_text_encoder_state_dict_keys_keeps_native_keys() -> None:
    state = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros((1, 1)),
    }

    normalized = _normalize_text_encoder_state_dict_keys(state)

    assert normalized == state
