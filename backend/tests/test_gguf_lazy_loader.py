from __future__ import annotations

import inspect

import pytest
import torch

from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline
from services.gguf_loader.gguf_lazy_loader import (
    GGUFEmbedding,
    GGUFLinear,
    GGUFParameter,
    assign_gguf_linear_weights,
)
from services.text_encoder.gguf_text_encoder_builder import (
    GGUFGemmaTextEncoderBuilder,
    remap_gemma3_gguf_keys,
)


def test_gguf_parameter_preserves_tensor_shape_on_to() -> None:
    raw = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    param = GGUFParameter(
        raw,
        requires_grad=False,
        quant_type=1,
        tensor_shape=(8, 8),
    )

    moved = param.to("cpu")

    assert isinstance(moved, GGUFParameter)
    assert moved.tensor_shape == (8, 8)


def test_assign_gguf_linear_weights_installs_quantized_param_directly() -> None:
    model = torch.nn.Sequential(GGUFLinear(8, 8, bias=False))
    weight = GGUFParameter(
        torch.arange(32, dtype=torch.uint8).reshape(2, 16),
        requires_grad=False,
        quant_type=1,
        tensor_shape=(8, 8),
    )

    _, remaining = assign_gguf_linear_weights(model, {"0.weight": weight})

    assert model[0].weight is weight
    assert remaining == {}


def test_assign_gguf_linear_weights_handles_nested_modules() -> None:
    model = torch.nn.Sequential(
        torch.nn.Sequential(
            GGUFLinear(8, 8, bias=False),
        )
    )
    weight = GGUFParameter(
        torch.arange(32, dtype=torch.uint8).reshape(2, 16),
        requires_grad=False,
        quant_type=1,
        tensor_shape=(8, 8),
    )

    _, remaining = assign_gguf_linear_weights(model, {"0.0.weight": weight})

    assert model[0][0].weight is weight
    assert remaining == {}


def test_assign_gguf_module_weights_supports_embeddings() -> None:
    model = torch.nn.Sequential(GGUFEmbedding(8, 8))
    weight = GGUFParameter(
        torch.arange(32, dtype=torch.uint8).reshape(2, 16),
        requires_grad=False,
        quant_type=1,
        tensor_shape=(8, 8),
    )

    _, remaining = assign_gguf_linear_weights(model, {"0.weight": weight})

    assert model[0].weight is weight
    assert remaining == {}


def test_remap_gemma3_gguf_keys_maps_llama_style_names() -> None:
    sd = {
        "token_embd.weight": torch.ones(1),
        "blk.0.attn_q.weight": torch.ones(1),
        "blk.0.attn_k_norm.weight": torch.ones(1),
        "blk.0.ffn_gate.weight": torch.ones(1),
        "output_norm.weight": torch.ones(1),
        "output.weight": torch.ones(1),
    }

    remapped = remap_gemma3_gguf_keys(sd)

    assert "model.language_model.embed_tokens.weight" in remapped
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in remapped
    assert "model.language_model.layers.0.self_attn.k_norm.weight" in remapped
    assert "model.language_model.layers.0.mlp.gate_proj.weight" in remapped
    assert "model.language_model.norm.weight" in remapped
    assert "lm_head.weight" in remapped


def test_low_vram_pipeline_uses_lazy_gguf_loader() -> None:
    source = inspect.getsource(LTXLowVRAMPipeline._load_gguf_transformer)

    assert "load_gguf_lazy_state_dict" in source
    assert "load_gguf_fast" not in source


def test_gguf_text_encoder_builder_uses_base_builder_model_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Loader:
        def load(self, paths, *, sd_ops, device):
            del sd_ops, device
            captured["paths"] = list(paths)
            class _State:
                sd = {}
            return _State()

    class _Builder:
        model_loader = _Loader()
        model_sd_ops = object()
        module_ops = ()
        model_path = ("projection.safetensors", "connector.safetensors")

        @staticmethod
        def model_config():
            return object()

        @staticmethod
        def meta_model(config, module_ops):
            del config, module_ops
            class _TextEncoder(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.model = torch.nn.Linear(1, 1)
            return _TextEncoder()

    monkeypatch.setattr(
        "services.text_encoder.gguf_text_encoder_builder.load_gemma_text_model_from_gguf",
        lambda gemma_model, gguf_path, *, dtype: gemma_model,
    )
    monkeypatch.setattr(
        "services.text_encoder.gguf_text_encoder_builder.patch_gemma_forward_hidden_states_only",
        lambda text_encoder: None,
    )

    builder = GGUFGemmaTextEncoderBuilder(
        base_builder=_Builder(),
        checkpoint_path="fallback.safetensors",
        gguf_path="gemma.gguf",
    )
    text_encoder = builder.build(device=torch.device("cpu"), dtype=torch.bfloat16)

    assert isinstance(text_encoder, torch.nn.Module)
    assert captured["paths"] == ["projection.safetensors", "connector.safetensors"]
