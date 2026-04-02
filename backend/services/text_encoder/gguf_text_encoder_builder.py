from __future__ import annotations

from types import MethodType, SimpleNamespace
import logging
from dataclasses import dataclass
from typing import Any

import torch

from services.gguf_loader.gguf_lazy_loader import (
    assign_gguf_linear_weights,
    load_gguf_lazy_state_dict,
    replace_embedding_with_gguf,
    replace_linear_with_gguf,
)

logger = logging.getLogger(__name__)


_GEMMA3_GGUF_REMAP: tuple[tuple[str, str], ...] = (
    ("blk.", "model.language_model.layers."),
    ("attn_q_norm.", "self_attn.q_norm."),
    ("attn_k_norm.", "self_attn.k_norm."),
    ("post_ffw_norm", "post_feedforward_layernorm"),
    ("post_attention_norm", "post_attention_layernorm"),
    ("attn_norm", "input_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_norm", "pre_feedforward_layernorm"),
    ("token_embd", "model.language_model.embed_tokens"),
    ("output_norm", "model.language_model.norm"),
    ("output.weight", "lm_head.weight"),
)


def remap_gemma3_gguf_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for original_key, tensor in state_dict.items():
        key = original_key
        for old, new in _GEMMA3_GGUF_REMAP:
            key = key.replace(old, new)
        remapped[key] = tensor
    return remapped


def _materialize_meta_module_on_cpu(module: torch.nn.Module) -> torch.nn.Module:
    has_meta = any(
        str(p.device) == "meta"
        for p in list(module.parameters()) + list(module.buffers())
    )
    if not has_meta:
        return module

    module = module.to_empty(device=torch.device("cpu"))
    for param in module.parameters():
        if str(param.device) == "meta":
            param.data = torch.zeros_like(param, device=torch.device("cpu"))
    for buf in module.buffers():
        if str(buf.device) == "meta":
            buf.data = torch.zeros_like(buf, device=torch.device("cpu"))
    return module


def load_gemma_text_model_from_gguf(
    gemma_model: torch.nn.Module,
    gguf_path: str,
    *,
    dtype: torch.dtype,
) -> torch.nn.Module:
    state_dict = load_gguf_lazy_state_dict(gguf_path, device="cpu")
    state_dict = remap_gemma3_gguf_keys(state_dict)

    replace_embedding_with_gguf(gemma_model, state_dict, compute_dtype=dtype)
    replace_linear_with_gguf(gemma_model, state_dict, compute_dtype=dtype)
    gemma_model, remaining_state_dict = assign_gguf_linear_weights(gemma_model, state_dict)
    gemma_model.load_state_dict(remaining_state_dict, strict=False, assign=True)
    return gemma_model


def patch_gemma_forward_hidden_states_only(text_encoder: torch.nn.Module) -> None:
    model = getattr(text_encoder, "model", None)
    if model is None or not hasattr(model, "model"):
        return
    if getattr(model, "_ltx_hidden_states_only_patched", False):
        return

    def _forward_hidden_states_only(self_model: object, *args: object, **kwargs: object) -> object:
        kwargs["return_dict"] = True
        outputs = getattr(self_model, "model")(*args, **kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        return SimpleNamespace(hidden_states=hidden_states)

    model.forward = MethodType(_forward_hidden_states_only, model)  # type: ignore[assignment]
    setattr(model, "_ltx_hidden_states_only_patched", True)


@dataclass(frozen=True)
class GGUFGemmaTextEncoderBuilder:
    """Build a Gemma text encoder with GGUF-backed language-model weights."""

    base_builder: Any
    checkpoint_path: Any
    gguf_path: str

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.nn.Module:
        target_device = torch.device("cuda") if device is None else device
        target_dtype = torch.bfloat16 if dtype is None else dtype

        config = self.base_builder.model_config()
        text_encoder = self.base_builder.meta_model(config, self.base_builder.module_ops)
        text_encoder = _materialize_meta_module_on_cpu(text_encoder)

        checkpoint_sources = getattr(self.base_builder, "model_path", self.checkpoint_path)
        if isinstance(checkpoint_sources, (list, tuple)):
            checkpoint_paths = [str(path) for path in checkpoint_sources]
        else:
            checkpoint_paths = [str(checkpoint_sources)]

        checkpoint_sd = self.base_builder.model_loader.load(
            checkpoint_paths,
            sd_ops=self.base_builder.model_sd_ops,
            device=torch.device("cpu"),
        )
        checkpoint_state = checkpoint_sd.sd
        if target_dtype is not None:
            checkpoint_state = {
                key: value.to(dtype=target_dtype) for key, value in checkpoint_state.items()
            }
        text_encoder.load_state_dict(checkpoint_state, strict=False, assign=True)

        if getattr(text_encoder, "model", None) is None:
            raise ValueError("Gemma text encoder missing language model component")

        text_encoder.model = load_gemma_text_model_from_gguf(
            text_encoder.model,
            self.gguf_path,
            dtype=target_dtype,
        )
        patch_gemma_forward_hidden_states_only(text_encoder)
        text_encoder._ltx_gguf_text_encoder = True  # type: ignore[attr-defined]
        logger.info("Using GGUF text encoder variant: %s", self.gguf_path)

        return text_encoder.to(target_device).eval()
