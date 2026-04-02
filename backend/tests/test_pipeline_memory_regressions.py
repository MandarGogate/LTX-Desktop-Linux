from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from services.fast_video_pipeline.ltx_low_vram_pipeline import (
    LTXLowVRAMPipeline,
    _build_split_video_vae_sd_ops,
)
from services.fast_video_pipeline.ltx_optimized_pipeline import LTXOptimizedPipeline
from services.vram_manager.vram_manager import OffloadStrategy, VRAMTier


class _SimpleBlock(nn.Module):
    def __init__(self, size: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _LanguageModel(nn.Module):
    def __init__(self, layer_count: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_SimpleBlock() for _ in range(layer_count))
        self.embed_tokens = nn.Embedding(16, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _GemmaInnerModel(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.other = nn.Linear(8, 8)


class _GemmaModel(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.model = _GemmaInnerModel(language_model)
        self.lm_head = nn.Linear(8, 8)


class _TextEncoder(nn.Module):
    def __init__(self, layer_count: int = 8) -> None:
        super().__init__()
        self.model = _GemmaModel(_LanguageModel(layer_count))
        self.proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _TransformerCore(nn.Module):
    def __init__(self, block_count: int = 6) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_SimpleBlock() for _ in range(block_count))
        self.head = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TransformerWrapper(nn.Module):
    def __init__(self, block_count: int = 6) -> None:
        super().__init__()
        self.velocity_model = _TransformerCore(block_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.velocity_model(x)


@dataclass
class _DummyVRAMManager:
    tier: VRAMTier = VRAMTier.HIGH
    offload_strategy: OffloadStrategy = OffloadStrategy.BLOCK_SWAP
    block_swap_keep_on_gpu: int = 2
    free_vram_mb: int = 24_000

    def get_free_vram_mb(self) -> int:
        return self.free_vram_mb


def _hook_counts(module: nn.Module) -> tuple[int, int]:
    return (len(module._forward_pre_hooks), len(module._forward_hooks))


class TestLowVRAMPipelineRegressions:
    def test_split_video_vae_decoder_sd_ops_accepts_split_layout(self) -> None:
        sd_ops = _build_split_video_vae_sd_ops("decoder")

        assert sd_ops.apply_to_key("decoder.conv_in.conv.weight") == "conv_in.conv.weight"
        assert sd_ops.apply_to_key("vae.decoder.conv_in.conv.weight") == "conv_in.conv.weight"
        assert (
            sd_ops.apply_to_key("per_channel_statistics.std-of-means")
            == "per_channel_statistics.std-of-means"
        )

    def test_split_video_vae_encoder_sd_ops_accepts_split_layout(self) -> None:
        sd_ops = _build_split_video_vae_sd_ops("encoder")

        assert sd_ops.apply_to_key("encoder.conv_in.conv.weight") == "conv_in.conv.weight"
        assert sd_ops.apply_to_key("vae.encoder.conv_in.conv.weight") == "conv_in.conv.weight"

    def test_distilled_gguf_uses_distilled_schedule_when_steps_are_explicitly_8(self) -> None:
        pipeline = LTXLowVRAMPipeline.__new__(LTXLowVRAMPipeline)
        pipeline._checkpoint_path = "/tmp/ltx-2.3_text_projection_bf16.safetensors"
        pipeline._gguf_path = "/tmp/ltx-2.3-22b-distilled-Q8_0.gguf"
        pipeline._lora_path = None
        pipeline._num_inference_steps = 8

        sigma_values = pipeline._get_sigma_schedule()

        assert len(sigma_values) == 9
        assert sigma_values != [float(1.0 - i / 8) for i in range(9)]

    def test_transformer_block_swap_wrapper_is_reused(self) -> None:
        pipeline = LTXLowVRAMPipeline.__new__(LTXLowVRAMPipeline)
        pipeline.device = torch.device("cpu")
        pipeline.vram_manager = _DummyVRAMManager()
        pipeline._block_swap_wrapper = None

        transformer = _TransformerWrapper(block_count=6)
        first_block = transformer.velocity_model.transformer_blocks[0]

        before = _hook_counts(first_block)
        pipeline._setup_block_swap_if_needed(transformer)
        after_first = _hook_counts(first_block)
        first_wrapper = pipeline._block_swap_wrapper

        pipeline._setup_block_swap_if_needed(transformer)
        after_second = _hook_counts(first_block)

        assert first_wrapper is not None
        assert pipeline._block_swap_wrapper is first_wrapper
        assert after_first[0] > before[0]
        assert after_first[1] > before[1]
        assert after_second == after_first

    def test_text_encoder_block_swap_wrapper_is_reused(self) -> None:
        pipeline = LTXLowVRAMPipeline.__new__(LTXLowVRAMPipeline)
        pipeline.device = torch.device("cpu")
        pipeline.vram_manager = _DummyVRAMManager()
        pipeline._te_block_swap_wrapper = None

        text_encoder = _TextEncoder(layer_count=8)
        first_layer = text_encoder.model.language_model.layers[0]

        before = _hook_counts(first_layer)
        wrapper_one = pipeline._setup_text_encoder_block_swap_cached(text_encoder)
        after_first = _hook_counts(first_layer)
        wrapper_two = pipeline._setup_text_encoder_block_swap_cached(text_encoder)
        after_second = _hook_counts(first_layer)

        assert wrapper_one is not None
        assert wrapper_two is wrapper_one
        assert pipeline._te_block_swap_wrapper is wrapper_one
        assert after_first[0] > before[0]
        assert after_first[1] > before[1]
        assert after_second == after_first


class TestOptimizedPipelineRegressions:
    def test_text_encoder_block_swap_wrapper_is_reused(self) -> None:
        pipeline = LTXOptimizedPipeline.__new__(LTXOptimizedPipeline)
        pipeline.device = torch.device("cpu")
        pipeline.vram_manager = _DummyVRAMManager()
        pipeline._te_block_swap_wrapper = None

        text_encoder = _TextEncoder(layer_count=8)
        first_layer = text_encoder.model.language_model.layers[0]

        before = _hook_counts(first_layer)
        wrapper_one = pipeline._setup_text_encoder_block_swap(text_encoder)
        after_first = _hook_counts(first_layer)
        wrapper_two = pipeline._setup_text_encoder_block_swap(text_encoder)
        after_second = _hook_counts(first_layer)

        assert wrapper_one is not None
        assert wrapper_two is wrapper_one
        assert pipeline._te_block_swap_wrapper is wrapper_one
        assert after_first[0] > before[0]
        assert after_first[1] > before[1]
        assert after_second == after_first

    def test_move_text_encoder_non_layers_keeps_lm_head_on_cpu(self) -> None:
        pipeline = LTXOptimizedPipeline.__new__(LTXOptimizedPipeline)
        pipeline.device = torch.device("cpu")

        text_encoder = _TextEncoder(layer_count=8)
        pipeline._move_text_encoder_non_layers_to_gpu(text_encoder)

        assert text_encoder.model.lm_head.weight.device.type == "cpu"
        assert text_encoder.model.lm_head.bias is not None
        assert text_encoder.model.lm_head.bias.device.type == "cpu"

    def test_high_tier_text_encoder_block_swap_keeps_subset_on_gpu(self) -> None:
        pipeline = LTXOptimizedPipeline.__new__(LTXOptimizedPipeline)
        pipeline.device = torch.device("cpu")
        pipeline.vram_manager = _DummyVRAMManager(tier=VRAMTier.HIGH, free_vram_mb=24_000)

        text_encoder = _TextEncoder(layer_count=8)
        wrapper = pipeline._create_text_encoder_block_swap(text_encoder)

        assert wrapper is not None
        assert wrapper.block_count == 8
        assert wrapper.blocks_to_keep_on_gpu == 6
        assert wrapper.blocks_to_keep_on_gpu < wrapper.block_count
