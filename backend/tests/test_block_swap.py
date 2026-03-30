"""Tests for block swap transformer wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from services.block_swap.block_swap import BlockSwapTransformerWrapper


class SimpleBlock(nn.Module):
    """Simple block for testing."""

    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleTransformer(nn.Module):
    """Simple transformer with named blocks for testing block swap."""

    def __init__(self, num_blocks: int = 10, block_size: int = 16) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [SimpleBlock(block_size) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(block_size, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return self.head(x)


class TestBlockSwapSetup:
    def test_discovers_blocks(self) -> None:
        model = SimpleTransformer(num_blocks=10)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=3,
            use_async_prefetch=False,
        )
        assert wrapper.block_count == 10

    def test_keeps_specified_blocks_on_device(self) -> None:
        model = SimpleTransformer(num_blocks=10)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=3,
            use_async_prefetch=False,
        )
        # On CPU device, all blocks end up on CPU, but the indices are tracked
        assert wrapper.gpu_block_count == 3

    def test_zero_keep_blocks(self) -> None:
        model = SimpleTransformer(num_blocks=5)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=0,
            use_async_prefetch=False,
        )
        assert wrapper.gpu_block_count == 0

    def test_keep_more_than_total(self) -> None:
        """If blocks_to_keep > total blocks, all blocks stay on GPU."""
        model = SimpleTransformer(num_blocks=5)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=10,
            use_async_prefetch=False,
        )
        assert wrapper.gpu_block_count == 5


class TestBlockSwapForward:
    def test_forward_pass_works(self) -> None:
        """Model should produce output even with block swap active."""
        model = SimpleTransformer(num_blocks=5, block_size=16)
        _wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=2,
            use_async_prefetch=False,
        )
        x = torch.randn(1, 16)
        output = model(x)
        assert output.shape == (1, 16)

    def test_multiple_forward_passes(self) -> None:
        """Multiple forward passes should work correctly."""
        model = SimpleTransformer(num_blocks=5, block_size=16)
        _wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=1,
            use_async_prefetch=False,
        )
        for _ in range(3):
            x = torch.randn(1, 16)
            output = model(x)
            assert output.shape == (1, 16)


class TestBlockSwapOffload:
    def test_offload_all(self) -> None:
        model = SimpleTransformer(num_blocks=5)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=3,
            use_async_prefetch=False,
        )
        wrapper.offload_all()
        assert wrapper.gpu_block_count == 0

    def test_restore_after_offload(self) -> None:
        model = SimpleTransformer(num_blocks=5)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=3,
            use_async_prefetch=False,
        )
        wrapper.offload_all()
        assert wrapper.gpu_block_count == 0

        wrapper.restore_gpu_blocks()
        assert wrapper.gpu_block_count == 3


class TestBlockSwapStats:
    def test_stats_structure(self) -> None:
        model = SimpleTransformer(num_blocks=8)
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=2,
            use_async_prefetch=False,
        )
        stats = wrapper.get_stats()
        assert stats["total_blocks"] == 8
        assert stats["gpu_blocks"] == 2
        assert stats["blocks_to_keep"] == 2
        assert stats["async_prefetch"] is False


class TestNoBlocksFound:
    def test_model_without_standard_block_names(self) -> None:
        """Model without recognized block containers should handle gracefully."""

        class CustomModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(16, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = CustomModel()
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cpu"),
            blocks_to_keep_on_gpu=2,
            use_async_prefetch=False,
        )
        assert wrapper.block_count == 0
