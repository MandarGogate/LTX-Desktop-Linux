"""Block swap mechanism for large transformer models.

Implements ComfyUI-style block swapping where transformer blocks are moved
between CPU and GPU on demand during forward passes. This allows running
models that wouldn't fit entirely in VRAM.

How it works:
1. At init, only `blocks_to_keep_on_gpu` blocks stay on GPU
2. Remaining blocks are offloaded to CPU
3. During forward pass, each block is:
   a. Moved to GPU (if not already there)
   b. Executed
   c. Moved back to CPU (if it wasn't in the "keep on GPU" set)
4. Optionally uses CUDA streams for async prefetching of the next block

Performance characteristics:
- PCIe 4.0 x16: ~25 GB/s transfer → ~100ms per 2.5GB block
- PCIe 3.0 x16: ~15 GB/s transfer → ~170ms per 2.5GB block
- With async prefetch: overlaps compute and transfer, reducing overhead by ~50%
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class BlockSwapTransformerWrapper:
    """Wraps a transformer model to swap blocks between CPU and GPU.

    This is modeled after ComfyUI's approach to handling large transformers
    on consumer GPUs. The key insight is that transformer blocks are processed
    sequentially, so only one block needs to be on GPU at a time.

    Usage:
        wrapper = BlockSwapTransformerWrapper(
            transformer=model,
            device=torch.device("cuda"),
            blocks_to_keep_on_gpu=5,
        )
        # The transformer's forward pass now auto-swaps blocks
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        device: torch.device,
        blocks_to_keep_on_gpu: int = 5,
        use_async_prefetch: bool = True,
    ) -> None:
        import torch as _torch

        self._torch = _torch
        self.device = device
        self.cpu_device = _torch.device("cpu")
        self.blocks_to_keep_on_gpu = blocks_to_keep_on_gpu
        self.use_async_prefetch = use_async_prefetch and device.type == "cuda"
        self._transformer = transformer
        self._original_forward: Any = None
        self._block_names: list[str] = []
        self._blocks: list[torch.nn.Module] = []
        self._gpu_block_indices: set[int] = set()

        # CUDA streams for async prefetch
        self._transfer_stream: torch.cuda.Stream | None = None
        if self.use_async_prefetch:
            self._transfer_stream = _torch.cuda.Stream(device=device)

        self._setup_blocks()

    def _setup_blocks(self) -> None:
        """Identify transformer blocks and set up offloading.

        Looks for common patterns in LTX/diffusers transformer architectures:
        - model.transformer_blocks (common in diffusers)
        - model.blocks (common in some architectures)
        - model.layers (fallback)
        """
        transformer = self._transformer
        blocks: list[tuple[str, Any]] = []

        # Try common block container names
        for attr_name in ("transformer_blocks", "blocks", "layers", "encoder_layers"):
            container = getattr(transformer, attr_name, None)
            if container is not None and hasattr(container, "__len__") and len(container) > 1:
                for i, block in enumerate(container):
                    blocks.append((f"{attr_name}.{i}", block))
                break

        if not blocks:
            logger.warning(
                "Could not find transformer blocks for block swap. "
                "Model attributes: %s",
                [name for name, _ in transformer.named_children()],
            )
            return

        self._block_names = [name for name, _ in blocks]
        self._blocks = [block for _, block in blocks]
        total_blocks = len(self._blocks)

        logger.info(
            "Block swap: found %d blocks, keeping %d on GPU",
            total_blocks, min(self.blocks_to_keep_on_gpu, total_blocks),
        )

        # Keep the first N blocks on GPU, offload the rest
        for i, block in enumerate(self._blocks):
            if i < self.blocks_to_keep_on_gpu:
                block.to(self.device)
                self._gpu_block_indices.add(i)
            else:
                block.to(self.cpu_device)

        # Install forward hooks
        self._install_hooks()

    def _install_hooks(self) -> None:
        """Install pre-forward and post-forward hooks on each block.

        Pre-forward: ensure block is on GPU (and prefetch next block)
        Post-forward: offload block back to CPU if not in keep set
        """
        for i, block in enumerate(self._blocks):
            # Use closures to capture block index
            def make_pre_hook(idx: int) -> Any:
                def pre_hook(module: Any, inputs: Any) -> None:
                    self._on_block_pre_forward(idx)
                return pre_hook

            def make_post_hook(idx: int) -> Any:
                def post_hook(module: Any, inputs: Any, output: Any) -> None:
                    self._on_block_post_forward(idx)
                return post_hook

            block.register_forward_pre_hook(make_pre_hook(i))
            block.register_forward_hook(make_post_hook(i))

    def _on_block_pre_forward(self, block_idx: int) -> None:
        """Called before a block's forward pass — ensure it's on GPU."""
        if block_idx in self._gpu_block_indices:
            return

        # Move this block to GPU
        block = self._blocks[block_idx]
        block.to(self.device)
        self._gpu_block_indices.add(block_idx)

        # Async prefetch the next block
        if self.use_async_prefetch and self._transfer_stream is not None:
            next_idx = block_idx + 1
            if next_idx < len(self._blocks) and next_idx not in self._gpu_block_indices:
                with self._torch.cuda.stream(self._transfer_stream):
                    self._blocks[next_idx].to(self.device, non_blocking=True)
                    self._gpu_block_indices.add(next_idx)

    def _on_block_post_forward(self, block_idx: int) -> None:
        """Called after a block's forward pass — offload if not in keep set."""
        if block_idx < self.blocks_to_keep_on_gpu:
            return  # This block should stay on GPU

        # Wait for any async transfer to complete
        if self.use_async_prefetch and self._transfer_stream is not None:
            self._transfer_stream.synchronize()

        # Offload to CPU
        block = self._blocks[block_idx]
        block.to(self.cpu_device)
        self._gpu_block_indices.discard(block_idx)

    def offload_all(self) -> None:
        """Offload all blocks to CPU (called when done with transformer)."""
        for i, block in enumerate(self._blocks):
            if i in self._gpu_block_indices:
                block.to(self.cpu_device)
        self._gpu_block_indices.clear()

        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()

    def restore_gpu_blocks(self) -> None:
        """Restore the 'keep on GPU' blocks after offload_all."""
        for i in range(min(self.blocks_to_keep_on_gpu, len(self._blocks))):
            self._blocks[i].to(self.device)
            self._gpu_block_indices.add(i)

    @property
    def block_count(self) -> int:
        """Total number of transformer blocks."""
        return len(self._blocks)

    @property
    def gpu_block_count(self) -> int:
        """Number of blocks currently on GPU."""
        return len(self._gpu_block_indices)

    def get_stats(self) -> dict[str, object]:
        """Return block swap statistics."""
        return {
            "total_blocks": self.block_count,
            "gpu_blocks": self.gpu_block_count,
            "blocks_to_keep": self.blocks_to_keep_on_gpu,
            "async_prefetch": self.use_async_prefetch,
        }
