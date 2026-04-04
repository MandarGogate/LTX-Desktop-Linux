"""Optimized block swap with pinned memory and true async double-buffering.

Key improvements over original block_swap.py:
1. Pin CPU memory for faster PCIe transfers (~2x throughput)
2. True double-buffering: compute on block N while transferring block N+2
3. Don't synchronize the transfer stream in post_forward — let it overlap
4. Batch offload: move blocks back to CPU in post_forward without sync
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class FastBlockSwapWrapper:
    """Block swap with pinned memory + async double-buffering.

    Compared to BlockSwapTransformerWrapper:
    - Pins CPU tensors for ~2x PCIe transfer speed
    - Prefetches block N+2 (not N+1) so transfer fully overlaps with compute
    - Defers offload to avoid synchronization stalls
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        device: torch.device,
        blocks_to_keep_on_gpu: int = 5,
        prefetch_distance: int = 2,
    ) -> None:
        import torch as _torch

        self._torch = _torch
        self.device = device
        self.cpu_device = _torch.device("cpu")
        self.blocks_to_keep_on_gpu = blocks_to_keep_on_gpu
        self.prefetch_distance = prefetch_distance
        self._transformer = transformer
        self._block_names: list[str] = []
        self._blocks: list[torch.nn.Module] = []
        self._gpu_block_indices: set[int] = set()
        # Blocks currently being prefetched to GPU on transfer stream.
        # They are "scheduled" for GPU but not yet safe to execute until
        # the transfer stream is synchronized.
        self._inflight_prefetch_indices: set[int] = set()
        self._pending_offload: list[int] = []

        # CUDA streams
        self._transfer_stream: torch.cuda.Stream | None = None
        self._offload_stream: torch.cuda.Stream | None = None
        self._compute_stream: torch.cuda.Stream | None = None
        if device.type == "cuda":
            self._transfer_stream = _torch.cuda.Stream(device=device)
            self._offload_stream = _torch.cuda.Stream(device=device)
            self._compute_stream = _torch.cuda.current_stream(device=device)

        self._setup_blocks()

    def _setup_blocks(self) -> None:
        transformer = self._transformer
        blocks: list[tuple[str, Any]] = []

        search_targets: list[tuple[str, Any]] = [("<root>", transformer)]
        for sub_name in ("velocity_model", "model", "inner_model"):
            sub = getattr(transformer, sub_name, None)
            if sub is not None:
                search_targets.append((sub_name, sub))

        for parent_name, parent in search_targets:
            for attr_name in (
                "transformer_blocks",
                "blocks",
                "layers",
                "encoder_layers",
            ):
                container = getattr(parent, attr_name, None)
                if (
                    container is not None
                    and hasattr(container, "__len__")
                    and len(container) > 1
                ):
                    prefix = (
                        f"{parent_name}.{attr_name}"
                        if parent_name != "<root>"
                        else attr_name
                    )
                    for i, block in enumerate(container):
                        blocks.append((f"{prefix}.{i}", block))
                    break
            if blocks:
                break

        if not blocks:
            logger.warning("Could not find transformer blocks for block swap")
            return

        self._block_names = [name for name, _ in blocks]
        self._blocks = [block for _, block in blocks]
        total_blocks = len(self._blocks)

        logger.info(
            "FastBlockSwap: %d blocks, keeping %d on GPU, prefetch_distance=%d",
            total_blocks,
            min(self.blocks_to_keep_on_gpu, total_blocks),
            self.prefetch_distance,
        )

        # Pin CPU memory for faster transfers and keep first N on GPU
        for i, block in enumerate(self._blocks):
            if i < self.blocks_to_keep_on_gpu:
                block.to(self.device)
                self._gpu_block_indices.add(i)
            else:
                block.to(self.cpu_device)
                self._pin_block(block)

        self._install_hooks()

    def _pin_block(self, block: torch.nn.Module) -> None:
        """Pin a CPU block's memory for faster async transfers."""
        if not self._torch.cuda.is_available():
            return
        try:
            for param in block.parameters():
                if param.device.type == "cpu" and not param.data.is_pinned():
                    param.data = param.data.pin_memory()
            for buf in block.buffers():
                if buf.device.type == "cpu" and not buf.data.is_pinned():
                    buf.data = buf.data.pin_memory()
        except Exception:
            pass  # Pinning failure is non-fatal

    def _install_hooks(self) -> None:
        for i, block in enumerate(self._blocks):

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
        # If this block was prefetched asynchronously, wait until the H2D
        # transfer is actually complete before executing it.
        if block_idx in self._inflight_prefetch_indices:
            if self._transfer_stream is not None:
                self._transfer_stream.synchronize()
            self._inflight_prefetch_indices.discard(block_idx)

        # Ensure this block is on GPU
        if block_idx not in self._gpu_block_indices:
            # Wait for any pending prefetch AND offload to complete.
            # The offload stream must be synced because a prior
            # _flush_offloads may have scheduled an async D2H copy for
            # this block.  Without the sync, _pin_block would read
            # partially-written CPU data and upload garbage weights.
            if self._transfer_stream is not None:
                self._transfer_stream.synchronize()
            if self._offload_stream is not None:
                self._offload_stream.synchronize()
            if block_idx not in self._gpu_block_indices:
                self._pin_block(self._blocks[block_idx])
                self._blocks[block_idx].to(self.device, non_blocking=False)
                self._gpu_block_indices.add(block_idx)

        # Process deferred offloads (non-blocking, on offload stream)
        self._flush_offloads()

        # Prefetch block N + prefetch_distance
        prefetch_idx = block_idx + self.prefetch_distance
        if (
            self._transfer_stream is not None
            and prefetch_idx < len(self._blocks)
            and prefetch_idx not in self._gpu_block_indices
        ):
            # Ensure any pending offload of the prefetch target has landed
            # on CPU before we try to pin + re-upload it.
            if self._offload_stream is not None:
                self._offload_stream.synchronize()
            with self._torch.cuda.stream(self._transfer_stream):
                self._pin_block(self._blocks[prefetch_idx])
                self._blocks[prefetch_idx].to(self.device, non_blocking=True)
                self._gpu_block_indices.add(prefetch_idx)
                self._inflight_prefetch_indices.add(prefetch_idx)

    def _on_block_post_forward(self, block_idx: int) -> None:
        # Defer offload — don't block here
        if block_idx >= self.blocks_to_keep_on_gpu:
            self._pending_offload.append(block_idx)

    def _flush_offloads(self) -> None:
        """Offload deferred blocks asynchronously."""
        if not self._pending_offload:
            return

        # Keep at most prefetch_distance blocks beyond the "keep" set
        # to avoid running out of VRAM while allowing overlap
        max_extra = self.prefetch_distance
        extra_on_gpu = len(self._gpu_block_indices) - self.blocks_to_keep_on_gpu
        if extra_on_gpu <= max_extra:
            return

        # Offload the oldest deferred blocks
        to_offload = self._pending_offload[: extra_on_gpu - max_extra]
        self._pending_offload = self._pending_offload[extra_on_gpu - max_extra :]

        for idx in to_offload:
            if idx in self._inflight_prefetch_indices:
                # Don't race an in-flight H2D prefetch with D2H offload.
                continue
            if idx in self._gpu_block_indices and idx >= self.blocks_to_keep_on_gpu:
                block = self._blocks[idx]
                if self._offload_stream is not None:
                    current_stream = self._torch.cuda.current_stream(device=self.device)
                    self._offload_stream.wait_stream(current_stream)
                    with self._torch.cuda.stream(self._offload_stream):
                        block.to(self.cpu_device, non_blocking=True)
                else:
                    block.to(self.cpu_device, non_blocking=False)
                self._gpu_block_indices.discard(idx)

    def offload_all(self) -> None:
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()
        if self._offload_stream is not None:
            self._offload_stream.synchronize()

        for i, block in enumerate(self._blocks):
            if i in self._gpu_block_indices:
                if self._offload_stream is not None:
                    current_stream = self._torch.cuda.current_stream(device=self.device)
                    self._offload_stream.wait_stream(current_stream)
                    with self._torch.cuda.stream(self._offload_stream):
                        block.to(self.cpu_device, non_blocking=True)
                else:
                    block.to(self.cpu_device)
        self._gpu_block_indices.clear()
        self._inflight_prefetch_indices.clear()
        self._pending_offload.clear()
        if self._offload_stream is not None:
            self._offload_stream.synchronize()

        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()

    def restore_gpu_blocks(self) -> None:
        self._inflight_prefetch_indices.clear()
        for i in range(min(self.blocks_to_keep_on_gpu, len(self._blocks))):
            self._blocks[i].to(self.device)
            self._gpu_block_indices.add(i)

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def gpu_block_count(self) -> int:
        return len(self._gpu_block_indices)
