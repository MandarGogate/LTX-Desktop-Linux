"""VRAM tier classification and memory budget management.

Inspired by ComfyUI's tiered memory management approach. Classifies GPUs into
tiers and provides resolution/frame limits, quantization policy, and offloading
strategy recommendations for each tier.
"""

from __future__ import annotations

import gc
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class VRAMTier(Enum):
    """GPU VRAM tier classification."""

    HIGH = "high"          # ≥24 GB (RTX 4090, RTX 3090, A5000, etc.)
    MEDIUM = "medium"      # 16-23 GB (RTX 4080, RTX 5080, A4000, etc.)
    LOW = "low"            # 12-15 GB (RTX 4070, RTX 3060 12GB, etc.)
    VERY_LOW = "very_low"  # 8-11 GB (RTX 4060, RTX 3060 8GB, etc.)


class OffloadStrategy(Enum):
    """Model offloading strategy."""

    NONE = "none"                # All models on GPU (≥31GB, original behavior)
    SEQUENTIAL = "sequential"    # One model at a time on GPU
    BLOCK_SWAP = "block_swap"    # Transformer blocks swapped between CPU/GPU


# Resolution maps per tier for 16:9 aspect ratio
_RESOLUTION_MAP_16_9: dict[VRAMTier, dict[str, tuple[int, int]]] = {
    VRAMTier.HIGH: {
        "1080p": (1920, 1088),
        "720p": (1280, 704),
        "540p": (960, 544),
    },
    VRAMTier.MEDIUM: {
        "720p": (1280, 704),
        "540p": (960, 544),
    },
    VRAMTier.LOW: {
        "540p": (960, 544),
        "480p": (768, 448),
    },
    VRAMTier.VERY_LOW: {
        "480p": (768, 448),
        "360p": (640, 384),
    },
}

# Maximum number of transformer blocks to keep on GPU during block swap.
# Lower tiers keep fewer blocks on GPU.
_BLOCK_SWAP_KEEP_ON_GPU: dict[VRAMTier, int] = {
    VRAMTier.HIGH: 0,       # No block swap needed
    VRAMTier.MEDIUM: 12,    # Keep 12 of 48 blocks on GPU
    VRAMTier.LOW: 6,        # Keep 6 of 48 blocks on GPU
    VRAMTier.VERY_LOW: 3,   # Keep only 3 of 48 blocks on GPU
}


class VRAMManager:
    """Manages GPU VRAM budget, tier classification, and offloading strategy.

    This is the central coordinator for low-VRAM operation. It determines:
    - Which VRAM tier the GPU falls into
    - Maximum resolution and frame counts
    - Quantization policy (FP8, GGUF quantized, etc.)
    - Offloading strategy (sequential, block swap)
    - Tiling configuration for VAE decode
    """

    def __init__(self, device: torch.device, total_vram_gb: int) -> None:
        import torch as _torch

        self.device = device
        self.total_vram_gb = total_vram_gb
        self.tier = self._classify_tier(total_vram_gb)
        self._torch = _torch
        logger.info(
            "VRAMManager initialized: device=%s vram=%dGB tier=%s",
            device, total_vram_gb, self.tier.value,
        )

    @staticmethod
    def _classify_tier(vram_gb: int) -> VRAMTier:
        """Classify GPU into a VRAM tier."""
        if vram_gb >= 24:
            return VRAMTier.HIGH
        if vram_gb >= 16:
            return VRAMTier.MEDIUM
        if vram_gb >= 12:
            return VRAMTier.LOW
        return VRAMTier.VERY_LOW

    @property
    def offload_strategy(self) -> OffloadStrategy:
        """Determine the offloading strategy for this tier."""
        if self.total_vram_gb >= 31:
            return OffloadStrategy.NONE
        if self.tier in (VRAMTier.LOW, VRAMTier.VERY_LOW):
            return OffloadStrategy.BLOCK_SWAP
        return OffloadStrategy.SEQUENTIAL

    @property
    def block_swap_keep_on_gpu(self) -> int:
        """Number of transformer blocks to keep on GPU during block swap."""
        return _BLOCK_SWAP_KEEP_ON_GPU.get(self.tier, 0)

    def get_available_resolutions(self) -> dict[str, tuple[int, int]]:
        """Return available resolutions for 16:9 aspect ratio."""
        return _RESOLUTION_MAP_16_9.get(self.tier, _RESOLUTION_MAP_16_9[VRAMTier.VERY_LOW]).copy()

    def get_max_resolution(self) -> tuple[int, int]:
        """Return the maximum (width, height) for this tier (16:9)."""
        resolutions = self.get_available_resolutions()
        # Return the highest resolution available
        for label in ("1080p", "720p", "540p", "480p", "360p"):
            if label in resolutions:
                return resolutions[label]
        return (640, 384)

    def get_max_frames(self, width: int, height: int, fps: int) -> int:
        """Estimate maximum frame count based on VRAM budget.

        Uses a conservative heuristic: reserves ~60% of VRAM for models and
        uses the remaining 40% for inference working memory. Each frame in
        latent space costs approximately (width * height * 0.001) MB.
        """
        pixels_per_frame = width * height
        # Reserve memory for the largest model (transformer) + overhead
        vram_for_inference_mb = self.total_vram_gb * 1024 * 0.4
        cost_per_frame_mb = pixels_per_frame * 0.001
        if cost_per_frame_mb <= 0:
            return 9

        raw_frames = int(vram_for_inference_mb / cost_per_frame_mb)
        # Align to 8-frame boundary + 1 (LTX requirement)
        aligned = ((raw_frames // 8) * 8) + 1
        return max(9, min(aligned, 201))

    def should_use_gguf(self) -> bool:
        """Whether to prefer GGUF quantized models for this tier."""
        # GGUF models use significantly less VRAM via quantization
        return self.tier in (VRAMTier.LOW, VRAMTier.VERY_LOW, VRAMTier.MEDIUM)

    def get_recommended_gguf_quant(self) -> str:
        """Return recommended GGUF quantization level."""
        match self.tier:
            case VRAMTier.HIGH:
                return "Q8_0"
            case VRAMTier.MEDIUM:
                return "Q5_1"
            case VRAMTier.LOW:
                return "Q4_K_M"
            case VRAMTier.VERY_LOW:
                return "Q4_0"

    def get_fp8_enabled(self) -> bool:
        """Whether FP8 quantization should be used (CUDA only)."""
        from services.services_utils import device_supports_fp8

        return device_supports_fp8(self.device)

    def ensure_on_gpu(self, model_name: str, model: torch.nn.Module) -> None:
        """Move a model to GPU with memory cleanup."""
        try:
            model.to(self.device)
            self._sync_device()
            logger.debug("Moved %s to %s", model_name, self.device)
        except Exception:
            logger.error("Failed to move %s to GPU", model_name, exc_info=True)
            raise

    def offload_to_cpu(self, model_name: str, model: torch.nn.Module) -> None:
        """Offload a model from GPU to CPU and free VRAM."""
        try:
            model.to("cpu")
            self._sync_device()
            self._empty_cache()
            gc.collect()
            logger.debug("Offloaded %s to CPU", model_name)
        except Exception:
            logger.warning("Failed to offload %s to CPU", model_name, exc_info=True)

    def _sync_device(self) -> None:
        """Synchronize the GPU device."""
        from services.services_utils import sync_device

        sync_device(self.device)

    def _empty_cache(self) -> None:
        """Empty the GPU memory cache."""
        from services.services_utils import empty_device_cache

        empty_device_cache(self.device)

    def cleanup(self) -> None:
        """Aggressive memory cleanup."""
        self._sync_device()
        self._empty_cache()
        gc.collect()
        self._empty_cache()

    def get_current_vram_usage_mb(self) -> int:
        """Get current VRAM usage in MB (CUDA only)."""
        try:
            if self._torch.cuda.is_available():
                return int(self._torch.cuda.memory_allocated(self.device) / (1024 * 1024))
        except Exception:
            pass
        return 0

    def get_free_vram_mb(self) -> int:
        """Get estimated free VRAM in MB."""
        total_mb = self.total_vram_gb * 1024
        used_mb = self.get_current_vram_usage_mb()
        return max(0, total_mb - used_mb)

    def to_profile_dict(self) -> dict[str, object]:
        """Return a JSON-serializable profile for the /vram-profile endpoint."""
        max_w, max_h = self.get_max_resolution()
        return {
            "tier": self.tier.value,
            "vram_total_gb": self.total_vram_gb,
            "offload_strategy": self.offload_strategy.value,
            "block_swap_blocks_on_gpu": self.block_swap_keep_on_gpu,
            "max_resolution_width": max_w,
            "max_resolution_height": max_h,
            "available_resolutions": {
                k: {"width": v[0], "height": v[1]}
                for k, v in self.get_available_resolutions().items()
            },
            "max_frames_540p_25fps": self.get_max_frames(960, 544, 25),
            "max_frames_720p_25fps": self.get_max_frames(1280, 704, 25),
            "max_frames_1080p_25fps": self.get_max_frames(1920, 1088, 25),
            "gguf_recommended": self.should_use_gguf(),
            "gguf_quant_level": self.get_recommended_gguf_quant(),
            "fp8_enabled": self.get_fp8_enabled(),
        }
