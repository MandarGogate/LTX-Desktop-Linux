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

    NONE = "none"                    # All models on GPU (≥48GB, original behavior)
    SEQUENTIAL = "sequential"        # One model at a time on GPU (no block swap)
    BLOCK_SWAP = "block_swap"        # Sequential offloading + transformer block swap
    BLOCK_SWAP_AGGRESSIVE = "block_swap_aggressive"  # Aggressive block swap for max resolution


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
# Maximum number of transformer blocks to keep on GPU during block swap.
# Lower values = less VRAM but more CPU↔GPU transfers.
# With async prefetch, keeping 2-3 blocks gives good overlap.
_BLOCK_SWAP_KEEP_ON_GPU: dict[VRAMTier, int] = {
    VRAMTier.HIGH: 5,       # Keep 5 of 48 blocks — FP8 block=370MB → ~1.85GB
    VRAMTier.MEDIUM: 4,     # More conservative than HIGH for 16-24GB low-VRAM runs
    VRAMTier.LOW: 3,        # Keep 3 of 48 blocks on GPU
    VRAMTier.VERY_LOW: 2,   # Keep only 2 of 48 blocks on GPU
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

    def __init__(
        self,
        device: torch.device,
        total_vram_gb: int,
        *,
        user_blocks_on_gpu: int = -1,
        user_run_mode: str = "auto",
    ) -> None:
        import torch as _torch

        self.device = device
        self.total_vram_gb = total_vram_gb
        self._user_blocks_on_gpu = user_blocks_on_gpu
        self._user_run_mode = user_run_mode
        self._has_logged_block_cap = False
        self.tier = self._resolve_tier(total_vram_gb, user_run_mode)
        self._torch = _torch
        logger.info(
            "VRAMManager initialized: device=%s vram=%dGB tier=%s run_mode=%s user_blocks=%d",
            device, total_vram_gb, self.tier.value, user_run_mode, user_blocks_on_gpu,
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

    @staticmethod
    def _resolve_tier(vram_gb: int, user_run_mode: str) -> VRAMTier:
        """Resolve tier from user run mode or auto-detect."""
        mode_to_tier: dict[str, VRAMTier] = {
            "high_vram": VRAMTier.HIGH,
            "medium_vram": VRAMTier.MEDIUM,
            "low_vram": VRAMTier.LOW,
            "very_low_vram": VRAMTier.VERY_LOW,
        }
        if user_run_mode in mode_to_tier:
            return mode_to_tier[user_run_mode]
        return VRAMManager._classify_tier(vram_gb)

    @property
    def offload_strategy(self) -> OffloadStrategy:
        """Determine the offloading strategy for this tier.

        The LTX 2.3 transformer is ~35GB bf16 / ~18GB FP8, so even 24GB GPUs
        cannot hold it entirely. Block swap is required for all tiers < 48GB.
        """
        if self.total_vram_gb >= 48:
            return OffloadStrategy.NONE
        if self.tier in (VRAMTier.LOW, VRAMTier.VERY_LOW):
            return OffloadStrategy.BLOCK_SWAP_AGGRESSIVE
        # HIGH and MEDIUM: sequential offloading + block swap
        return OffloadStrategy.BLOCK_SWAP

    @property
    def block_swap_keep_on_gpu(self) -> int:
        """Number of transformer blocks to keep on GPU during block swap.

        If the user has set a custom value (>= 0), use that. Otherwise,
        use the tier-based default.
        """
        auto_blocks = _BLOCK_SWAP_KEEP_ON_GPU.get(self.tier, 0)
        safe_max = self._max_safe_block_swap_keep_on_gpu()
        if self._user_blocks_on_gpu < 0:
            return min(auto_blocks, safe_max)

        requested = min(self._user_blocks_on_gpu, 48)
        effective = min(requested, safe_max)
        if requested > effective and not self._has_logged_block_cap:
            logger.warning(
                "Capping block-swap GPU blocks from %d to %d for %dGB/%s to avoid OOM",
                requested,
                effective,
                self.total_vram_gb,
                self.tier.value,
            )
            self._has_logged_block_cap = True
        return effective

    def _max_safe_block_swap_keep_on_gpu(self) -> int:
        """Return the runtime safety cap for resident transformer blocks.

        User overrides are intentionally bounded here. Large values can force
        tens of gigabytes of blocks onto GPU before denoising starts, which
        defeats low-VRAM mode and reliably OOMs on 24GB-class cards.
        """
        if self.total_vram_gb >= 48:
            return 48
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

        With sequential offloading + block swap, the VRAM budget during
        denoising is dominated by:
        - Active transformer blocks (~2-4GB)
        - Latent tensors (scale with resolution × frames)
        - Intermediate activations (~2× latent size)

        Latent space is (H/32)×(W/32)×(F/8) × channels(128) × 2 bytes (bf16).
        Working memory is roughly 3× latent size for noise, predicted noise, etc.
        """
        # Latent dimensions
        lat_h = height // 32
        lat_w = width // 32
        channels = 128
        bytes_per_element = 2  # bf16

        # During denoising, we need: latent + noise + predicted + intermediate
        # ≈ 4 copies of the latent tensor
        copies_needed = 4

        # VRAM available for latents during denoising phase
        # With block swap: only a few blocks on GPU + non-block overhead
        blocks_on_gpu = self.block_swap_keep_on_gpu
        fp8_block_mb = 370  # FP8 block is ~370MB
        bf16_block_mb = 738  # bf16 block is ~738MB
        block_mb = fp8_block_mb if self.get_fp8_enabled() else bf16_block_mb
        transformer_overhead_mb = blocks_on_gpu * block_mb + 1024  # +1GB for non-block parts

        vram_total_mb = self.total_vram_gb * 1024
        vram_for_latents_mb = max(vram_total_mb - transformer_overhead_mb - 512, 1024)  # 512MB safety margin

        # Cost per frame in latent space
        cost_per_frame_bytes = lat_h * lat_w * channels * bytes_per_element * copies_needed
        cost_per_frame_mb = cost_per_frame_bytes / (1024 * 1024)
        # Also account for temporal latent compression (frames/8)
        # Actually each latent frame = 1/8 of a video frame
        cost_per_8_frames_mb = cost_per_frame_mb  # one latent frame covers 8 video frames

        if cost_per_8_frames_mb <= 0:
            return 9

        raw_latent_frames = int(vram_for_latents_mb / cost_per_8_frames_mb)
        raw_video_frames = raw_latent_frames * 8

        # Align to 8-frame boundary + 1 (LTX requirement: frames = 8k + 1)
        aligned = ((raw_video_frames // 8) * 8) + 1
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
        # Second gc pass catches reference cycles freed by first pass
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

    def get_vram_debug_stats_mb(self) -> dict[str, int]:
        """Return VRAM stats comparable to PyTorch and driver-level views."""
        stats = {
            "memory_allocated_mb": 0,
            "memory_reserved_mb": 0,
            "driver_used_mb": 0,
        }
        try:
            if not self._torch.cuda.is_available():
                return stats

            stats["memory_allocated_mb"] = int(
                self._torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            )
            stats["memory_reserved_mb"] = int(
                self._torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            )

            free_bytes, total_bytes = self._torch.cuda.mem_get_info(self.device)
            stats["driver_used_mb"] = int((total_bytes - free_bytes) / (1024 * 1024))
        except Exception:
            pass
        return stats

    def get_free_vram_mb(self) -> int:
        """Get estimated free VRAM in MB."""
        total_mb = self.total_vram_gb * 1024
        used_mb = self.get_current_vram_usage_mb()
        return max(0, total_mb - used_mb)

    def to_profile_dict(self) -> dict[str, object]:
        """Return a JSON-serializable profile for the /vram-profile endpoint."""
        max_w, max_h = self.get_max_resolution()
        auto_tier = self._classify_tier(self.total_vram_gb)
        auto_blocks = _BLOCK_SWAP_KEEP_ON_GPU.get(auto_tier, 0)
        return {
            "tier": self.tier.value,
            "vram_total_gb": self.total_vram_gb,
            "offload_strategy": self.offload_strategy.value,
            "block_swap_blocks_on_gpu": self.block_swap_keep_on_gpu,
            "auto_blocks_on_gpu": auto_blocks,
            "max_blocks": 48,
            "user_blocks_on_gpu": self._user_blocks_on_gpu,
            "user_run_mode": self._user_run_mode,
            "auto_tier": auto_tier.value,
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
            "run_modes": [
                {"value": "auto", "label": f"Auto ({auto_tier.value})", "description": "Detects optimal settings from your GPU"},
                {"value": "high_vram", "label": "High VRAM (≥24 GB)", "description": "Max quality, minimal offloading"},
                {"value": "medium_vram", "label": "Medium VRAM (16-23 GB)", "description": "Balanced quality and speed"},
                {"value": "low_vram", "label": "Low VRAM (12-15 GB)", "description": "Aggressive offloading, lower resolution"},
                {"value": "very_low_vram", "label": "Very Low VRAM (8-11 GB)", "description": "Maximum offloading, lowest resolution"},
            ],
        }
