"""Runtime policy decisions for forced API mode."""

from __future__ import annotations

# Minimum VRAM (GB) required for local generation.
# With sequential offloading + GGUF + block swap, 8GB is feasible.
MIN_LOCAL_VRAM_GB = 8


def decide_force_api_generations(system: str, cuda_available: bool, vram_gb: int | None) -> bool:
    """Return whether API-only generation must be forced for this runtime.

    With low-VRAM support (sequential offloading, GGUF quantized models,
    and block swap), consumer GPUs with ≥8 GB VRAM can generate locally.
    """
    if system == "Darwin":
        # macOS MPS: experimental local support, allow if enough memory
        return True

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        if vram_gb is None:
            return True
        return vram_gb < MIN_LOCAL_VRAM_GB

    # Fail closed for non-target platforms unless explicitly relaxed.
    return True
