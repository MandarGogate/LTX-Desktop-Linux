"""GGUF model loader for LTX-2.3 transformer models.

Loads quantized GGUF models from repos like:
- https://huggingface.co/unsloth/LTX-2.3-GGUF
- https://huggingface.co/Kijai/LTX2.3_comfy/

GGUF format stores quantized weights (Q4_0, Q4_K_M, Q5_1, Q8_0, etc.)
which dramatically reduce VRAM usage compared to bf16/fp16 safetensors.

Typical VRAM savings:
- bf16 safetensors: ~43 GB (full model)
- Q8_0 GGUF: ~12 GB
- Q5_1 GGUF: ~8 GB
- Q4_K_M GGUF: ~6 GB
- Q4_0 GGUF: ~5 GB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Known GGUF model repos and their quantization variants
# Actual filenames verified against huggingface.co/unsloth/LTX-2.3-GGUF
GGUF_REPOS: dict[str, dict[str, str]] = {
    "unsloth": {
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "Q8_0": "distilled/ltx-2.3-22b-distilled-Q8_0.gguf",
        "Q5_1": "distilled/ltx-2.3-22b-distilled-Q5_1.gguf",
        "Q4_K_M": "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
        "Q4_0": "distilled/ltx-2.3-22b-distilled-Q4_0.gguf",
    },
    "kijai": {
        "repo_id": "Kijai/LTX2.3_comfy",
        # Kijai's repo has safetensors converted for ComfyUI, check for GGUF variants
        "bf16": "ltx2.3_distilled_transformer_bf16.safetensors",
        "fp8": "ltx2.3_distilled_transformer_fp8_e4m3fn.safetensors",
    },
}


def get_gguf_filename(quant_level: str, source: str = "unsloth") -> str | None:
    """Get the GGUF filename for a given quantization level."""
    repo_info = GGUF_REPOS.get(source)
    if repo_info is None:
        return None
    return repo_info.get(quant_level)


def get_gguf_repo_id(source: str = "unsloth") -> str:
    """Get the HuggingFace repo ID for GGUF models."""
    return GGUF_REPOS[source]["repo_id"]


class GGUFModelLoader:
    """Loads GGUF quantized transformer models for LTX-2.3.

    The GGUF format packs quantized weights in a single file with metadata.
    This loader handles:
    1. Detecting available GGUF files in the models directory
    2. Loading GGUF weights and dequantizing on-the-fly during inference
    3. Providing a state_dict compatible interface for the transformer
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._gguf_dir = models_dir / "gguf"

    def find_gguf_model(self, preferred_quant: str = "Q8_0") -> Path | None:
        """Find a GGUF model file, preferring the specified quantization.

        Searches in order: preferred quant, then Q8_0, Q5_1, Q4_K_M, Q4_0.
        Searches gguf dir (including subdirs) and models dir.
        """
        search_order = [preferred_quant]
        for q in ["Q8_0", "Q5_1", "Q4_K_M", "Q4_0"]:
            if q not in search_order:
                search_order.append(q)

        search_dirs = [self._gguf_dir, self.models_dir]

        for quant in search_order:
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                # Search recursively for GGUF files matching this quant
                for gguf_file in search_dir.rglob(f"*{quant}*.gguf"):
                    return gguf_file

        # Fallback: any .gguf file
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for gguf_file in search_dir.rglob("*.gguf"):
                return gguf_file

        return None

    def is_available(self, preferred_quant: str = "Q8_0") -> bool:
        """Check if any GGUF model is available."""
        return self.find_gguf_model(preferred_quant) is not None

    @staticmethod
    def _dequantize_tensor(
        tensor: Any,
    ) -> Any:
        """Dequantize a single GGUF tensor to float32 numpy array.

        Handles F32 (passthrough), BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
        Q4_K, Q5_K, Q6_K, and other types supported by the gguf library.
        """
        import numpy as np

        from gguf import GGMLQuantizationType  # type: ignore[reportMissingImports]
        from gguf import dequantize  # type: ignore[reportMissingImports]

        qtype = GGMLQuantizationType(tensor.tensor_type)
        target_shape = tuple(reversed(tensor.shape))

        if qtype == GGMLQuantizationType.F32:
            # Already float32, just reshape
            return tensor.data.astype(np.float32).reshape(target_shape)

        # All other types: dequantize then reshape
        deq = dequantize(tensor.data, qtype)
        return deq.reshape(target_shape)

    @staticmethod
    def load_gguf_state_dict(
        gguf_path: Path,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load a GGUF file and return a state dict with dequantized tensors.

        This uses the `gguf` Python library to read quantized tensors,
        dequantizes them to float32, then converts to bfloat16 torch tensors
        for compatibility with the LTX transformer.
        """
        import torch as _torch

        try:
            from gguf import GGUFReader  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "The 'gguf' package is required for GGUF model loading. "
                "Install it with: pip install gguf"
            ) from None

        logger.info("Loading GGUF model from %s", gguf_path)
        reader = GGUFReader(str(gguf_path))

        state_dict: dict[str, Any] = {}
        for tensor in reader.tensors:
            name = tensor.name
            # Dequantize (handles Q4_0, Q4_1, Q5_1, Q8_0, BF16, F32, etc.)
            np_data = GGUFModelLoader._dequantize_tensor(tensor)
            t = _torch.from_numpy(np_data).to(dtype=_torch.bfloat16, device=device)
            state_dict[name] = t

        logger.info("Loaded %d tensors from GGUF", len(state_dict))
        return state_dict

    @staticmethod
    def load_gguf_sd_for_diffusers(
        gguf_path: Path,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load GGUF and return state dict with keys mapped for diffusers/LTX.

        The unsloth/LTX-2.3-GGUF repo uses bare key names that already match
        the LTXModel state dict (e.g. ``transformer_blocks.0.scale_shift_table``).
        Other community repos may prefix keys with ``model.diffusion_model.`` or
        ``transformer.`` — those prefixes are stripped here.
        """
        import torch as _torch

        try:
            from gguf import GGUFReader  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "The 'gguf' package is required. Install with: pip install gguf"
            ) from None

        logger.info("Loading GGUF (diffusers-compatible) from %s", gguf_path)
        reader = GGUFReader(str(gguf_path))

        state_dict: dict[str, Any] = {}
        for tensor in reader.tensors:
            name = tensor.name
            # Dequantize (handles Q4_0, Q4_1, Q5_1, Q8_0, BF16, F32, etc.)
            np_data = GGUFModelLoader._dequantize_tensor(tensor)
            t = _torch.from_numpy(np_data).to(dtype=_torch.bfloat16, device=device)

            # Common key remapping for community GGUF exports
            # (unsloth uses bare keys, but some repos add prefixes)
            remapped_name = name
            if name.startswith("model.diffusion_model."):
                remapped_name = name.replace("model.diffusion_model.", "")
            elif name.startswith("transformer."):
                remapped_name = name.replace("transformer.", "")

            state_dict[remapped_name] = t

        logger.info("Loaded %d tensors (diffusers-mapped) from GGUF", len(state_dict))
        return state_dict

    def get_gguf_info(self) -> dict[str, object]:
        """Return info about available GGUF models for API responses."""
        available: list[dict[str, object]] = []

        search_dirs = [self._gguf_dir, self.models_dir]
        seen: set[str] = set()

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for gguf_file in search_dir.rglob("*.gguf"):
                if gguf_file.name in seen:
                    continue
                seen.add(gguf_file.name)
                size_mb = gguf_file.stat().st_size / (1024 * 1024)
                # Infer quant level from filename
                quant = "unknown"
                for q in ["Q8_0", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q2_K"]:
                    if q in gguf_file.name:
                        quant = q
                        break
                available.append({
                    "filename": gguf_file.name,
                    "path": str(gguf_file),
                    "size_mb": round(size_mb, 1),
                    "quant_level": quant,
                })

        return {
            "available_models": available,
            "gguf_dir": str(self._gguf_dir),
            "models_dir": str(self.models_dir),
        }
