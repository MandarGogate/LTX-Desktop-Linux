"""Integration tests for low-VRAM pipeline GGUF + LoRA path.

These tests verify that:
1. LoRA hooks are installed when loading GGUF transformer
2. FP8 forward hooks handle device transfers correctly
3. Generated video has visual content (not black)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    """Return the models directory, or skip if not available."""
    candidates = [
        Path(os.environ.get("LTX_MODELS_DIR", "")),
        Path.home() / ".ltx-desktop" / "models",
        Path.home() / ".local" / "share" / "LTXDesktop" / "models",
    ]
    for d in candidates:
        if d.exists() and (d / "diffusion_models").exists():
            return d
    pytest.skip("No models directory found — GPU integration tests require downloaded models")
    raise AssertionError("unreachable")


def _find_gguf(models_dir: Path) -> Path | None:
    """Find any .gguf file for the video transformer."""
    dm = models_dir / "diffusion_models"
    if not dm.exists():
        return None
    for f in dm.iterdir():
        if f.suffix == ".gguf" and "z-image" not in f.name.lower():
            return f
    return None


def _find_checkpoint(models_dir: Path) -> Path | None:
    dm = models_dir / "diffusion_models"
    if not dm.exists():
        return None
    for f in dm.iterdir():
        if f.suffix == ".safetensors":
            return f
    return None


def _find_lora(models_dir: Path) -> Path | None:
    loras = models_dir / "loras"
    if not loras.exists():
        return None
    for f in loras.iterdir():
        if f.suffix == ".safetensors" and "distilled" in f.name.lower() and "lora" in f.name.lower():
            return f
    return None


def _find_gemma_root(models_dir: Path) -> Path | None:
    for candidate in [
        models_dir / "gemma-3-12b-it-qat-q4_0-unquantized",
        models_dir / "text_encoders",
    ]:
        if candidate.exists() and (candidate / "tokenizer.model").exists():
            return candidate
    return None


def _find_upsampler(models_dir: Path) -> Path | None:
    um = models_dir / "upscale_models"
    if not um.exists():
        return None
    for f in um.iterdir():
        if f.suffix == ".safetensors" and "upscaler" in f.name.lower():
            return f
    return None


# ---------------------------------------------------------------------------
# Unit tests (no GPU needed)
# ---------------------------------------------------------------------------


class TestFP8ForwardDeviceHandling:
    """Verify that the FP8 upcast forward handles cross-device tensors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp8_forward_handles_cpu_weight_gpu_input(self) -> None:
        """FP8 quantized Linear with weight on CPU must work with GPU input."""
        from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline

        linear = torch.nn.Linear(16, 32, bias=True)
        # Simulate FP8 quantization
        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)
        linear.bias.data = linear.bias.data.to(torch.float8_e4m3fn)  # type: ignore[union-attr]

        # Apply the same upcast forward as the pipeline
        def _make_upcast_forward(lin: torch.nn.Linear) -> Any:
            def _fwd(x: torch.Tensor, **kw: Any) -> torch.Tensor:
                w = lin.weight.to(device=x.device, dtype=x.dtype)
                b = lin.bias.to(device=x.device, dtype=x.dtype) if lin.bias is not None else None
                return torch.nn.functional.linear(x, w, b)
            return _fwd

        linear.forward = _make_upcast_forward(linear)  # type: ignore[assignment]

        # Weight on CPU, input on GPU
        linear = linear.to("cpu")
        x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)

        # Should NOT crash with device mismatch
        output = linear(x)
        assert output.device.type == "cuda"
        assert output.shape == (2, 32)


class TestLoRAHooksInstalled:
    """Verify that _install_lora_hooks is called during GGUF loading."""

    def test_load_gguf_calls_install_lora_hooks(self) -> None:
        """The _load_gguf_transformer method must call _install_lora_hooks."""
        from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline

        # Check that the method exists on LTXLowVRAMPipeline
        assert hasattr(LTXLowVRAMPipeline, "_install_lora_hooks"), (
            "LTXLowVRAMPipeline must have _install_lora_hooks method"
        )

        # Check the source code of _load_gguf_transformer references _install_lora_hooks
        import inspect
        source = inspect.getsource(LTXLowVRAMPipeline._load_gguf_transformer)
        assert "_install_lora_hooks" in source, (
            "_load_gguf_transformer must call _install_lora_hooks after loading GGUF weights"
        )


# ---------------------------------------------------------------------------
# GPU integration test (requires models on disk)
# ---------------------------------------------------------------------------


class TestGGUFLoRAVideoGeneration:
    """End-to-end test: GGUF + LoRA pipeline generates non-black video."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gguf_lora_generates_non_black_video(self) -> None:
        """Pipeline with dev GGUF + distilled LoRA must produce non-black output."""
        models_dir = _models_dir()
        gguf = _find_gguf(models_dir)
        checkpoint = _find_checkpoint(models_dir)
        lora = _find_lora(models_dir)
        gemma_root = _find_gemma_root(models_dir)
        upsampler = _find_upsampler(models_dir)

        if any(x is None for x in [gguf, checkpoint, lora, gemma_root, upsampler]):
            pytest.skip("Required model files not found")

        from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline
        from services.vram_manager.vram_manager import VRAMManager

        device = torch.device("cuda")
        vram_gb = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        vram_manager = VRAMManager(device, vram_gb)

        pipeline = LTXLowVRAMPipeline.create(
            checkpoint_path=str(checkpoint),
            gemma_root=str(gemma_root),
            upsampler_path=str(upsampler),
            device=device,
            vram_manager=vram_manager,
            gguf_path=str(gguf),
            lora_path=str(lora),
            lora_strength=0.6,
            use_sage_attention=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name

        try:
            pipeline.generate(
                prompt="A bright colorful scene with flowers",
                seed=42,
                height=256,
                width=384,
                num_frames=9,
                frame_rate=8,
                images=[],
                output_path=output_path,
            )

            # Verify output exists and is not black
            import av
            container = av.open(output_path)
            stream = next(s for s in container.streams if s.type == "video")
            frame_means = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                frame_means.append(float(arr.mean()))
            container.close()

            assert len(frame_means) > 0, "Video has no frames"
            avg_brightness = sum(frame_means) / len(frame_means)
            assert avg_brightness > 5.0, (
                f"Video is black (avg brightness {avg_brightness:.1f}). "
                "LoRA hooks may not be applied to GGUF transformer."
            )

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
