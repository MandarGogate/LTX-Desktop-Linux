#!/usr/bin/env python3
"""Benchmark: old vs optimized GGUF/LoRA loading + generation for LTX Desktop.

Tests three resolutions (540p, 720p, 1080p) with:
  A) Original pipeline (dequantize-at-load GGUF + pre-fused LoRA)
  B) Optimized pipeline (lazy GGUF + fast block swap + StateDictRegistry)

Measures: model load time, LoRA load/fuse time, generation time, peak VRAM.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = Path(os.environ.get("LTX_MODELS_DIR", str(Path.home() / ".ltx-desktop" / "models")))

CHECKPOINT_PATH = str(MODELS_DIR / "ltx-2.3-22b-distilled.safetensors")
GGUF_PATH = str(MODELS_DIR / "gguf" / "ltx-2.3-22b-dev-Q8_0.gguf")
UPSAMPLER_PATH = str(MODELS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
GEMMA_ROOT = str(MODELS_DIR / "gemma-3-12b-it-qat-q4_0-unquantized")
DISTILLED_LORA_PATH = str(MODELS_DIR / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors")
TEXT_ENCODER_VARIANT = str(MODELS_DIR / "text_encoders" / "gemma_3_12B_it_fp4_mixed.safetensors")

DEVICE = torch.device("cuda")
PROMPT = "A cinematic shot of a golden retriever running through a sunlit meadow, slow motion, bokeh background, 4K quality"
SEED = 42
FRAME_RATE = 25.0
NUM_FRAMES = 33  # 8k+1

RESOLUTIONS = {
    "540p": (960, 544),
    "720p": (1280, 704),
    # "1080p": (1920, 1088),  # May OOM on 24GB - enabled conditionally
}

OUTPUT_DIR = BACKEND_DIR / "_benchmark_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class BenchmarkResult:
    method: str
    resolution: str
    width: int
    height: int
    num_frames: int
    gguf_load_time_s: float = 0.0
    lora_load_time_s: float = 0.0
    model_init_time_s: float = 0.0
    text_encode_time_s: float = 0.0
    denoise_time_s: float = 0.0
    vae_decode_time_s: float = 0.0
    total_generation_time_s: float = 0.0
    peak_vram_mb: int = 0
    error: str = ""


def get_vram_mb() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    return 0


def get_peak_vram_mb() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0


def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Method A: Original GGUF loader (dequantize at load time)
# ============================================================================

def benchmark_original_gguf_load() -> float:
    """Benchmark the original GGUF loading approach (full dequantization)."""
    from services.gguf_loader.gguf_loader import GGUFModelLoader

    cleanup_gpu()
    t0 = time.perf_counter()
    sd = GGUFModelLoader.load_gguf_sd_for_diffusers(Path(GGUF_PATH), device="cpu")
    elapsed = time.perf_counter() - t0
    n_tensors = len(sd)
    total_bytes = sum(t.nbytes for t in sd.values())
    logger.info(
        "Original GGUF load: %d tensors, %.1f GB, %.2fs",
        n_tensors,
        total_bytes / 1e9,
        elapsed,
    )
    del sd
    cleanup_gpu()
    return elapsed


# ============================================================================
# Method B: Optimized lazy GGUF loader (keep quantized)
# ============================================================================

def benchmark_lazy_gguf_load() -> float:
    """Benchmark the lazy GGUF loading approach (keep quantized)."""
    from services.gguf_loader.gguf_lazy_loader import load_gguf_lazy_state_dict

    cleanup_gpu()
    t0 = time.perf_counter()
    sd = load_gguf_lazy_state_dict(GGUF_PATH, device="cpu")
    elapsed = time.perf_counter() - t0
    n_tensors = len(sd)
    total_bytes = sum(t.nbytes for t in sd.values())
    logger.info(
        "Lazy GGUF load: %d tensors, %.1f GB (quantized), %.2fs",
        n_tensors,
        total_bytes / 1e9,
        elapsed,
    )
    del sd
    cleanup_gpu()
    return elapsed


# ============================================================================
# Method A: Original pipeline full generation
# ============================================================================

def benchmark_original_pipeline(
    width: int, height: int, num_frames: int, resolution_name: str
) -> BenchmarkResult:
    """Full generation using the original LTXLowVRAMPipeline."""
    result = BenchmarkResult(
        method="original",
        resolution=resolution_name,
        width=width,
        height=height,
        num_frames=num_frames,
    )

    try:
        cleanup_gpu()
        reset_vram_stats()

        from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline
        from services.vram_manager.vram_manager import VRAMManager

        vram_gb = int(torch.cuda.get_device_properties(0).total_memory // (1024**3))
        vram_manager = VRAMManager(DEVICE, vram_gb)

        t_init = time.perf_counter()
        pipeline = LTXLowVRAMPipeline.create(
            checkpoint_path=CHECKPOINT_PATH,
            gemma_root=GEMMA_ROOT,
            upsampler_path=UPSAMPLER_PATH,
            device=DEVICE,
            vram_manager=vram_manager,
            gguf_path=GGUF_PATH,
            lora_path=DISTILLED_LORA_PATH,
            lora_strength=0.6,
            use_sage_attention=True,
            text_encoder_variant_path=TEXT_ENCODER_VARIANT,
        )
        result.model_init_time_s = time.perf_counter() - t_init

        output_path = str(OUTPUT_DIR / f"original_{resolution_name}.mp4")

        t_gen = time.perf_counter()
        pipeline.generate(
            prompt=PROMPT,
            seed=SEED,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=FRAME_RATE,
            images=[],
            output_path=output_path,
        )
        result.total_generation_time_s = time.perf_counter() - t_gen
        result.peak_vram_mb = get_peak_vram_mb()

        logger.info(
            "Original %s: init=%.1fs gen=%.1fs peak_vram=%dMB",
            resolution_name,
            result.model_init_time_s,
            result.total_generation_time_s,
            result.peak_vram_mb,
        )

        del pipeline
        cleanup_gpu()

    except Exception as e:
        result.error = str(e)
        logger.error("Original %s failed: %s", resolution_name, e, exc_info=True)
        cleanup_gpu()

    return result


# ============================================================================
# Method B: Optimized pipeline full generation
# ============================================================================

def benchmark_optimized_pipeline(
    width: int, height: int, num_frames: int, resolution_name: str
) -> BenchmarkResult:
    """Full generation using optimized lazy GGUF + fast block swap + registry."""
    result = BenchmarkResult(
        method="optimized",
        resolution=resolution_name,
        width=width,
        height=height,
        num_frames=num_frames,
    )

    try:
        cleanup_gpu()
        reset_vram_stats()

        from services.fast_video_pipeline.ltx_optimized_pipeline import (
            LTXOptimizedPipeline,
        )
        from services.vram_manager.vram_manager import VRAMManager

        vram_gb = int(torch.cuda.get_device_properties(0).total_memory // (1024**3))
        vram_manager = VRAMManager(DEVICE, vram_gb)

        t_init = time.perf_counter()
        pipeline = LTXOptimizedPipeline.create(
            checkpoint_path=CHECKPOINT_PATH,
            gemma_root=GEMMA_ROOT,
            upsampler_path=UPSAMPLER_PATH,
            device=DEVICE,
            vram_manager=vram_manager,
            gguf_path=GGUF_PATH,
            lora_path=DISTILLED_LORA_PATH,
            lora_strength=0.6,
            use_sage_attention=True,
            text_encoder_variant_path=TEXT_ENCODER_VARIANT,
        )
        result.model_init_time_s = time.perf_counter() - t_init

        output_path = str(OUTPUT_DIR / f"optimized_{resolution_name}.mp4")

        t_gen = time.perf_counter()
        pipeline.generate(
            prompt=PROMPT,
            seed=SEED,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=FRAME_RATE,
            images=[],
            output_path=output_path,
        )
        result.total_generation_time_s = time.perf_counter() - t_gen
        result.peak_vram_mb = get_peak_vram_mb()

        logger.info(
            "Optimized %s: init=%.1fs gen=%.1fs peak_vram=%dMB",
            resolution_name,
            result.model_init_time_s,
            result.total_generation_time_s,
            result.peak_vram_mb,
        )

        del pipeline
        cleanup_gpu()

    except Exception as e:
        result.error = str(e)
        logger.error("Optimized %s failed: %s", resolution_name, e, exc_info=True)
        cleanup_gpu()

    return result


# ============================================================================
# GGUF load-only benchmark
# ============================================================================

def run_load_benchmarks() -> list[dict[str, object]]:
    """Compare GGUF load times."""
    results: list[dict[str, object]] = []

    if not Path(GGUF_PATH).exists():
        logger.warning("GGUF file not found: %s — skipping load benchmarks", GGUF_PATH)
        return results

    logger.info("=" * 60)
    logger.info("GGUF LOAD BENCHMARK")
    logger.info("=" * 60)

    # Warm filesystem cache
    logger.info("Warming filesystem cache...")
    with open(GGUF_PATH, "rb") as f:
        _ = f.read(1024 * 1024)

    # Original
    logger.info("--- Original GGUF loader (dequantize to bf16) ---")
    t_orig = benchmark_original_gguf_load()
    results.append({"method": "original_gguf_load", "time_s": t_orig})

    # Optimized
    logger.info("--- Lazy GGUF loader (keep quantized) ---")
    t_lazy = benchmark_lazy_gguf_load()
    results.append({"method": "lazy_gguf_load", "time_s": t_lazy})

    speedup = t_orig / t_lazy if t_lazy > 0 else float("inf")
    logger.info("GGUF load speedup: %.1fx (%.1fs → %.1fs)", speedup, t_orig, t_lazy)

    return results


# ============================================================================
# Full generation benchmark
# ============================================================================

def run_generation_benchmarks() -> list[BenchmarkResult]:
    """Run full generation benchmarks at each resolution."""
    results: list[BenchmarkResult] = []

    vram_gb = int(torch.cuda.get_device_properties(0).total_memory // (1024**3))

    # Enable 1080p only on ≥24GB GPUs
    resolutions = dict(RESOLUTIONS)
    if vram_gb >= 23:
        resolutions["1080p"] = (1920, 1088)

    for res_name, (w, h) in resolutions.items():
        logger.info("=" * 60)
        logger.info("GENERATION BENCHMARK: %s (%dx%d, %d frames)", res_name, w, h, NUM_FRAMES)
        logger.info("=" * 60)

        # Optimized first (faster, validates the code)
        logger.info("--- Optimized pipeline ---")
        r_opt = benchmark_optimized_pipeline(w, h, NUM_FRAMES, res_name)
        results.append(r_opt)

        # Original
        logger.info("--- Original pipeline ---")
        r_orig = benchmark_original_pipeline(w, h, NUM_FRAMES, res_name)
        results.append(r_orig)

        # Compare
        if not r_orig.error and not r_opt.error:
            speedup_init = (
                r_orig.model_init_time_s / r_opt.model_init_time_s
                if r_opt.model_init_time_s > 0
                else float("inf")
            )
            speedup_gen = (
                r_orig.total_generation_time_s / r_opt.total_generation_time_s
                if r_opt.total_generation_time_s > 0
                else float("inf")
            )
            logger.info(
                "%s speedup: init=%.1fx gen=%.1fx vram_saved=%dMB",
                res_name,
                speedup_init,
                speedup_gen,
                r_orig.peak_vram_mb - r_opt.peak_vram_mb,
            )

    return results


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    logger.info("LTX Desktop Loading & Generation Benchmark")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("Models dir: %s", MODELS_DIR)
    logger.info("")

    # Verify files exist
    for label, path in [
        ("Checkpoint", CHECKPOINT_PATH),
        ("GGUF", GGUF_PATH),
        ("Upsampler", UPSAMPLER_PATH),
        ("Gemma root", GEMMA_ROOT),
        ("Distilled LoRA", DISTILLED_LORA_PATH),
    ]:
        exists = Path(path).exists()
        size = ""
        if exists and Path(path).is_file():
            size = f" ({Path(path).stat().st_size / 1e9:.1f} GB)"
        logger.info("  %s: %s%s %s", label, path, size, "✓" if exists else "✗ MISSING")

    logger.info("")

    all_results: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }

    # Load benchmarks
    load_results = run_load_benchmarks()
    all_results["load_benchmarks"] = load_results

    # Generation benchmarks
    gen_results = run_generation_benchmarks()
    all_results["generation_benchmarks"] = [asdict(r) for r in gen_results]

    # Save results
    results_path = OUTPUT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-12s %-10s %10s %10s %10s %10s",
        "Method", "Resolution", "Init(s)", "Gen(s)", "Total(s)", "VRAM(MB)",
    )
    logger.info("-" * 80)
    for r in gen_results:
        total = r.model_init_time_s + r.total_generation_time_s
        logger.info(
            "%-12s %-10s %10.1f %10.1f %10.1f %10d %s",
            r.method,
            r.resolution,
            r.model_init_time_s,
            r.total_generation_time_s,
            total,
            r.peak_vram_mb,
            f"ERROR: {r.error}" if r.error else "",
        )


if __name__ == "__main__":
    main()
