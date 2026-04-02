#!/usr/bin/env python3
"""Benchmark: Q4 GGUF vs Q8 GGUF vs Full safetensors (bf16) at 540p, 720p, 1080p.

Each model variant is run through the optimized pipeline with prompt caching
to measure cold-start and warm (cached) generation times.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for ns in ("services.fast_video_pipeline", "services.gguf_loader", "services.block_swap"):
    logging.getLogger(ns).setLevel(logging.INFO)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

# ───── paths ─────
M = Path(os.environ.get("LTX_MODELS_DIR", str(Path.home() / ".ltx-desktop" / "models")))
CHECKPOINT = str(M / "ltx-2.3-22b-distilled.safetensors")
Q8_GGUF    = str(M / "gguf" / "distilled" / "ltx-2.3-22b-distilled-Q8_0.gguf")
Q4_GGUF    = str(M / "gguf" / "distilled" / "ltx-2.3-22b-distilled-Q4_K_M.gguf")
UPSAMPLER  = str(M / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
GEMMA      = str(M / "gemma-3-12b-it-qat-q4_0-unquantized")
LORA       = str(M / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors")
TE_VARIANT = str(M / "text_encoders" / "gemma_3_12B_it_fp4_mixed.safetensors")

PROMPT = "A cinematic shot of a golden retriever running through a sunlit meadow, slow motion, bokeh"
DEVICE = torch.device("cuda")
SEED   = 42
FPS    = 25.0
FRAMES = 33

RESOLUTIONS = [
    ("540p",  960,  544),
    ("720p",  1280, 704),
]

# Only add 1080p if ≥ 23 GB VRAM
vram_gb = int(torch.cuda.get_device_properties(0).total_memory // (1024**3))
if vram_gb >= 23:
    RESOLUTIONS.append(("1080p", 1920, 1088))

# ───── model configs ─────
MODEL_CONFIGS: list[dict[str, object]] = [
    {
        "name": "Q4_K_M GGUF (distilled, 14GB)",
        "gguf_path": Q4_GGUF,
        "checkpoint": CHECKPOINT,
        "lora": None,            # distilled model, no extra LoRA needed
        "steps": None,           # use distilled sigma schedule (8 steps)
    },
    {
        "name": "Q8_0 GGUF (distilled, 22GB)",
        "gguf_path": Q8_GGUF,
        "checkpoint": CHECKPOINT,
        "lora": None,
        "steps": None,
    },
    {
        "name": "Full bf16 safetensors (distilled, 46GB)",
        "gguf_path": None,
        "checkpoint": CHECKPOINT,
        "lora": None,
        "steps": None,
    },
]

OUT = BACKEND / "_bench_quant"
OUT.mkdir(exist_ok=True)


@dataclass
class Result:
    model: str
    resolution: str
    width: int
    height: int
    run: str  # "cold" or "warm"
    init_s: float = 0.0
    text_enc_s: float = 0.0
    denoise_s: float = 0.0
    vae_s: float = 0.0
    total_s: float = 0.0
    peak_vram_mb: int = 0
    error: str = ""


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_one(
    config: dict[str, object],
    res_name: str,
    w: int,
    h: int,
    run_label: str,
    pipeline: object | None = None,
) -> tuple[Result, object]:
    """Run a single generation, returning (result, pipeline)."""
    result = Result(
        model=str(config["name"]),
        resolution=res_name,
        width=w,
        height=h,
        run=run_label,
    )
    try:
        cleanup()
        torch.cuda.reset_peak_memory_stats()

        from services.fast_video_pipeline.ltx_optimized_pipeline import LTXOptimizedPipeline
        from services.vram_manager.vram_manager import VRAMManager

        vm = VRAMManager(DEVICE, vram_gb)

        if pipeline is None:
            t_init = time.perf_counter()
            gguf = config.get("gguf_path")
            lora = config.get("lora")
            pipeline = LTXOptimizedPipeline.create(
                checkpoint_path=str(config["checkpoint"]),
                gemma_root=GEMMA,
                upsampler_path=UPSAMPLER,
                device=DEVICE,
                vram_manager=vm,
                gguf_path=str(gguf) if gguf else None,
                lora_path=str(lora) if lora else None,
                lora_strength=0.6 if lora else 1.0,
                use_sage_attention=True,
                text_encoder_variant_path=None,  # skip variant for clean comparison
                num_inference_steps=config.get("steps") if config.get("steps") else None,  # type: ignore[arg-type]
            )
            result.init_s = time.perf_counter() - t_init

        out_path = str(OUT / f"{config['name'][:10]}_{res_name}_{run_label}.mp4")
        t_gen = time.perf_counter()
        pipeline.generate(  # type: ignore[union-attr]
            prompt=PROMPT,
            seed=SEED if run_label == "cold" else SEED + 1,
            height=h,
            width=w,
            num_frames=FRAMES,
            frame_rate=FPS,
            images=[],
            output_path=out_path,
        )
        result.total_s = time.perf_counter() - t_gen
        result.peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

    except Exception as e:
        result.error = str(e)[:300]
        logger.error("%s %s %s: %s", config["name"], res_name, run_label, e, exc_info=True)

    return result, pipeline


def main() -> None:
    logger.info("=" * 70)
    logger.info("QUANTIZATION BENCHMARK: Q4 vs Q8 vs Full (bf16)")
    logger.info("GPU: %s  VRAM: %d GB", torch.cuda.get_device_name(0), vram_gb)
    logger.info("Resolutions: %s  Frames: %d", [r[0] for r in RESOLUTIONS], FRAMES)
    logger.info("=" * 70)

    # Verify files
    for label, path in [
        ("Checkpoint", CHECKPOINT), ("Q8 GGUF", Q8_GGUF), ("Q4 GGUF", Q4_GGUF),
        ("Upsampler", UPSAMPLER), ("Gemma", GEMMA),
    ]:
        p = Path(path)
        sz = f" ({p.stat().st_size / 1e9:.1f}GB)" if p.is_file() else ""
        logger.info("  %s: %s%s %s", label, path, sz, "✓" if p.exists() else "✗")

    results: list[Result] = []

    for config in MODEL_CONFIGS:
        name = config["name"]
        gguf_path = config.get("gguf_path")

        # Skip if file missing
        if gguf_path and not Path(str(gguf_path)).exists():
            logger.warning("SKIP %s — file not found", name)
            continue
        if not gguf_path and not Path(str(config["checkpoint"])).exists():
            logger.warning("SKIP %s — checkpoint not found", name)
            continue

        logger.info("")
        logger.info("━" * 50)
        logger.info("MODEL: %s", name)
        logger.info("━" * 50)

        for res_name, w, h in RESOLUTIONS:
            pipeline = None

            # Cold run (includes init + first text encode)
            logger.info("  %s cold...", res_name)
            r_cold, pipeline = run_one(config, res_name, w, h, "cold", pipeline=None)
            results.append(r_cold)
            if r_cold.error:
                logger.error("  %s cold FAILED: %s", res_name, r_cold.error[:80])
                cleanup()
                continue

            logger.info(
                "  %s cold: init=%.1fs gen=%.1fs peak=%dMB",
                res_name, r_cold.init_s, r_cold.total_s, r_cold.peak_vram_mb,
            )

            # Warm run (same prompt → cached; models in CPU RAM)
            logger.info("  %s warm...", res_name)
            r_warm, pipeline = run_one(config, res_name, w, h, "warm", pipeline=pipeline)
            results.append(r_warm)

            if not r_warm.error:
                logger.info(
                    "  %s warm: gen=%.1fs peak=%dMB (saved %.1fs)",
                    res_name, r_warm.total_s, r_warm.peak_vram_mb,
                    r_cold.total_s - r_warm.total_s,
                )

            # Cleanup pipeline between model configs
            del pipeline
            cleanup()

    # ─── Summary table ───
    logger.info("")
    logger.info("=" * 100)
    logger.info("RESULTS")
    logger.info("=" * 100)
    logger.info(
        "%-35s %-6s %-5s %8s %8s %8s %s",
        "Model", "Res", "Run", "Init", "Gen", "Total", "VRAM",
    )
    logger.info("-" * 100)
    for r in results:
        total = r.init_s + r.total_s
        err = f"  ERR: {r.error[:40]}" if r.error else ""
        logger.info(
            "%-35s %-6s %-5s %7.1fs %7.1fs %7.1fs %6dMB%s",
            r.model[:35], r.resolution, r.run,
            r.init_s, r.total_s, total, r.peak_vram_mb, err,
        )

    # ─── Comparison per resolution ───
    logger.info("")
    logger.info("COMPARISON (warm runs, prompt-cached):")
    for res_name, _, _ in RESOLUTIONS:
        warm = [r for r in results if r.resolution == res_name and r.run == "warm" and not r.error]
        if len(warm) < 2:
            continue
        fastest = min(warm, key=lambda r: r.total_s)
        logger.info("  %s:", res_name)
        for r in warm:
            marker = " ★ FASTEST" if r is fastest else ""
            logger.info(
                "    %-35s  %7.1fs  %5dMB%s",
                r.model[:35], r.total_s, r.peak_vram_mb, marker,
            )

    # Save JSON
    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("\nJSON saved to %s", out_path)


if __name__ == "__main__":
    main()
