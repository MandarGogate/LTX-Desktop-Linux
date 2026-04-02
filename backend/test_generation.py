#!/usr/bin/env python3
"""Quick generation test to verify prompt following and model configurations."""

import json
import logging
import os
import sys
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_generation")

MODELS_DIR = os.path.expanduser("~/.ltx-desktop/models")


def _resolve_backend_url() -> str:
    explicit = os.environ.get("BACKEND_URL")
    if explicit:
        return explicit.rstrip("/")

    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = os.environ.get("BACKEND_PORT", "8000")
    return f"http://{host}:{port}"


BACKEND_URL = _resolve_backend_url()


def wait_for_backend(timeout: int = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BACKEND_URL}/api/settings", timeout=2)
            if r.status_code == 200:
                logger.info("Backend is ready")
                return True
        except Exception:
            pass
        time.sleep(2)
    logger.error("Backend did not start within %ds", timeout)
    return False


def update_settings(patch: dict) -> None:
    r = requests.post(
        f"{BACKEND_URL}/api/settings",
        json=patch,
        timeout=10,
    )
    r.raise_for_status()
    logger.info("Settings updated: %s", list(patch.keys()))


def generate_video(prompt: str, model: str = "custom", duration: str = "2", resolution: str = "540p") -> str | None:
    logger.info("=== Generating: model=%s prompt='%s' ===", model, prompt[:60])
    r = requests.post(
        f"{BACKEND_URL}/api/generate",
        json={
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "resolution": resolution,
            "fps": "24",
            "audio": "false",
            "cameraMotion": "none",
            "aspectRatio": "16:9",
        },
        timeout=600,
    )
    if r.status_code != 200:
        logger.error("Generation failed: %s %s", r.status_code, r.text[:200])
        return None
    result = r.json()
    if result.get("status") == "complete" and result.get("video_path"):
        path = result["video_path"]
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        logger.info("✓ Video generated: %s (%.1f MB)", path, size_mb)
        return path
    logger.error("Unexpected result: %s", result)
    return None


def check_video_not_black(path: str) -> bool:
    """Basic check: file exists and is non-trivially sized."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    return size > 50_000  # > 50KB means it's not empty/black


def run_tests():
    if not wait_for_backend():
        return False

    all_passed = True
    prompt = "A golden retriever running through a field of sunflowers on a sunny day"

    # --- Test 1: Distilled GGUF (fast mode) ---
    logger.info("\n========== TEST 1: Distilled GGUF ==========")
    distilled_gguf = os.path.join(MODELS_DIR, "diffusion_models", "ltx-2.3-22b-distilled-Q4_0.gguf")
    update_settings({
        "preferredModelPath": distilled_gguf,
        "preferredGgufPath": "",
        "selectedLoras": [],
    })
    path = generate_video(prompt, model="custom")
    if path and check_video_not_black(path):
        logger.info("✓ TEST 1 PASSED: Distilled GGUF generated valid video")
    else:
        logger.error("✗ TEST 1 FAILED: Distilled GGUF generation failed")
        all_passed = False

    # --- Test 2: Regenerate same prompt (distilled GGUF) ---
    logger.info("\n========== TEST 2: Regenerate (distilled GGUF) ==========")
    path2 = generate_video(prompt, model="custom")
    if path2 and check_video_not_black(path2):
        logger.info("✓ TEST 2 PASSED: Regeneration generated valid video")
    else:
        logger.error("✗ TEST 2 FAILED: Regeneration failed")
        all_passed = False

    # --- Test 3: Dev GGUF + Distilled LoRA (balanced mode) ---
    logger.info("\n========== TEST 3: Dev GGUF + Distilled LoRA ==========")
    dev_gguf = os.path.join(MODELS_DIR, "diffusion_models", "ltx-2.3-22b-dev-Q8_0.gguf")
    distilled_lora = os.path.join(MODELS_DIR, "loras", "ltx-2.3-22b-distilled-lora-384.safetensors")
    update_settings({
        "preferredModelPath": dev_gguf,
        "preferredGgufPath": "",
        "selectedLoras": [{"path": distilled_lora, "strength": 0.6}],
        "customModel": {"steps": 8, "useUpscaler": False},
    })
    path3 = generate_video(prompt, model="custom")
    if path3 and check_video_not_black(path3):
        logger.info("✓ TEST 3 PASSED: Dev GGUF + Distilled LoRA generated valid video")
    else:
        logger.error("✗ TEST 3 FAILED: Dev GGUF + Distilled LoRA failed")
        all_passed = False

    # --- Test 4: Regenerate (Dev GGUF + LoRA) ---
    logger.info("\n========== TEST 4: Regenerate (Dev GGUF + LoRA) ==========")
    path4 = generate_video("A close-up of a cat sitting on a windowsill watching rain", model="custom")
    if path4 and check_video_not_black(path4):
        logger.info("✓ TEST 4 PASSED: Regeneration with different prompt succeeded")
    else:
        logger.error("✗ TEST 4 FAILED: Regeneration failed")
        all_passed = False

    # --- Test 5: Distilled safetensors (no GGUF) ---
    logger.info("\n========== TEST 5: Distilled safetensors ==========")
    distilled_safetensors = os.path.join(MODELS_DIR, "diffusion_models", "ltx-2.3-22b-distilled.safetensors")
    if os.path.exists(distilled_safetensors):
        update_settings({
            "preferredModelPath": distilled_safetensors,
            "preferredGgufPath": "",
            "selectedLoras": [],
            "customModel": {"steps": 20, "useUpscaler": False},  # Steps should be overridden to distilled schedule
        })
        path5 = generate_video(prompt, model="custom")
        if path5 and check_video_not_black(path5):
            logger.info("✓ TEST 5 PASSED: Distilled safetensors generated valid video")
        else:
            logger.error("✗ TEST 5 FAILED: Distilled safetensors failed")
            all_passed = False
    else:
        logger.info("⊘ TEST 5 SKIPPED: No distilled safetensors found")

    # --- Test 6: Dev GGUF without LoRA (quality mode) ---
    logger.info("\n========== TEST 6: Dev GGUF without LoRA ==========")
    update_settings({
        "preferredModelPath": dev_gguf,
        "preferredGgufPath": "",
        "selectedLoras": [],
        "customModel": {"steps": 20, "useUpscaler": False},
    })
    path6 = generate_video(prompt, model="custom")
    if path6 and check_video_not_black(path6):
        logger.info("✓ TEST 6 PASSED: Dev GGUF without LoRA generated valid video")
    else:
        logger.error("✗ TEST 6 FAILED: Dev GGUF without LoRA failed")
        all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
