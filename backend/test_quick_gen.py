#!/usr/bin/env python3
"""Quick single-generation test to verify prompt following."""

import json
import logging
import os
import sys
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_quick_gen")

MODELS_DIR = os.path.expanduser("~/.ltx-desktop/models")


def _resolve_backend_url() -> str:
    explicit = os.environ.get("BACKEND_URL")
    if explicit:
        return explicit.rstrip("/")

    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = os.environ.get("BACKEND_PORT", "8001")
    return f"http://{host}:{port}"


BACKEND_URL = _resolve_backend_url()


def wait_for_backend(timeout: int = 180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                logger.info("Backend ready: %s", data)
                return True
        except Exception:
            pass
        time.sleep(3)
    logger.error("Backend did not start within %ds", timeout)
    return False


def update_settings(patch: dict) -> None:
    r = requests.post(f"{BACKEND_URL}/api/settings", json=patch, timeout=10)
    r.raise_for_status()
    logger.info("Settings updated: %s", json.dumps(patch, indent=2))


def generate_video(prompt: str, model: str = "custom") -> str | None:
    logger.info("=== Generating: model=%s prompt='%s' ===", model, prompt[:80])
    r = requests.post(
        f"{BACKEND_URL}/api/generate",
        json={
            "prompt": prompt,
            "model": model,
            "duration": "2",
            "resolution": "540p",
            "fps": "24",
            "audio": "false",
            "cameraMotion": "none",
            "aspectRatio": "16:9",
        },
        timeout=600,
    )
    if r.status_code != 200:
        logger.error("Generation failed: %s %s", r.status_code, r.text[:500])
        return None
    result = r.json()
    if result.get("status") == "complete" and result.get("video_path"):
        path = result["video_path"]
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        logger.info("✓ Video generated: %s (%.1f MB)", path, size_mb)
        return path
    logger.error("Unexpected result: %s", result)
    return None


def run_test():
    if not wait_for_backend():
        return False

    prompt = "A golden retriever running through a field of sunflowers on a sunny day"

    # Test with distilled GGUF
    distilled_gguf = os.path.join(MODELS_DIR, "diffusion_models", "ltx-2.3-22b-distilled-Q4_0.gguf")
    logger.info("\n========== TEST: Distilled GGUF ==========")
    update_settings({
        "preferredModelPath": distilled_gguf,
        "preferredGgufPath": "",
        "selectedLoras": [],
    })
    path = generate_video(prompt, model="custom")
    if path and os.path.exists(path):
        size = os.path.getsize(path)
        logger.info("✓ PASSED: Video size = %d bytes", size)
        return True
    else:
        logger.error("✗ FAILED")
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
