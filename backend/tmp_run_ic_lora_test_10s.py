from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from ltx2_server import app, handler


def main() -> None:
    video_path = Path.home() / ".ltx-desktop/outputs/ltx2_video_20260403_110134_b77eea2f.mp4"
    assert video_path.exists(), f"Missing input video: {video_path}"

    payload = {
        "video_path": str(video_path),
        "model_type": "union",
        "conditioning_type": "pose",
        "conditioning_strength": 1.0,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "duration": 10,
        "prompt": "cinematic drone shot of a person walking through a futuristic city, natural motion, detailed lighting",
        "images": [],
    }

    print("Current run mode:", handler.state.app_settings.run_mode)
    print("Current num_blocks_to_swap:", handler.state.app_settings.num_blocks_to_swap)
    print("Input:", video_path)
    print("Payload:", json.dumps(payload, indent=2))

    started = time.perf_counter()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/ic-lora/generate", json=payload, timeout=None)
    elapsed = time.perf_counter() - started

    print("Status:", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
