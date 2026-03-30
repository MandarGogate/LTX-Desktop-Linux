#!/usr/bin/env python3
"""Cross-platform launcher for LTX Desktop in web mode.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --host 0.0.0.0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch LTX Desktop as a web app")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--data-dir", type=str, default=None, help="App data directory")
    parser.add_argument("--no-sage", action="store_true", help="Disable SageAttention")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Set environment
    data_dir = args.data_dir or os.environ.get(
        "LTX_APP_DATA_DIR",
        str(Path.home() / ".ltx-desktop"),
    )
    os.environ["LTX_APP_DATA_DIR"] = data_dir
    os.environ["LTX_WEB_MODE"] = "1"
    os.environ["LTX_PORT"] = str(args.port)

    if args.no_sage:
        os.environ["USE_SAGE_ATTENTION"] = "0"
    if args.debug:
        os.environ["BACKEND_DEBUG"] = "1"

    # Create directories
    Path(data_dir, "models").mkdir(parents=True, exist_ok=True)
    Path(data_dir, "outputs").mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("LTX Desktop — Web Mode")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"URL: http://{args.host}:{args.port}")
    print()

    # Launch backend
    backend_dir = Path(__file__).parent / "backend"
    server_script = backend_dir / "ltx2_server.py"

    if not server_script.exists():
        print(f"Error: Backend script not found at {server_script}", file=sys.stderr)
        sys.exit(1)

    os.chdir(str(backend_dir))
    os.execv(sys.executable, [sys.executable, str(server_script)])


if __name__ == "__main__":
    main()
