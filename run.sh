#!/bin/bash
# run.sh — Launch LTX Desktop as a standalone web app (no Electron required)
#
# Usage:
#   ./run.sh                    # Start with default settings
#   ./run.sh --port 8080        # Custom port
#   ./run.sh --host 0.0.0.0    # Listen on all interfaces
#
# Requirements:
#   - Python 3.12+ with uv
#   - NVIDIA GPU with ≥8 GB VRAM (for local generation)
#   - Models downloaded to ~/.ltx-desktop/models/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

# Default configuration
export LTX_APP_DATA_DIR="${LTX_APP_DATA_DIR:-$HOME/.ltx-desktop}"
export LTX_WEB_MODE="${LTX_WEB_MODE:-1}"
export LTX_PORT="${LTX_PORT:-8000}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            export LTX_PORT="$2"
            shift 2
            ;;
        --host)
            export LTX_HOST="$2"
            shift 2
            ;;
        --no-sage)
            export USE_SAGE_ATTENTION="0"
            shift
            ;;
        --debug)
            export BACKEND_DEBUG="1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--host HOST] [--no-sage] [--debug]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "LTX Desktop — Web Mode"
echo "=========================================="
echo "Data directory: $LTX_APP_DATA_DIR"
echo "Port: $LTX_PORT"
echo ""

# Create data directories
mkdir -p "$LTX_APP_DATA_DIR/models"
mkdir -p "$LTX_APP_DATA_DIR/outputs"

# Check for uv
if command -v uv &> /dev/null; then
    echo "Starting backend with uv..."
    cd "$BACKEND_DIR"
    exec uv run python ltx2_server.py "$@"
else
    echo "Starting backend with python..."
    cd "$BACKEND_DIR"
    exec python ltx2_server.py "$@"
fi
