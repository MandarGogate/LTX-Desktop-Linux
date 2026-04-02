#!/bin/bash

set -euo pipefail

resolve_backend_host() {
    if [ -n "${BACKEND_HOST:-}" ]; then
        printf '%s\n' "$BACKEND_HOST"
    elif [ -n "${1:-}" ]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "127.0.0.1"
    fi
}

resolve_backend_url() {
    local host="$1"
    local port="$2"
    if [ -n "${BACKEND_URL:-}" ]; then
        printf '%s\n' "${BACKEND_URL%/}"
    else
        printf 'http://%s:%s\n' "$host" "$port"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Restarting LTX Desktop Web App..."

# Start with same args as run-web.sh
HOST="$(resolve_backend_host "${1:-}")"
PORT="${2:-${BACKEND_PORT:-8001}}"
BACKEND_URL="$(resolve_backend_url "$HOST" "$PORT")"
APP_DATA_DIR="${APP_DATA_DIR:-$HOME/.ltx-desktop}"
XDG_APP_STATE_DIR="${XDG_APP_STATE_DIR:-$HOME/.local/share/LTXDesktop}"
MODELS_DIR="$APP_DATA_DIR/models"
LOGS_DIR="$APP_DATA_DIR/logs"
DISTILLED_LORA_PATH="$MODELS_DIR/loras/ltx-2.3-22b-distilled-lora-384.safetensors"
LTX_A2V_DEBUG_BLOCK_SWAP="${LTX_A2V_DEBUG_BLOCK_SWAP:-0}"
LTX_A2V_DEBUG_TENSORS="${LTX_A2V_DEBUG_TENSORS:-0}"
BACKEND_LOG_FILE="$LOGS_DIR/backend-web.log"

# Stop existing processes (pass port for fuser)
"$SCRIPT_DIR/stop-web.sh" "$PORT"

sleep 3

# Normalize settings so restarts consistently use the existing local models.
mkdir -p "$MODELS_DIR" "$APP_DATA_DIR/outputs" "$XDG_APP_STATE_DIR" "$LOGS_DIR"

export APP_DATA_DIR
export XDG_APP_STATE_DIR

python3 - <<'PY'
import json
import os
from pathlib import Path

app_data_dir = Path(os.environ["APP_DATA_DIR"])
xdg_app_state_dir = Path(os.environ["XDG_APP_STATE_DIR"])
models_dir = app_data_dir / "models"
distilled_lora_path = models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"

settings_paths = [
    app_data_dir / "settings.json",
    xdg_app_state_dir / "settings.json",
]

for settings_path in settings_paths:
    if settings_path.exists():
        data = json.loads(settings_path.read_text())
    else:
        data = {}
    data["models_dir"] = str(models_dir)
    data["preferred_zit_model_path"] = ""
    if distilled_lora_path.exists():
        data["preferred_lora_path"] = str(distilled_lora_path)
        data["preferred_lora_strength"] = 0.6
    settings_path.write_text(json.dumps(data, indent=2) + "\n")
PY

# Start backend
cd "$SCRIPT_DIR/backend"
echo "Starting backend with logs at: $BACKEND_LOG_FILE"
: > "$BACKEND_LOG_FILE"
LTX_APP_DATA_DIR="$APP_DATA_DIR" \
LTX_PORT="$PORT" \
BACKEND_HOST="$HOST" \
CORS_ORIGINS="*" \
LTX_A2V_DEBUG_BLOCK_SWAP="$LTX_A2V_DEBUG_BLOCK_SWAP" \
LTX_A2V_DEBUG_TENSORS="$LTX_A2V_DEBUG_TENSORS" \
uv run python ltx2_server.py 2>&1 | tee -a "$BACKEND_LOG_FILE" &
BACKEND_PID=$!

sleep 10

# Start frontend
cd "$SCRIPT_DIR"
BACKEND_URL="$BACKEND_URL" \
CORS_ORIGINS="*" \
WEB_MODE=true \
VITE_HOST=0.0.0.0 \
npx vite --host &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "LTX Desktop restarted!"
echo "Frontend: http://${HOST}:5173"
echo "Backend:  ${BACKEND_URL}"
echo "Backend log: ${BACKEND_LOG_FILE}"
echo "A2V debug block swap: ${LTX_A2V_DEBUG_BLOCK_SWAP}"
echo "A2V debug tensors:    ${LTX_A2V_DEBUG_TENSORS}"
echo "Tip: tail -f ${BACKEND_LOG_FILE}"
echo "=========================================="

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

wait
