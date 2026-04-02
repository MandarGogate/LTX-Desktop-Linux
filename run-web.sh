#!/bin/bash

set -e

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
cd "$SCRIPT_DIR"

export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

HOST="$(resolve_backend_host "${1:-}")"
PORT="${2:-${BACKEND_PORT:-8001}}"
BACKEND_URL="$(resolve_backend_url "$HOST" "$PORT")"
APP_DATA_DIR="${APP_DATA_DIR:-$HOME/.ltx-desktop}"

echo "Starting LTX Desktop Web App"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Backend URL: $BACKEND_URL"
echo "App Data Dir: $APP_DATA_DIR"
echo ""

echo "=== Starting Backend ==="
cd "$SCRIPT_DIR/backend"
LTX_APP_DATA_DIR="$APP_DATA_DIR" \
LTX_PORT="$PORT" \
BACKEND_HOST="$HOST" \
CORS_ORIGINS="*" \
uv run python ltx2_server.py &
BACKEND_PID=$!

sleep 10

echo ""
echo "=== Starting Frontend ==="
cd "$SCRIPT_DIR"
BACKEND_URL="$BACKEND_URL" \
CORS_ORIGINS="*" \
WEB_MODE=true \
VITE_HOST=0.0.0.0 \
npx vite --host &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "LTX Desktop is running!"
echo "Frontend: http://${HOST}:5173"
echo "Backend:  http://${HOST}:${PORT}"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

wait
