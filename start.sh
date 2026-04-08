#!/usr/bin/env bash
set -euo pipefail

# ---------- 1. Start FastAPI backend ----------
echo "[start.sh] Starting FastAPI backend on :8000 ..."
cd /app
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# ---------- 2. Start Next.js frontend ----------
echo "[start.sh] Starting Next.js frontend on :3000 ..."
cd /app/frontend
PORT=3000 npx next start &
FRONTEND_PID=$!

# ---------- 3. Wait briefly for upstreams ----------
echo "[start.sh] Waiting for backend and frontend to initialize ..."
sleep 4

# ---------- 4. Start Nginx (foreground, keeps container alive) ----------
echo "[start.sh] Starting Nginx on :7860 ..."
nginx -g "daemon off;" -c /app/nginx.conf &
NGINX_PID=$!

echo "[start.sh] All services started. PIDs: backend=$BACKEND_PID frontend=$FRONTEND_PID nginx=$NGINX_PID"

# ---------- 5. Graceful shutdown on SIGTERM ----------
cleanup() {
    echo "[start.sh] Caught signal, shutting down ..."
    kill $NGINX_PID   2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    kill $BACKEND_PID  2>/dev/null || true
    wait
    echo "[start.sh] All processes stopped."
}
trap cleanup SIGTERM SIGINT

# Keep the script alive until any child exits
wait -n
echo "[start.sh] A child process exited, shutting down remaining ..."
cleanup
