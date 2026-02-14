#!/bin/bash
set -euo pipefail

# Entrypoint for Fly.io: starts Helix-DB, waits for it, then runs MCP server.
# Designed to run under tini (PID 1) for proper signal forwarding.

echo "[entrypoint] Starting Helix-DB..."
HELIX_CORES_OVERRIDE="${HELIX_CORES_OVERRIDE:-2}" \
TOKIO_WORKER_THREADS="${TOKIO_WORKER_THREADS:-2}" \
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}" \
/usr/local/bin/helix-container &
HELIX_PID=$!

# Wait for Helix to accept connections (up to 30s)
echo "[entrypoint] Waiting for Helix-DB on port 6969..."
for i in $(seq 1 30); do
    if nc -z 127.0.0.1 6969 2>/dev/null; then
        echo "[entrypoint] Helix-DB ready (${i}s)"
        break
    fi
    if ! kill -0 "$HELIX_PID" 2>/dev/null; then
        echo "[entrypoint] ERROR: Helix-DB exited prematurely"
        exit 1
    fi
    sleep 1
done

# Final check
if ! nc -z 127.0.0.1 6969 2>/dev/null; then
    echo "[entrypoint] ERROR: Helix-DB did not start within 30s"
    kill "$HELIX_PID" 2>/dev/null || true
    exit 1
fi

# Start MCP server in foreground (exec replaces shell, tini forwards signals)
echo "[entrypoint] Starting MCP server..."
exec python -m src.mcp_server
