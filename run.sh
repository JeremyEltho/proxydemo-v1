#!/usr/bin/env bash
# Start the router locally against Ollama. Creates the venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${ROUTER_PORT:-8000}"

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements-dev.txt
fi

if ! curl -sf -m 3 "${OLLAMA_URL:-http://127.0.0.1:11434}/api/tags" >/dev/null; then
  echo "!! Ollama is not reachable. Start it with: ollama serve"
  echo "!! Then pull a small model:               ollama pull qwen2.5:3b"
  echo "!! Starting anyway in routing-only mode."
fi

echo "==> http://localhost:${PORT}   (UI)  ·  /docs (OpenAPI)"
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" "$@"
