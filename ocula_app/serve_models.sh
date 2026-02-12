#!/bin/bash
# Serves GGUF model files for the Ocula app to download.
# The app expects models at http://localhost:8080/models/<filename>.gguf
#
# Layout:
#   assets/models/          — SmolVLM2 (free tier, bundled in APK)
#   assets/models_server/   — Moondream 2 + Qwen3 (Plus/Pro, download-only)
#
# The server merges both into a single /models/ endpoint by symlinking
# the server-only files into assets/models/ at runtime.
#
# Usage:
#   1. Run fetch_ocula_stack.sh first to download models
#   2. Run this script: bash serve_models.sh
#   3. The Flutter app will download models from this server on first launch.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVE_DIR="$SCRIPT_DIR/assets"
MODELS_DIR="$SERVE_DIR/models"
SERVER_DIR="$SERVE_DIR/models_server"

if [ ! -d "$MODELS_DIR" ]; then
  echo "Error: $MODELS_DIR not found."
  exit 1
fi

# Symlink server-only models into the served models/ directory
if [ -d "$SERVER_DIR" ]; then
  for f in "$SERVER_DIR"/*.gguf; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    if [ ! -e "$MODELS_DIR/$base" ]; then
      ln -sf "../models_server/$base" "$MODELS_DIR/$base"
      echo "Linked: $base"
    fi
  done
fi

echo ""
echo "Serving models at http://localhost:8080/models/"
echo "Models directory: $MODELS_DIR"
echo ""
ls -lh "$MODELS_DIR/"*.gguf 2>/dev/null || echo "(no .gguf files found)"
echo ""
echo "Press Ctrl+C to stop."

cd "$SERVE_DIR" && python3 -m http.server 8080
