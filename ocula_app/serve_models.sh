#!/bin/bash
# Serves GGUF model files for the Ocula app to download.
# The app expects models at http://localhost:8080/models/<filename>.gguf
#
# Usage:
#   1. Run fetch_ocula_stack.sh first to download models into assets/models/
#   2. Run this script: bash serve_models.sh
#   3. The Flutter app will download models from this server on first launch.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVE_DIR="$SCRIPT_DIR/assets"

if [ ! -d "$SERVE_DIR/models" ]; then
  echo "Error: $SERVE_DIR/models not found."
  echo "Run fetch_ocula_stack.sh first to download models."
  exit 1
fi

echo "Serving models at http://localhost:8080/models/"
echo "Models directory: $SERVE_DIR/models"
echo ""
ls -lh "$SERVE_DIR/models/"*.gguf 2>/dev/null || echo "(no .gguf files found)"
echo ""
echo "Press Ctrl+C to stop."

cd "$SERVE_DIR" && python3 -m http.server 8080
