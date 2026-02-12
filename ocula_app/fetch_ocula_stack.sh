#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$SCRIPT_DIR/assets/models"
LLAMA_CPP="$SCRIPT_DIR/../llama.cpp"
mkdir -p "$DEST_DIR"

echo "🚀 Downloading the 2026 Ocula Intelligence Stack..."

# --- TIER 1: THE ALWAYS-ON SENSOR (Fast & Tiny) ---
# SmolVLM2-256M: Use for instant ID and background video monitoring.
echo "📥 Tier 1 — SmolVLM2 (Sensor)..."
hf download ggml-org/SmolVLM2-256M-Video-Instruct-GGUF \
    SmolVLM2-256M-Video-Instruct-Q8_0.gguf \
    mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf \
    --local-dir "$DEST_DIR"

# --- TIER 2: THE SPECIALIST (Spatial & Pointing) ---
# Moondream 2 (April 2025): Pointing (x,y), counting, spatial tasks.
# Only f16 GGUF is published — we quantize to Q4_K_M for mobile.
echo "📥 Tier 2 — Moondream 2 (Specialist)..."
QUANTIZED="$DEST_DIR/moondream2-text-model-Q4_K_M.gguf"

if [ -f "$QUANTIZED" ]; then
    echo "   ✅ Quantized model already exists, skipping."
else
    hf download ggml-org/moondream2-20250414-GGUF \
        moondream2-text-model-f16_ct-vicuna.gguf \
        moondream2-mmproj-f16-20250414.gguf \
        --local-dir "$DEST_DIR"

    # Build quantize tool if not already built
    QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
    if [ ! -f "$QUANTIZE_BIN" ]; then
        echo "   🔨 Building llama-quantize..."
        cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON 2>&1 | tail -3
        cmake --build "$LLAMA_CPP/build" --target llama-quantize -j$(sysctl -n hw.ncpu) 2>&1 | tail -3
    fi

    # Quantize f16 → Q4_K_M (~900 MB)
    echo "   ⚡ Quantizing f16 → Q4_K_M..."
    "$QUANTIZE_BIN" \
        "$DEST_DIR/moondream2-text-model-f16_ct-vicuna.gguf" \
        "$QUANTIZED" \
        Q4_K_M

    # Remove the large f16 file to save disk
    rm -f "$DEST_DIR/moondream2-text-model-f16_ct-vicuna.gguf"
    echo "   ✅ Quantized to $(du -h "$QUANTIZED" | cut -f1)"
fi

# --- TIER 3: THE THINKER (Reasoning & Chat) ---
# Qwen3-VL-2B-Thinking: The absolute Top 1 for mobile logic.
echo "📥 Tier 3 — Qwen3-VL-2B (Thinker)..."
hf download Qwen/Qwen3-VL-2B-Thinking-GGUF \
    Qwen3-VL-2B-Thinking-Q4_K_M.gguf \
    mmproj-Qwen3-VL-2B-Thinking-F16.gguf \
    --local-dir "$DEST_DIR"

echo ""
echo "✅ All GGUF models and mmproj adapters are ready."
echo ""
ls -lh "$DEST_DIR"/*.gguf
