#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIER="all"
BACKEND="auto"
USE_4BIT=false
MAX_SAMPLES=10000
COMPRESS="balanced"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --tier TIER         all|lite|plus|pro|embed (default: all)
  --backend BACKEND   auto|cuda|mlx|mps|cpu (default: auto)
  --4bit              Enable 4-bit path where supported
  --max-samples N     Sample cap for vision pipelines (default: 10000)
  --compress PROFILE  balanced|aggressive|extreme (default: balanced)
  -h, --help          Show this help

Examples:
  $0 --tier lite --backend cuda --compress aggressive
  $0 --tier plus --backend cuda --4bit
  $0 --tier embed
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --4bit) USE_4BIT=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --compress) COMPRESS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

py() {
    if [[ -x "$SCRIPT_DIR/ocula_env/bin/python3" ]]; then
        "$SCRIPT_DIR/ocula_env/bin/python3" "$@"
    else
        python3 "$@"
    fi
}

backend_flag=()
case "$BACKEND" in
    cuda) backend_flag=(--backend cuda) ;;
    mlx) backend_flag=(--backend mlx) ;;
    mps) backend_flag=(--backend mps) ;;
    cpu) backend_flag=(--backend cpu) ;;
    auto) backend_flag=(--backend auto) ;;
    *) echo "Unsupported backend: $BACKEND"; exit 1 ;;
esac

run_lite() {
    echo "[ocula-lite] Train Qwen2.5-1.5B text"
    text_backend="auto"
    merge_backend="cuda"
    if [[ "$BACKEND" == "mlx" ]]; then
        text_backend="mlx"
        merge_backend="mlx"
    elif [[ "$BACKEND" == "cuda" ]]; then
        text_backend="cuda"
        merge_backend="cuda"
    fi
    py scripts/04_train_text_qwen.py --config configs/qwen25_1_5b_text.yaml --backend "$text_backend"
    py scripts/06_merge_lora.py --model ocula_lite --backend "$merge_backend"
    py scripts/07_quantize_gguf.py --model ocula_lite --mobile-preset "$COMPRESS"
}

run_plus() {
    echo "[ocula-plus] Train Qwen3-VL-2B vision"
    py scripts/04a_prepare_qwen3vl_data.py --max-samples "$MAX_SAMPLES"
    plus_args=(
        --config configs/qwen3vl_vision.yaml
        --train-data data/vision_qwen3vl/qwen3vl_vision_train.jsonl
        --val-data data/vision_qwen3vl/qwen3vl_vision_val.jsonl
        --max-samples "$MAX_SAMPLES"
    )
    plus_args+=("${backend_flag[@]}")
    if $USE_4BIT; then
        plus_args+=(--use-4bit)
    fi
    py scripts/04b_train_vision_qwen.py "${plus_args[@]}"
    if [[ "$BACKEND" == "mlx" ]]; then
        py -m mlx_vlm.fuse \
            --model models/base/qwen3-vl-2b \
            --adapter-path models/lora_adapters/ocula-plus-qwen3vl-2b-vision-mlx \
            --save-path models/merged/ocula-plus-qwen3vl-2b-vision-merged
    fi
    py scripts/07_quantize_gguf.py --model ocula_plus --mobile-preset "$COMPRESS"
}

run_pro() {
    echo "[ocula-pro] Train Qwen2.5-VL-7B vision"
    py scripts/04a_prepare_qwen3vl_data.py --max-samples "$MAX_SAMPLES"
    mkdir -p data/vision_qwen25vl7b
    cp -f data/vision_qwen3vl/qwen3vl_vision_train.jsonl data/vision_qwen25vl7b/qwen25vl_vision_train.jsonl
    cp -f data/vision_qwen3vl/qwen3vl_vision_val.jsonl data/vision_qwen25vl7b/qwen25vl_vision_val.jsonl
    if [[ -d data/vision_qwen3vl/images ]] && [[ ! -e data/vision_qwen25vl7b/images ]]; then
        ln -s ../vision_qwen3vl/images data/vision_qwen25vl7b/images
    fi
    pro_args=(
        --config configs/qwen25vl_7b_vision.yaml
        --train-data data/vision_qwen25vl7b/qwen25vl_vision_train.jsonl
        --val-data data/vision_qwen25vl7b/qwen25vl_vision_val.jsonl
        --max-samples "$MAX_SAMPLES"
    )
    pro_args+=("${backend_flag[@]}")
    if $USE_4BIT; then
        pro_args+=(--use-4bit)
    fi
    py scripts/04b_train_vision_qwen.py "${pro_args[@]}"
    if [[ "$BACKEND" == "mlx" ]]; then
        py -m mlx_vlm.fuse \
            --model models/base/qwen2.5-vl-7b-instruct \
            --adapter-path models/lora_adapters/ocula-pro-qwen25vl-7b-vision-mlx \
            --save-path models/merged/ocula-pro-qwen25vl-7b-vision-merged
    fi
    py scripts/07_quantize_gguf.py --model ocula_pro --mobile-preset "$COMPRESS"
}

run_embed() {
    echo "[embed] Train Qwen3-Embedding-0.6B"
    py scripts/05_train_embedding.py --config configs/qwen3embedding.yaml
    py scripts/07_quantize_gguf.py --model qwen3embed --quant-types Q8_0
}

case "$TIER" in
    all)
        run_lite
        run_plus
        run_pro
        run_embed
        ;;
    lite) run_lite ;;
    plus) run_plus ;;
    pro) run_pro ;;
    embed) run_embed ;;
    *)
        echo "Unsupported tier: $TIER"
        usage
        exit 1
        ;;
esac

echo ""
echo "[OK] Ocula family pipeline complete for tier=$TIER"
