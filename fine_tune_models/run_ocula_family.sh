#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIER="all"
BACKEND="auto"
USE_4BIT=false
MAX_SAMPLES=10000
COMPRESS="balanced"
VERSION=""
PREV_VERSION=""
PROMOTE_MERGED_BASE=true
TMP_CONFIGS=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --tier TIER         all|lite|plus|pro|embed (default: all)
  --backend BACKEND   auto|cuda|mlx|mps|cpu (default: auto)
  --4bit              Enable 4-bit path where supported
  --max-samples N     Sample cap for vision pipelines (default: 10000)
  --compress PROFILE  balanced|aggressive|extreme (default: balanced)
  --version V         Version label (example: v2). Enables versioned outputs.
  --prev-version V    Previous version label (example: v1). Defaults to v(N-1) when --version is vN.
  --no-promote-merged-base
                      Disable auto-using previous merged model as training base.
  -h, --help          Show this help

Examples:
  $0 --tier lite --backend cuda --compress aggressive
  $0 --tier lite --backend cuda --version v2
  $0 --tier all --backend cuda --version v3 --prev-version v2
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
        --version) VERSION="$2"; shift 2 ;;
        --prev-version) PREV_VERSION="$2"; shift 2 ;;
        --no-promote-merged-base) PROMOTE_MERGED_BASE=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cleanup_tmp_configs() {
    for f in "${TMP_CONFIGS[@]:-}"; do
        [[ -f "$f" ]] && rm -f "$f"
    done
}
trap cleanup_tmp_configs EXIT

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

if [[ -n "$VERSION" && -z "$PREV_VERSION" && "$VERSION" =~ ^v([0-9]+)$ ]]; then
    vn="${BASH_REMATCH[1]}"
    if (( vn > 1 )); then
        PREV_VERSION="v$((vn - 1))"
    fi
fi

tmp_config_path() {
    local cfg="$1"
    local tag="$2"
    if [[ -z "$VERSION" ]]; then
        echo "$cfg"
    else
        echo "$SCRIPT_DIR/configs/.tmp.$tag.$VERSION.yaml"
    fi
}

build_versioned_config() {
    local src_cfg="$1"
    local out_cfg="$2"
    local base_path="$3"
    local short_name="$4"
    if [[ "$base_path" == *.gguf ]]; then
        echo "[!] Refusing GGUF as training base: $base_path"
        exit 1
    fi
    py -c 'import sys,yaml
src,out,base,short=sys.argv[1:5]
with open(src) as f:
    cfg=yaml.safe_load(f) or {}
cfg.setdefault("model", {})["local_path"]=base
cfg["model"]["short_name"]=short
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
' "$src_cfg" "$out_cfg" "$base_path" "$short_name"
}

pick_base_model() {
    local default_base="$1"
    local prev_merged="$2"
    local label="$3"
    local chosen="$default_base"
    if $PROMOTE_MERGED_BASE && [[ -n "$PREV_VERSION" ]] && [[ -d "$prev_merged" ]]; then
        chosen="$prev_merged"
        echo "[*] [$label] Using previous merged base: $chosen" >&2
    else
        echo "[*] [$label] Using configured base: $chosen" >&2
    fi
    echo "$chosen"
}

get_model_local_path() {
    local cfg="$1"
    py -c 'import sys,yaml
with open(sys.argv[1]) as f:
    cfg=yaml.safe_load(f) or {}
print(cfg.get("model", {}).get("local_path", ""))
' "$cfg"
}

run_lite() {
    echo "[ocula-lite] Train Qwen3-1.7B text"
    local cfg="configs/qwen25_1_5b_text.yaml"
    local short_name="ocula-lite-qwen3-1_7b"
    local default_base="models/base/qwen3-1.7b"
    local prev_merged="models/merged/ocula-lite-${PREV_VERSION}-merged"
    local merged_out="models/merged/ocula-lite-${VERSION}-merged"
    local quant_name="Ocula-Lite-Qwen3-1.7B-${VERSION}"
    local cfg_to_use="$cfg"

    if [[ -n "$VERSION" ]]; then
        short_name="${short_name}-${VERSION}"
        cfg_to_use="$(tmp_config_path "$cfg" "lite")"
        local base_to_use
        base_to_use="$(pick_base_model "$default_base" "$prev_merged" "ocula-lite")"
        build_versioned_config "$cfg" "$cfg_to_use" "$base_to_use" "$short_name"
        TMP_CONFIGS+=("$cfg_to_use")
    fi

    text_backend="auto"
    merge_backend="cuda"
    if [[ "$BACKEND" == "mlx" ]]; then
        text_backend="mlx"
        merge_backend="mlx"
    elif [[ "$BACKEND" == "cuda" ]]; then
        text_backend="cuda"
        merge_backend="cuda"
    fi
    py scripts/04_train_text_qwen.py --config "$cfg_to_use" --backend "$text_backend"
    if [[ -n "$VERSION" ]]; then
        local base_to_use
        base_to_use="$(pick_base_model "$default_base" "$prev_merged" "ocula-lite-merge")"
        py scripts/06_merge_lora.py --model ocula_lite --backend "$merge_backend" \
            --base-path "$base_to_use" \
            --output-path "$merged_out"
        py scripts/07_quantize_gguf.py --model ocula_lite --mobile-preset "$COMPRESS" \
            --merged-path "$merged_out" \
            --output-name "$quant_name" \
            --config "$cfg_to_use"
    else
        py scripts/06_merge_lora.py --model ocula_lite --backend "$merge_backend"
        py scripts/07_quantize_gguf.py --model ocula_lite --mobile-preset "$COMPRESS"
    fi
}

run_plus() {
    echo "[ocula-plus] Train Qwen3-VL-2B vision"
    local cfg="configs/qwen3vl_vision.yaml"
    local short_name="ocula-plus-qwen3vl-2b"
    local default_base="models/base/qwen3-vl-2b"
    local prev_merged="models/merged/ocula-plus-${PREV_VERSION}-vision-merged"
    local cfg_to_use="$cfg"
    local merged_out="models/merged/ocula-plus-${VERSION}-vision-merged"
    local quant_name="Ocula-Plus-Qwen3-VL-2B-${VERSION}"
    local train_output=""

    if [[ -n "$VERSION" ]]; then
        short_name="ocula-plus-${VERSION}"
        cfg_to_use="$(tmp_config_path "$cfg" "plus")"
        local base_to_use
        base_to_use="$(pick_base_model "$default_base" "$prev_merged" "ocula-plus")"
        build_versioned_config "$cfg" "$cfg_to_use" "$base_to_use" "$short_name"
        TMP_CONFIGS+=("$cfg_to_use")
        train_output="models/merged/${short_name}-vision-adapters"
    fi

    py scripts/04a_prepare_qwen3vl_data.py --max-samples "$MAX_SAMPLES"
    plus_args=(
        --config "$cfg_to_use"
        --train-data data/vision_qwen3vl/qwen3vl_vision_train.jsonl
        --val-data data/vision_qwen3vl/qwen3vl_vision_val.jsonl
        --max-samples "$MAX_SAMPLES"
    )
    if [[ -n "$train_output" ]]; then
        plus_args+=(--output "$train_output")
    fi
    plus_args+=("${backend_flag[@]}")
    if $USE_4BIT; then
        plus_args+=(--use-4bit)
    fi
    py scripts/04b_train_vision_qwen.py "${plus_args[@]}"
    if [[ "$BACKEND" == "mlx" ]]; then
        if [[ -n "$VERSION" ]]; then
            local base_to_use
            base_to_use="$(get_model_local_path "$cfg_to_use")"
            py -m mlx_vlm.fuse \
                --model "$base_to_use" \
                --adapter-path "$train_output" \
                --save-path "$merged_out"
        else
        py -m mlx_vlm.fuse \
            --model models/base/qwen3-vl-2b \
            --adapter-path models/lora_adapters/ocula-plus-qwen3vl-2b-vision-mlx \
            --save-path models/merged/ocula-plus-qwen3vl-2b-vision-merged
        fi
    fi
    if [[ -n "$VERSION" ]]; then
        py scripts/07_quantize_gguf.py --model ocula_plus --mobile-preset "$COMPRESS" \
            --merged-path "$merged_out" \
            --output-name "$quant_name" \
            --config "$cfg_to_use"
    else
        py scripts/07_quantize_gguf.py --model ocula_plus --mobile-preset "$COMPRESS"
    fi
}

run_pro() {
    echo "[ocula-pro] Train Qwen2.5-VL-7B vision"
    local cfg="configs/qwen25vl_7b_vision.yaml"
    local short_name="ocula-pro-qwen25vl-7b"
    local default_base="models/base/qwen2.5-vl-7b-instruct"
    local prev_merged="models/merged/ocula-pro-${PREV_VERSION}-vision-merged"
    local cfg_to_use="$cfg"
    local merged_out="models/merged/ocula-pro-${VERSION}-vision-merged"
    local quant_name="Ocula-Pro-Qwen2.5-VL-7B-${VERSION}"
    local train_output=""

    if [[ -n "$VERSION" ]]; then
        short_name="ocula-pro-${VERSION}"
        cfg_to_use="$(tmp_config_path "$cfg" "pro")"
        local base_to_use
        base_to_use="$(pick_base_model "$default_base" "$prev_merged" "ocula-pro")"
        build_versioned_config "$cfg" "$cfg_to_use" "$base_to_use" "$short_name"
        TMP_CONFIGS+=("$cfg_to_use")
        train_output="models/merged/${short_name}-vision-adapters"
    fi

    py scripts/04a_prepare_qwen3vl_data.py --max-samples "$MAX_SAMPLES"
    mkdir -p data/vision_qwen25vl7b
    cp -f data/vision_qwen3vl/qwen3vl_vision_train.jsonl data/vision_qwen25vl7b/qwen25vl_vision_train.jsonl
    cp -f data/vision_qwen3vl/qwen3vl_vision_val.jsonl data/vision_qwen25vl7b/qwen25vl_vision_val.jsonl
    if [[ -d data/vision_qwen3vl/images ]] && [[ ! -e data/vision_qwen25vl7b/images ]]; then
        ln -s ../vision_qwen3vl/images data/vision_qwen25vl7b/images
    fi
    pro_args=(
        --config "$cfg_to_use"
        --train-data data/vision_qwen25vl7b/qwen25vl_vision_train.jsonl
        --val-data data/vision_qwen25vl7b/qwen25vl_vision_val.jsonl
        --max-samples "$MAX_SAMPLES"
    )
    if [[ -n "$train_output" ]]; then
        pro_args+=(--output "$train_output")
    fi
    pro_args+=("${backend_flag[@]}")
    if $USE_4BIT; then
        pro_args+=(--use-4bit)
    fi
    py scripts/04b_train_vision_qwen.py "${pro_args[@]}"
    if [[ "$BACKEND" == "mlx" ]]; then
        if [[ -n "$VERSION" ]]; then
            local base_to_use
            base_to_use="$(get_model_local_path "$cfg_to_use")"
            py -m mlx_vlm.fuse \
                --model "$base_to_use" \
                --adapter-path "$train_output" \
                --save-path "$merged_out"
        else
        py -m mlx_vlm.fuse \
            --model models/base/qwen2.5-vl-7b-instruct \
            --adapter-path models/lora_adapters/ocula-pro-qwen25vl-7b-vision-mlx \
            --save-path models/merged/ocula-pro-qwen25vl-7b-vision-merged
        fi
    fi
    if [[ -n "$VERSION" ]]; then
        py scripts/07_quantize_gguf.py --model ocula_pro --mobile-preset "$COMPRESS" \
            --merged-path "$merged_out" \
            --output-name "$quant_name" \
            --config "$cfg_to_use"
    else
        py scripts/07_quantize_gguf.py --model ocula_pro --mobile-preset "$COMPRESS"
    fi
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
