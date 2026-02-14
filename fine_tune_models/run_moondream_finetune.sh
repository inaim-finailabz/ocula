#!/usr/bin/env bash
set -euo pipefail
#
# Moondream2 Vision Fine-tuning — Quick Run
# ==========================================
# End-to-end: Download data → Train → Merge → GGUF → Deploy
#
# Usage:
#   ./run_moondream_finetune.sh              # Full pipeline
#   ./run_moondream_finetune.sh --mlx        # Force MLX
#   ./run_moondream_finetune.sh --cuda       # Force CUDA
#   ./run_moondream_finetune.sh --4bit       # QLoRA 4-bit (CUDA)
#   ./run_moondream_finetune.sh --from merge # Resume from merge step
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Defaults ──
BACKEND="auto"
MAX_SAMPLES=8000
DATA_ONLY=false
TRAIN_ONLY=false
FROM_STEP=""
DEPLOY=false
USE_4BIT=false

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --mlx)       BACKEND="mlx"    ;;
        --cuda)      BACKEND="cuda"   ;;
        --mps)       BACKEND="mps"    ;;
        --data-only) DATA_ONLY=true   ;;
        --train-only) TRAIN_ONLY=true ;;
        --deploy)    DEPLOY=true      ;;
        --4bit)      USE_4BIT=true    ;;
        --from)      FROM_STEP="$2"; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mlx         Force MLX backend (Apple Silicon)"
            echo "  --cuda        Force CUDA backend (NVIDIA GPU)"
            echo "  --mps         Force MPS backend"
            echo "  --data-only   Only download and prepare training data"
            echo "  --train-only  Only train (assumes data already prepared)"
            echo "  --deploy      Copy final GGUF to ocula_app/assets/models/"
            echo "  --4bit        Use 4-bit QLoRA (CUDA only)"
            echo "  --from STEP   Resume from step: data|train|merge|export"
            echo "  --max-samples N  Max samples per dataset (default: 8000)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${BOLD}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Moondream2 Vision Fine-tuning Pipeline               ║${NC}"
echo -e "${BOLD}║  Plus Tier — Image Understanding + VQA               ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check Python ──
# Prefer local venv if present to avoid missing deps.
if [[ -z "${PYTHON:-}" && -x "$SCRIPT_DIR/ocula_env/bin/python3" ]]; then
    PYTHON="$SCRIPT_DIR/ocula_env/bin/python3"
else
    PYTHON="${PYTHON:-python3}"
fi
if ! command -v "$PYTHON" &>/dev/null; then
    echo -e "${RED}[!] Python not found. Install Python 3.10+${NC}"
    exit 1
fi

# ── Detect backend ──
if [[ "$BACKEND" == "auto" ]]; then
    if [[ "$(uname -m)" == "arm64" && "$(uname -s)" == "Darwin" ]]; then
        if "$PYTHON" -c "import mlx_vlm" 2>/dev/null; then
            BACKEND="mlx"
        elif "$PYTHON" -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
            BACKEND="mps"
        fi
    elif "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        BACKEND="cuda"
    fi
    [[ "$BACKEND" == "auto" ]] && BACKEND="cpu"
fi

echo -e "${CYAN}[*] Backend: ${BOLD}$BACKEND${NC}"
echo -e "${CYAN}[*] Max samples per dataset: $MAX_SAMPLES${NC}"
echo ""

should_run() {
    local step="$1"
    if [[ -n "$FROM_STEP" ]]; then
        case "$FROM_STEP" in
            data)   [[ "$step" =~ ^(data|train|merge|export|deploy)$ ]] ;;
            train)  [[ "$step" =~ ^(train|merge|export|deploy)$ ]] ;;
            merge)  [[ "$step" =~ ^(merge|export|deploy)$ ]] ;;
            export) [[ "$step" =~ ^(export|deploy)$ ]] ;;
            deploy) [[ "$step" == "deploy" ]] ;;
            *)      true ;;
        esac
    else
        true
    fi
}

# ═══════════════════════════════════════════════════════════════
# STEP 1: Prepare Training Data
# ═══════════════════════════════════════════════════════════════

if ! $TRAIN_ONLY && should_run "data"; then
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  STEP 1/5: Download & Prepare Moondream2 Vision Data${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd scripts
    "$PYTHON" 03a_prepare_moondream_data.py --max-samples "$MAX_SAMPLES"
    cd ..

    if $DATA_ONLY; then
        echo -e "\n${GREEN}[OK] Data preparation complete.${NC}"
        echo -e "     Next: $0 --train-only --$BACKEND"
        exit 0
    fi
fi

# ═══════════════════════════════════════════════════════════════
# STEP 2: Train
# ═══════════════════════════════════════════════════════════════

if should_run "train"; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  STEP 2/5: Train Moondream2 (QLoRA, $BACKEND)${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd scripts

    TRAIN_ARGS=(
        --backend "$BACKEND"
        --config ../configs/moondream2_vision.yaml
    )
    if $USE_4BIT; then
        TRAIN_ARGS+=(--use-4bit)
    fi
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
        TRAIN_ARGS+=(--max-samples "$MAX_SAMPLES")
    fi

    "$PYTHON" 03b_train_moondream2_vision.py "${TRAIN_ARGS[@]}"

    cd ..
fi

# ═══════════════════════════════════════════════════════════════
# STEP 3: Merge LoRA into base model
# ═══════════════════════════════════════════════════════════════

if should_run "merge"; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  STEP 3/5: Merge LoRA Adapters${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd scripts

    if [[ "$BACKEND" == "mlx" ]]; then
        MODEL_PATH=$(grep 'name:' ../configs/moondream2_vision.yaml | head -1 | awk '{print $2}' | tr -d '"')
        ADAPTER_PATH="../models/lora_adapters/moondream2-vision-mlx"
        SAVE_PATH="../models/merged/moondream2-vision-merged"

        echo -e "${CYAN}[*] Fusing MLX LoRA adapters...${NC}"
        "$PYTHON" -m mlx_vlm.fuse \
            --model "$MODEL_PATH" \
            --adapter-path "$ADAPTER_PATH" \
            --save-path "$SAVE_PATH"
    else
        "$PYTHON" 06_merge_lora.py --model moondream2 \
            --adapter-path ../models/lora_adapters/moondream2-vision \
            --output-path ../models/merged/moondream2-vision-merged
    fi

    cd ..
fi

# ═══════════════════════════════════════════════════════════════
# STEP 4: Export to GGUF
# ═══════════════════════════════════════════════════════════════

if should_run "export"; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  STEP 4/5: Convert to GGUF + Quantize${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd scripts
    "$PYTHON" 07b_export_moondream2_gguf.py --quant-types Q4_K_M Q8_0
    cd ..
fi

# ═══════════════════════════════════════════════════════════════
# STEP 5: Deploy to Ocula app
# ═══════════════════════════════════════════════════════════════

if should_run "deploy" && $DEPLOY; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  STEP 5/5: Deploy to Ocula App${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd scripts
    "$PYTHON" 07b_export_moondream2_gguf.py --deploy --skip-projector
    cd ..
fi

# ═══════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MOONDREAM2 PIPELINE COMPLETE${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Output GGUF:  models/gguf/"
echo -e "  Adapters:     models/lora_adapters/moondream2-vision*/"
echo -e "  Merged:       models/merged/moondream2-vision-merged/"
echo ""

if ! $DEPLOY; then
    echo -e "  ${YELLOW}To deploy to Ocula app:${NC}"
    echo -e "  ${CYAN}cp models/gguf/moondream2-finetuned-Q4_K_M.gguf \\${NC}"
    echo -e "     ${CYAN}../ocula_app/assets/models/moondream2-q4.gguf${NC}"
    echo ""
fi

echo -e "  ${YELLOW}To test on Mac:${NC}"
echo -e "  ${CYAN}cd ../ocula_app && flutter run -d macos --dart-define-from-file=.env.dev${NC}"
echo ""
