#!/usr/bin/env bash
set -euo pipefail
#
# Ocula MLOps Pipeline Orchestrator
# ==================================
# Runs end-to-end training pipelines for all vision models.
# Delegates to individual model pipelines for actual execution.
#
# Usage:
#   ./run_pipeline.sh                             # All models, full pipeline
#   ./run_pipeline.sh --model smolvlm2            # Single model
#   ./run_pipeline.sh --model qwen3vl --from export  # Resume from export
#   ./run_pipeline.sh --all --deploy              # Full pipeline + deploy
#   ./run_pipeline.sh --all --dry-run             # Preview what would run
#
# Models:
#   smolvlm2   → SmolVLM2-500M  (Lite tier, image detection + docs)
#   moondream2 → Moondream2-2B  (Plus tier, image understanding + VQA)
#   qwen3vl    → Qwen3-VL-2B   (Pro tier, document + chart + reasoning)
#   minilm     → MiniLM-L6     (Embedding model, sentence similarity)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${SCRIPT_DIR}/logs"

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Defaults ────────────────────────────────────────────────────
MODEL="all"
BACKEND="auto"
FROM_STEP=""
DEPLOY=false
USE_4BIT=false
MAX_SAMPLES=10000
DRY_RUN=false
NO_COMPARE=false
LOG_FILE=""
COMPARE_BACKEND="${COMPARE_BACKEND:-cli}"

ALL_MODELS=(smolvlm2 moondream2 qwen3vl minilm)

# ─────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
${BOLD}Ocula MLOps Pipeline Orchestrator${NC}

${CYAN}Usage:${NC}
  $0 [OPTIONS]

${CYAN}Options:${NC}
  --all                  Train all models (default)
  --model MODEL          Train specific model: smolvlm2|moondream2|qwen3vl|minilm
  --backend BACKEND      Force backend: cuda|mlx|mps|cpu|auto (default: auto)
  --from STEP            Resume from step: data|train|export|compare|deploy
  --deploy               Deploy final GGUFs to ocula_app/assets/models/
  --4bit                 Use 4-bit QLoRA (CUDA only)
  --max-samples N        Max training samples per dataset (default: 10000)
  --no-compare           Skip comparison step
  --dry-run              Preview pipeline without executing
  --log FILE             Log all output to file
  -h, --help             Show this help

${CYAN}Pipeline Steps (per model):${NC}
  1. data     → Download and prepare model-specific training data
  2. train    → Fine-tune with LoRA/QLoRA (merge handled by Unsloth)
  3. export   → Convert to GGUF + quantize (Q4_K_M, Q8_0)
  4. compare  → Compare base vs fine-tuned via local CLI inference
  5. deploy   → Copy GGUF to ocula_app/assets/models/

${CYAN}Examples:${NC}
  $0 --all                                      # Full pipeline, all models
  $0 --model smolvlm2 --backend mlx             # Single model on Apple Silicon
  $0 --model qwen3vl --from export              # Resume from export step
  $0 --all --deploy --4bit                      # Full pipeline, QLoRA, deploy
  $0 --model moondream2 --from compare          # Re-run comparison only

EOF
    exit 0
}

# ── Parse args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) MODEL="all"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --from) FROM_STEP="$2"; shift 2 ;;
        --deploy) DEPLOY=true; shift ;;
        --4bit) USE_4BIT=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --no-compare) NO_COMPARE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --log) LOG_FILE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; usage ;;
    esac
done

# ── Resolve models ──────────────────────────────────────────────
if [[ "$MODEL" == "all" ]]; then
    ACTIVE_MODELS=("${ALL_MODELS[@]}")
else
    ACTIVE_MODELS=("$MODEL")
fi

# ── Logging ─────────────────────────────────────────────────────
mkdir -p "$LOGS"
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="${LOGS}/pipeline_$(date +%Y%m%d_%H%M%S).log"
fi

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo -e "$msg"
    echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
}

# ── Build common args for individual pipeline scripts ───────────
build_args() {
    local model="$1"
    local args=()

    # Backend
    case "$BACKEND" in
        mlx)  args+=(--mlx) ;;
        cuda) args+=(--cuda) ;;
        mps)  args+=(--mps) ;;
        cpu)  ;; # default fallback
        auto) ;; # let sub-script auto-detect
    esac

    # Resume from step
    if [[ -n "$FROM_STEP" ]]; then
        args+=(--from "$FROM_STEP")
    fi

    # Deploy
    if $DEPLOY; then
        args+=(--deploy)
    fi

    # 4-bit QLoRA
    if $USE_4BIT; then
        args+=(--4bit)
    fi

    # Max samples
    args+=(--max-samples "$MAX_SAMPLES")

    echo "${args[@]}"
}

# ── Map model → individual pipeline script ──────────────────────
get_pipeline_script() {
    local model="$1"
    case "$model" in
        smolvlm2)   echo "${SCRIPT_DIR}/run_vision_finetune.sh" ;;
        moondream2) echo "${SCRIPT_DIR}/run_moondream_finetune.sh" ;;
        qwen3vl)    echo "${SCRIPT_DIR}/run_qwen3vl_finetune.sh" ;;
        minilm)     echo "${SCRIPT_DIR}/scripts/05_train_minilm.py" ;;
        *)
            log "${RED}Unknown model: ${model}${NC}"
            return 1
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Ocula MLOps Pipeline${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Models:      ${CYAN}${ACTIVE_MODELS[*]}${NC}"
echo -e "  Backend:     ${CYAN}${BACKEND}${NC}"
echo -e "  Max Samples: ${CYAN}${MAX_SAMPLES}${NC}"
echo -e "  Deploy:      ${CYAN}${DEPLOY}${NC}"
echo -e "  Compare:     ${CYAN}$(if $NO_COMPARE; then echo "SKIP"; else echo "${COMPARE_BACKEND}"; fi)${NC}"
if [[ -n "$FROM_STEP" ]]; then
    echo -e "  Resume From: ${CYAN}${FROM_STEP}${NC}"
fi
if [[ "$DRY_RUN" == true ]]; then
    echo -e "  Mode:        ${YELLOW}DRY RUN${NC}"
fi
echo -e "  Log:         ${DIM}${LOG_FILE}${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Check Python (for minilm) ──
if [[ -z "${PYTHON:-}" && -x "$SCRIPT_DIR/ocula_env/bin/python3" ]]; then
    PYTHON="$SCRIPT_DIR/ocula_env/bin/python3"
else
    PYTHON="${PYTHON:-python3}"
fi

# Track results
declare -A MODEL_RESULTS
PIPELINE_START=$(date +%s)
TOTAL=0
PASSED=0
FAILED=0

for model in "${ACTIVE_MODELS[@]}"; do
    ((TOTAL++))
    log "${BOLD}── Model: ${model} ──${NC}"

    pipeline_script="$(get_pipeline_script "$model")"
    if [[ ! -f "$pipeline_script" ]]; then
        log "  ${RED}Pipeline script not found: ${pipeline_script}${NC}"
        MODEL_RESULTS["$model"]="FAILED"
        ((FAILED++))
        continue
    fi

    if [[ "$DRY_RUN" == true ]]; then
        local_args=$(build_args "$model")
        log "  ${DIM}[DRY RUN] Would run: bash ${pipeline_script} ${local_args}${NC}"
        MODEL_RESULTS["$model"]="DRY_RUN"
        continue
    fi

    if [[ "$model" == "minilm" ]]; then
        # MiniLM is a simple embedding model — just run the Python script
        log "  ${CYAN}Training MiniLM embedding model...${NC}"
        if $PYTHON "$pipeline_script" 2>&1 | tee -a "$LOG_FILE"; then
            MODEL_RESULTS["$model"]="OK"
            ((PASSED++))
            log "  ${GREEN}MiniLM training complete${NC}"
        else
            MODEL_RESULTS["$model"]="FAILED"
            ((FAILED++))
            log "  ${RED}MiniLM training failed${NC}"
        fi
        continue
    fi

    # Vision models — delegate to individual pipeline script
    local_args=$(build_args "$model")

    # Export compare backend so sub-scripts inherit it
    export COMPARE_BACKEND

    log "  ${CYAN}Running: bash ${pipeline_script} ${local_args}${NC}"
    echo ""

    if bash "$pipeline_script" $local_args 2>&1 | tee -a "$LOG_FILE"; then
        MODEL_RESULTS["$model"]="OK"
        ((PASSED++))
        log "  ${GREEN}${model} pipeline complete${NC}"
    else
        MODEL_RESULTS["$model"]="FAILED"
        ((FAILED++))
        log "  ${RED}${model} pipeline failed${NC}"
    fi
    echo ""
done

PIPELINE_END=$(date +%s)
ELAPSED=$((PIPELINE_END - PIPELINE_START))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

# ── Final Summary ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Pipeline Summary${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Duration:  ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo -e "  ${GREEN}Passed:  ${PASSED}/${TOTAL}${NC}"
if [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}Failed:  ${FAILED}/${TOTAL}${NC}"
fi
echo ""

# Model breakdown
echo -e "  ${DIM}Model Results:${NC}"
for model in "${ACTIVE_MODELS[@]}"; do
    status="${MODEL_RESULTS[$model]:-—}"
    case "$status" in
        OK)      echo -e "    ${GREEN}✓${NC} ${model}" ;;
        FAILED)  echo -e "    ${RED}✗${NC} ${model}" ;;
        DRY_RUN) echo -e "    ${DIM}~${NC} ${model} (dry run)" ;;
        *)       echo -e "    ${DIM}—${NC} ${model}" ;;
    esac
done

echo ""
echo -e "  Log: ${DIM}${LOG_FILE}${NC}"

if $DEPLOY && [[ $PASSED -gt 0 ]]; then
    echo ""
    echo -e "  ${GREEN}Deployed models ready for app rebuild:${NC}"
    echo -e "    cd ocula_app && flutter build ios --release"
    echo -e "    cd ocula_app && flutter build apk --release"
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit $FAILED
