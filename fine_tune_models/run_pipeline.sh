#!/usr/bin/env bash
set -euo pipefail
#
# Ocula MLOps Pipeline Orchestrator
# End-to-end: Data Prep → Train → Merge → Quantize → Evaluate → Deploy
#
# Usage:
#   ./run_pipeline.sh --all                    # Full pipeline, all models
#   ./run_pipeline.sh --model smolvlm2         # Single model end-to-end
#   ./run_pipeline.sh --model moondream2 --from train  # Resume from training
#   ./run_pipeline.sh --stages train,merge,quantize    # Specific stages
#   ./run_pipeline.sh --all --dry-run          # Preview what would run
#
# Stages (in order):
#   prepare  → Format raw data into training datasets
#   train    → Fine-tune model (LoRA/QLoRA/full)
#   merge    → Merge LoRA adapter into base model
#   quantize → Convert to GGUF + quantize
#   evaluate → Benchmark base vs fine-tuned
#   deploy   → Copy GGUF to model server / local
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${SCRIPT_DIR}/scripts"
LOGS="${SCRIPT_DIR}/logs"
PYTHON="${PYTHON:-python3}"

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
STAGES="prepare,train,merge,quantize,evaluate,deploy"
FROM_STAGE=""
DEPLOY_TARGET="local"
DRY_RUN=false
SKIP_EVAL=false
NO_DEPLOY=false
LOG_FILE=""

ALL_STAGES=(prepare train merge quantize evaluate deploy)
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
  --backend BACKEND      Force backend: cuda|mlx|auto (default: auto)
  --stages STAGES        Comma-separated stages to run (default: all)
  --from STAGE           Start from this stage (skip earlier stages)
  --deploy-target TGT    Deploy target: local|ssh://...|http://... (default: local)
  --skip-eval            Skip evaluation stage
  --no-deploy            Skip deployment stage
  --dry-run              Preview pipeline without executing
  --log FILE             Log all output to file
  -h, --help             Show this help

${CYAN}Stages:${NC}
  prepare   Format raw data → training datasets
  train     Fine-tune model (LoRA/QLoRA/full)
  merge     Merge LoRA adapter → full model weights
  quantize  Convert merged model → GGUF + quantize
  evaluate  Benchmark base vs fine-tuned models
  deploy    Copy GGUF to model server

${CYAN}Examples:${NC}
  $0 --all                                      # Full pipeline
  $0 --model smolvlm2 --backend mlx             # Single model on Apple Silicon
  $0 --model qwen3vl --from merge               # Resume from merge step
  $0 --stages train,merge --model moondream2     # Only train + merge
  $0 --all --skip-eval --no-deploy              # Train everything, stop before deploy

EOF
    exit 0
}

# ── Parse args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) MODEL="all"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --from) FROM_STAGE="$2"; shift 2 ;;
        --deploy-target) DEPLOY_TARGET="$2"; shift 2 ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --no-deploy) NO_DEPLOY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --log) LOG_FILE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; usage ;;
    esac
done

# ── Resolve stages ──────────────────────────────────────────────
resolve_stages() {
    local result=()

    if [[ -n "$FROM_STAGE" ]]; then
        local found=false
        for s in "${ALL_STAGES[@]}"; do
            if [[ "$s" == "$FROM_STAGE" ]]; then
                found=true
            fi
            if [[ "$found" == true ]]; then
                result+=("$s")
            fi
        done
        if [[ "$found" == false ]]; then
            echo -e "${RED}Unknown stage: ${FROM_STAGE}${NC}"
            echo "Valid stages: ${ALL_STAGES[*]}"
            exit 1
        fi
    else
        IFS=',' read -ra result <<< "$STAGES"
    fi

    # Apply skip flags
    if [[ "$SKIP_EVAL" == true ]]; then
        result=("${result[@]/evaluate/}")
    fi
    if [[ "$NO_DEPLOY" == true ]]; then
        result=("${result[@]/deploy/}")
    fi

    # Remove empty entries
    local clean=()
    for s in "${result[@]}"; do
        [[ -n "$s" ]] && clean+=("$s")
    done
    echo "${clean[@]}"
}

ACTIVE_STAGES=($(resolve_stages))

# ── Resolve models ──────────────────────────────────────────────
if [[ "$MODEL" == "all" ]]; then
    ACTIVE_MODELS=("${ALL_MODELS[@]}")
else
    ACTIVE_MODELS=("$MODEL")
fi

# ── Logging ─────────────────────────────────────────────────────
if [[ -z "$LOG_FILE" ]]; then
    mkdir -p "$LOGS"
    LOG_FILE="${LOGS}/pipeline_$(date +%Y%m%d_%H%M%S).log"
fi

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo -e "$msg"
    echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
}

# ── Stage runner ────────────────────────────────────────────────
run_stage() {
    local stage=$1
    local model=$2
    local exit_code=0

    log "${CYAN}▸ ${stage}${NC} (${model})"

    if [[ "$DRY_RUN" == true ]]; then
        log "  ${DIM}[DRY RUN] Would run ${stage} for ${model}${NC}"
        return 0
    fi

    case "$stage" in
        prepare)
            $PYTHON "${SCRIPTS}/01_prepare_data.py" \
                --input ../data/raw/ \
                --output ../data/processed/ \
                2>&1 | tee -a "$LOG_FILE" || exit_code=$?
            ;;

        train)
            local script=""
            case "$model" in
                smolvlm2)   script="02_train_smolvlm2.py" ;;
                moondream2) script="03_train_moondream2.py" ;;
                qwen3vl)    script="04_train_qwen3vl.py" ;;
                minilm)     script="05_train_minilm.py" ;;
            esac

            local backend_arg=""
            if [[ "$BACKEND" != "auto" ]] && [[ "$model" != "minilm" ]]; then
                backend_arg="--backend ${BACKEND}"
            fi

            $PYTHON "${SCRIPTS}/${script}" ${backend_arg} \
                2>&1 | tee -a "$LOG_FILE" || exit_code=$?
            ;;

        merge)
            if [[ "$model" == "minilm" ]]; then
                log "  ${DIM}Skip merge for embedding model (full fine-tune)${NC}"
                return 0
            fi

            local backend_arg=""
            if [[ "$BACKEND" != "auto" ]]; then
                backend_arg="--backend ${BACKEND}"
            fi

            $PYTHON "${SCRIPTS}/06_merge_lora.py" \
                --model "$model" ${backend_arg} \
                2>&1 | tee -a "$LOG_FILE" || exit_code=$?
            ;;

        quantize)
            $PYTHON "${SCRIPTS}/07_quantize_gguf.py" \
                --model "$model" \
                2>&1 | tee -a "$LOG_FILE" || exit_code=$?
            ;;

        evaluate)
            $PYTHON "${SCRIPTS}/08_evaluate.py" \
                --model "$model" \
                2>&1 | tee -a "$LOG_FILE" || exit_code=$?
            ;;

        deploy)
            bash "${SCRIPTS}/09_deploy.sh" \
                --model "$model" \
                --target "$DEPLOY_TARGET" \
                2>&1 | tee -a "$LOG_FILE" || exit_code=$?
            ;;
    esac

    return $exit_code
}

# ─────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Ocula MLOps Pipeline${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Models:  ${CYAN}${ACTIVE_MODELS[*]}${NC}"
echo -e "  Stages:  ${CYAN}${ACTIVE_STAGES[*]}${NC}"
echo -e "  Backend: ${CYAN}${BACKEND}${NC}"
echo -e "  Deploy:  ${CYAN}${DEPLOY_TARGET}${NC}"
echo -e "  Log:     ${DIM}${LOG_FILE}${NC}"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "  Mode:    ${YELLOW}DRY RUN${NC}"
fi
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Track results
declare -A STAGE_RESULTS
PIPELINE_START=$(date +%s)
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Data preparation runs once (not per-model)
if [[ " ${ACTIVE_STAGES[*]} " == *" prepare "* ]]; then
    log "${BOLD}── Stage: prepare ──${NC}"
    ((TOTAL++))
    if run_stage "prepare" "all"; then
        STAGE_RESULTS["prepare:all"]="OK"
        ((PASSED++))
    else
        STAGE_RESULTS["prepare:all"]="FAILED"
        ((FAILED++))
        log "${RED}[!] Data preparation failed — aborting pipeline${NC}"
        # Don't abort — downstream stages might still work with existing data
    fi
    echo ""
fi

# Per-model stages
for model in "${ACTIVE_MODELS[@]}"; do
    log "${BOLD}── Model: ${model} ──${NC}"

    for stage in "${ACTIVE_STAGES[@]}"; do
        # Skip prepare (already ran) and handle stage ordering
        [[ "$stage" == "prepare" ]] && continue

        ((TOTAL++))

        if run_stage "$stage" "$model"; then
            STAGE_RESULTS["${stage}:${model}"]="OK"
            ((PASSED++))
            log "  ${GREEN}✓ ${stage} complete${NC}"
        else
            STAGE_RESULTS["${stage}:${model}"]="FAILED"
            ((FAILED++))
            log "  ${RED}✗ ${stage} failed${NC}"

            # Critical failures stop downstream for this model
            if [[ "$stage" == "train" ]]; then
                log "  ${YELLOW}⚠ Training failed — skipping merge/quantize/deploy for ${model}${NC}"
                break
            fi
        fi
    done
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

# Detailed breakdown
echo -e "  ${DIM}Stage Results:${NC}"
printf "  %-12s" ""
for model in "${ACTIVE_MODELS[@]}"; do
    printf "%-14s" "$model"
done
echo ""

for stage in "${ACTIVE_STAGES[@]}"; do
    printf "  %-12s" "$stage"
    for model in "${ACTIVE_MODELS[@]}"; do
        local_key="${stage}:${model}"
        if [[ "$stage" == "prepare" ]]; then
            local_key="prepare:all"
        fi
        status="${STAGE_RESULTS[$local_key]:-—}"
        case "$status" in
            OK)     printf "${GREEN}%-14s${NC}" "✓" ;;
            FAILED) printf "${RED}%-14s${NC}" "✗" ;;
            *)      printf "${DIM}%-14s${NC}" "—" ;;
        esac
    done
    echo ""
done

echo ""
echo -e "  Log: ${DIM}${LOG_FILE}${NC}"

# List deployed models
if [[ " ${ACTIVE_STAGES[*]} " == *" deploy "* ]] && [[ $PASSED -gt 0 ]]; then
    echo ""
    echo -e "  ${GREEN}Deployed models ready for app rebuild:${NC}"
    echo -e "    cd ocula_app && flutter build ios --release"
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit $FAILED
