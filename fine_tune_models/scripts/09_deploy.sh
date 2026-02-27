#!/usr/bin/env bash
set -euo pipefail
#
# Deploy fine-tuned GGUF models to the Ocula model server / local device.
#
# Usage:
#   ./09_deploy.sh                              # Deploy to local models/ dir
#   ./09_deploy.sh --target local                # Same as above
#   ./09_deploy.sh --target ssh://pi@192.168.3.14:/models
#   ./09_deploy.sh --target http://192.168.3.14:8080
#   ./09_deploy.sh --model ocula_lite           # Deploy only one model
#   ./09_deploy.sh --dry-run                     # Preview without copying
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GGUF_DIR="${SCRIPT_DIR}/../models/gguf"
LOCAL_MODELS_DIR="${SCRIPT_DIR}/../../models"

# ── Defaults ────────────────────────────────────────────────────
TARGET="local"
MODEL="all"
DRY_RUN=false
QUANT_PREFERENCE="Q4_K_M"  # Preferred quantization for deployment

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Model mapping (gguf prefix → deployment filename) ──────────
declare -A MODEL_MAP
MODEL_MAP[qwen3embed]="Qwen3-Embedding-0.6B"
MODEL_MAP[ocula_lite]="Ocula-Lite-Qwen3-1.7B"
MODEL_MAP[ocula_plus]="Ocula-Plus-Qwen3-VL-2B"
MODEL_MAP[ocula_pro]="Ocula-Pro-Qwen2.5-VL-7B"

# ── Deployment filenames (what Ocula app expects) ──────────────
declare -A DEPLOY_NAMES
DEPLOY_NAMES[qwen3embed]="Qwen3-Embedding-0.6B-Q8_0.gguf"
DEPLOY_NAMES[ocula_lite]="Qwen3-1.7B-Ocula-Lite-Q4_K_M.gguf"
DEPLOY_NAMES[ocula_plus]="Qwen3-VL-2B-Ocula-Plus-Q4_K_M.gguf"
DEPLOY_NAMES[ocula_pro]="Qwen2.5-VL-7B-Ocula-Pro-Q4_K_M.gguf"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --target TARGET    Deployment target (default: local)"
    echo "                     local          → ../../models/"
    echo "                     ssh://...      → SCP to remote server"
    echo "                     http://...     → Upload via model server API"
    echo "  --model MODEL      Deploy specific model (ocula_lite|ocula_plus|ocula_pro|qwen3embed|all)"
    echo "  --quant TYPE       Preferred quantization (default: Q4_K_M)"
    echo "  --dry-run          Preview deployment without copying files"
    echo "  -h, --help         Show this help"
    exit 0
}

# ── Parse args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --quant) QUANT_PREFERENCE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Find best GGUF for deployment ──────────────────────────────
find_best_gguf() {
    local model_key=$1
    local prefix="${MODEL_MAP[$model_key]}"
    local preferred="${GGUF_DIR}/${prefix}-${QUANT_PREFERENCE}.gguf"

    if [[ -f "$preferred" ]]; then
        echo "$preferred"
        return
    fi

    # Fallback: any quantized version (not F16/F32)
    local fallback
    fallback=$(find "${GGUF_DIR}" -name "${prefix}-*.gguf" \
        ! -name "*-F16.gguf" ! -name "*-F32.gguf" \
        -print -quit 2>/dev/null || true)

    if [[ -n "$fallback" ]]; then
        echo "$fallback"
        return
    fi

    echo ""
}

# ── Deploy: Local ──────────────────────────────────────────────
deploy_local() {
    local src=$1
    local deploy_name=$2

    local dst="${LOCAL_MODELS_DIR}/${deploy_name}"
    local size_mb
    size_mb=$(du -m "$src" | cut -f1)

    echo -e "  ${CYAN}→${NC} ${src##*/} → ${dst}"
    echo -e "    Size: ${size_mb} MB"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "    ${YELLOW}[DRY RUN] Would copy${NC}"
        return 0
    fi

    # Backup existing model
    if [[ -f "$dst" ]]; then
        local backup="${dst}.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "    ${YELLOW}Backing up existing model → ${backup##*/}${NC}"
        mv "$dst" "$backup"
    fi

    mkdir -p "$LOCAL_MODELS_DIR"
    cp "$src" "$dst"
    echo -e "    ${GREEN}✓ Deployed${NC}"
}

# ── Deploy: SSH/SCP ────────────────────────────────────────────
deploy_ssh() {
    local src=$1
    local deploy_name=$2
    local ssh_target=$3  # ssh://user@host:/path

    # Parse ssh://user@host:/remote/path
    local remote="${ssh_target#ssh://}"
    local host="${remote%%:*}"
    local remote_dir="${remote#*:}"

    local dst="${host}:${remote_dir}/${deploy_name}"
    local size_mb
    size_mb=$(du -m "$src" | cut -f1)

    echo -e "  ${CYAN}→${NC} ${src##*/} → ${dst}"
    echo -e "    Size: ${size_mb} MB"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "    ${YELLOW}[DRY RUN] Would scp${NC}"
        return 0
    fi

    # Create remote directory
    # shellcheck disable=SC2029
    ssh "$host" "mkdir -p '${remote_dir}'"

    # Backup existing remote model
    # shellcheck disable=SC2029
    ssh "$host" "if [ -f '${remote_dir}/${deploy_name}' ]; then \
        mv '${remote_dir}/${deploy_name}' \
           '${remote_dir}/${deploy_name}.backup.\$(date +%Y%m%d_%H%M%S)'; fi"

    scp -q "$src" "${dst}"
    echo -e "    ${GREEN}✓ Deployed via SCP${NC}"
}

# ── Deploy: HTTP Upload ────────────────────────────────────────
deploy_http() {
    local src=$1
    local deploy_name=$2
    local api_url=$3

    local upload_url="${api_url}/api/v1/models/upload"
    local size_mb
    size_mb=$(du -m "$src" | cut -f1)

    echo -e "  ${CYAN}→${NC} ${src##*/} → ${upload_url}"
    echo -e "    Size: ${size_mb} MB"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "    ${YELLOW}[DRY RUN] Would upload${NC}"
        return 0
    fi

    local response
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -F "file=@${src}" \
        -F "name=${deploy_name}" \
        "${upload_url}" 2>&1)

    local http_code
    http_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | head -n -1)

    if [[ "$http_code" == "200" || "$http_code" == "201" ]]; then
        echo -e "    ${GREEN}✓ Uploaded (HTTP ${http_code})${NC}"
    else
        echo -e "    ${RED}✗ Upload failed (HTTP ${http_code}): ${body}${NC}"
        return 1
    fi
}

# ── Deploy dispatcher ──────────────────────────────────────────
deploy_file() {
    local src=$1
    local deploy_name=$2

    case "$TARGET" in
        local)     deploy_local "$src" "$deploy_name" ;;
        ssh://*)   deploy_ssh "$src" "$deploy_name" "$TARGET" ;;
        http://*)  deploy_http "$src" "$deploy_name" "$TARGET" ;;
        https://*) deploy_http "$src" "$deploy_name" "$TARGET" ;;
        *)
            echo -e "${RED}Unknown target type: ${TARGET}${NC}"
            echo "  Supported: local, ssh://..., http://..."
            exit 1
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ocula Model Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Target: ${CYAN}${TARGET}${NC}"
echo -e "  Quant:  ${CYAN}${QUANT_PREFERENCE}${NC}"
echo -e "  Models: ${CYAN}${MODEL}${NC}"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "  Mode:   ${YELLOW}DRY RUN${NC}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ ! -d "$GGUF_DIR" ]]; then
    echo -e "${RED}[!] GGUF directory not found: ${GGUF_DIR}${NC}"
    echo "    Run 07_quantize_gguf.py first."
    exit 1
fi

# Build target list
if [[ "$MODEL" == "all" ]]; then
    TARGETS=(ocula_lite ocula_plus ocula_pro qwen3embed)
else
    TARGETS=("$MODEL")
fi

SUCCESS=0
SKIPPED=0
FAILED=0

for model_key in "${TARGETS[@]}"; do
    echo -e "${CYAN}[$model_key]${NC}"

    gguf_path=$(find_best_gguf "$model_key")

    if [[ -z "$gguf_path" ]]; then
        echo -e "  ${YELLOW}⊘ No GGUF found for ${model_key}, skipping${NC}"
        ((SKIPPED++))
        echo ""
        continue
    fi

    deploy_name="${DEPLOY_NAMES[$model_key]}"

    if deploy_file "$gguf_path" "$deploy_name"; then
        ((SUCCESS++))
    else
        ((FAILED++))
    fi
    echo ""
done

# ── Summary ────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deployment Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${GREEN}✓ Deployed: ${SUCCESS}${NC}"
echo -e "  ${YELLOW}⊘ Skipped:  ${SKIPPED}${NC}"
if [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}✗ Failed:   ${FAILED}${NC}"
fi
echo ""

if [[ "$TARGET" == "local" ]] && [[ $SUCCESS -gt 0 ]]; then
    echo "  Deployed models in: ${LOCAL_MODELS_DIR}/"
    echo "  Rebuild the app to pick up new models:"
    echo "    cd ocula_app && flutter build ios --release"
    echo ""
fi

exit $FAILED
