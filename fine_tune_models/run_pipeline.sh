#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="full"
CYCLES=2
COMPRESS="balanced"
MAX_SAMPLES=10000
SKIP_SETUP=false
SKIP_DATA=false

usage() {
  cat <<EOF
Ocula CUDA Pipeline

Usage:
  ./run_pipeline.sh [OPTIONS]

Options:
  --mode MODE          init|continuous|full (default: full)
  --cycles N           Continuous refinement cycles (default: 2)
  --compress PROFILE   balanced|aggressive|extreme (default: balanced)
  --max-samples N      Vision sample cap per run (default: 10000)
  --skip-setup         Skip environment setup
  --skip-data          Skip dataset download/prep
  -h, --help           Show help

Examples:
  ./run_pipeline.sh
  ./run_pipeline.sh --mode init --compress balanced
  ./run_pipeline.sh --mode continuous --cycles 4 --compress aggressive
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --compress) COMPRESS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=true; shift ;;
    --skip-data) SKIP_DATA=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if ! $SKIP_SETUP; then
  ./setup_env.sh
fi

if ! $SKIP_DATA; then
  PYTHON="${SCRIPT_DIR}/ocula_env/bin/python3"
  [[ -x "$PYTHON" ]] || PYTHON="python3"
  "$PYTHON" scripts/01_prepare_data.py --download-datasets --model ocula_lite
  "$PYTHON" scripts/01_prepare_data.py --download-datasets --model ocula_plus
  "$PYTHON" scripts/01_prepare_data.py --download-datasets --model ocula_pro
  "$PYTHON" scripts/01_prepare_data.py --download-datasets --model qwen3embed
fi

if [[ "$MODE" == "init" || "$MODE" == "full" ]]; then
  ./run_ocula_family.sh --tier lite --backend cuda --compress "$COMPRESS"
  ./run_ocula_family.sh --tier plus --backend cuda --4bit --compress "$COMPRESS" --max-samples "$MAX_SAMPLES"
  ./run_ocula_family.sh --tier pro --backend cuda --4bit --compress "$COMPRESS" --max-samples "$MAX_SAMPLES"
  ./run_ocula_family.sh --tier embed --backend cuda
fi

if [[ "$MODE" == "continuous" || "$MODE" == "full" ]]; then
  PYTHON="${SCRIPT_DIR}/ocula_env/bin/python3"
  [[ -x "$PYTHON" ]] || PYTHON="python3"
  (
    cd scripts
    "$PYTHON" 10_continuous_mlops.py \
      --cycles "$CYCLES" \
      --backend cuda \
      --compress "$COMPRESS" \
      --max-samples "$MAX_SAMPLES"
  )
fi

echo ""
echo "[OK] Ocula CUDA pipeline finished (mode=${MODE})"
