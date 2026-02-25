#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# Ocula Fine-Tuning Environment Setup
# Supports: CUDA (RTX 4070 Super Ti) + MLX (Apple Silicon M1+)
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="ocula-finetune"
PYTHON_VERSION="3.11"

echo "╔═══════════════════════════════════════════════╗"
echo "║   Ocula Fine-Tuning Environment Setup         ║"
echo "╚═══════════════════════════════════════════════╝"

# ── Detect platform ──
if [[ "$(uname -m)" == "arm64" && "$(uname -s)" == "Darwin" ]]; then
    PLATFORM="mlx"
    echo "[*] Detected Apple Silicon — will install MLX + PyTorch MPS"
elif command -v nvidia-smi &>/dev/null; then
    PLATFORM="cuda"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
    echo "[*] Detected NVIDIA GPU: $GPU_INFO"
else
    PLATFORM="cpu"
    echo "[!] No GPU detected — training will be slow. Consider using a cloud GPU."
fi

# ── Create conda environment ──
if ! command -v conda &>/dev/null; then
    echo "[!] conda not found. Install Miniforge first:"
    echo "    brew install miniforge   # macOS"
    echo "    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh  # Linux"
    exit 1
fi

echo "[*] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Install PyTorch ──
echo "[*] Installing PyTorch..."
if [[ "$PLATFORM" == "cuda" ]]; then
    # CUDA 12.4 for RTX 4070 Super Ti
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [[ "$PLATFORM" == "mlx" ]]; then
    # PyTorch with MPS backend for Apple Silicon
    pip install torch torchvision torchaudio
fi

# ── Core training dependencies ──
echo "[*] Installing training libraries..."
pip install \
    transformers>=4.46.0 \
    datasets>=3.0.0 \
    accelerate>=1.0.0 \
    peft>=0.13.0 \
    bitsandbytes>=0.44.0 \
    trl>=0.12.0 \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface-hub \
    wandb \
    tensorboard \
    evaluate \
    scikit-learn

# ── Sentence-transformers for MiniLM ──
echo "[*] Installing sentence-transformers..."
pip install sentence-transformers>=3.3.0

# ── MLX (Apple Silicon only) ──
if [[ "$PLATFORM" == "mlx" ]]; then
    echo "[*] Installing MLX framework..."
    pip install \
        mlx>=0.21.0 \
        mlx-lm>=0.20.0 \
        mlx-vlm>=0.1.0
fi

# ── GGUF conversion tools ──
echo "[*] Installing llama.cpp Python bindings..."
pip install llama-cpp-python gguf

# ── Build llama.cpp from source (for convert/quantize tools) ──
LLAMA_CPP_DIR="$SCRIPT_DIR/tools/llama.cpp"
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "[*] Cloning llama.cpp..."
    mkdir -p "$SCRIPT_DIR/tools"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
    if [[ "$PLATFORM" == "cuda" ]]; then
        cmake -B build -DGGML_CUDA=ON
    elif [[ "$PLATFORM" == "mlx" ]]; then
        cmake -B build -DGGML_METAL=ON
    else
        cmake -B build
    fi
    cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    cd "$SCRIPT_DIR"
else
    echo "[*] llama.cpp already present at $LLAMA_CPP_DIR"
fi

# ── Install llama.cpp Python convert requirements ──
pip install -r "$LLAMA_CPP_DIR/requirements.txt" 2>/dev/null || true

# ── Download base models ──
echo ""
echo "[*] Downloading base models to models/base/..."
python -c "
from huggingface_hub import snapshot_download
import os

base_dir = os.path.join('$SCRIPT_DIR', 'models', 'base')
os.makedirs(base_dir, exist_ok=True)

models = {
    'qwen3-vl-2b': 'Qwen/Qwen3-VL-2B-Instruct',
    'qwen2.5-1.5b-instruct': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-vl-7b-instruct': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'qwen3-embedding-0.6b': 'Qwen/Qwen3-Embedding-0.6B',
}

for name, repo_id in models.items():
    target = os.path.join(base_dir, name)
    if os.path.exists(target):
        print(f'  [skip] {name} already downloaded')
        continue
    print(f'  [download] {repo_id} → {target}')
    try:
        snapshot_download(repo_id, local_dir=target, ignore_patterns=['*.bin', '*.pt'])
    except Exception as e:
        print(f'  [warn] Failed to download {name}: {e}')
        print(f'         You can download manually: huggingface-cli download {repo_id} --local-dir {target}')
"

# ── Verify installation ──
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║   Verification                                ║"
echo "╚═══════════════════════════════════════════════╝"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')

import transformers, peft, trl, datasets
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'Datasets: {datasets.__version__}')

try:
    import mlx
    print(f'MLX: {mlx.__version__}')
except ImportError:
    pass

import sentence_transformers
print(f'Sentence-Transformers: {sentence_transformers.__version__}')
"

echo ""
echo "[OK] Environment ready. Activate with: conda activate $ENV_NAME"
echo "[OK] Next: prepare your data in data/raw/, then run:"
echo "     python scripts/01_prepare_data.py"
