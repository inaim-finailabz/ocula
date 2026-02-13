# Ocula Model Fine-Tuning Pipeline

Fine-tune all Ocula on-device models with LoRA/QLoRA, then quantize to GGUF for deployment.

## Models

| Model | Params | Method | VRAM (CUDA) | RAM (MLX) |
|-------|--------|--------|-------------|-----------|
| SmolVLM2-500M | 500M | Full / LoRA | ~4 GB | ~3 GB |
| Moondream 2 | 1.8B | QLoRA (4-bit) | ~8 GB | ~6 GB |
| Qwen3-VL-2B | 2B | QLoRA (4-bit) | ~10 GB | ~8 GB |
| MiniLM-L6-v2 | 22M | Full fine-tune | ~2 GB | ~1 GB |

RTX 4070 Super Ti (16 GB VRAM) can handle all models. M1 (16 GB) can handle all via MLX.

## Directory Structure

```
fine_tune_models/
├── README.md                  # This file
├── setup_env.sh               # One-command environment setup
├── run_pipeline.sh            # Full MLOps pipeline orchestrator
├── configs/
│   ├── smolvlm2.yaml          # SmolVLM2-500M training config
│   ├── moondream2.yaml        # Moondream 2 training config
│   ├── qwen3vl.yaml           # Qwen3-VL-2B training config
│   └── minilm.yaml            # MiniLM embedding training config
├── scripts/
│   ├── 01_prepare_data.py     # Convert raw data → training format
│   ├── 02_train_smolvlm2.py   # Fine-tune SmolVLM2-500M
│   ├── 03_train_moondream2.py  # Fine-tune Moondream 2
│   ├── 04_train_qwen3vl.py    # Fine-tune Qwen3-VL-2B
│   ├── 05_train_minilm.py     # Fine-tune MiniLM-L6-v2
│   ├── 06_merge_lora.py       # Merge LoRA adapters → full model
│   ├── 07_quantize_gguf.py    # Convert to GGUF + quantize
│   ├── 08_evaluate.py         # Benchmark before/after
│   └── 09_deploy.sh           # Copy GGUF to model server
├── data/
│   ├── raw/                   # Your raw training data (JSON, CSV, images)
│   ├── processed/             # Formatted datasets ready for training
│   └── eval/                  # Evaluation/test sets
├── models/
│   ├── base/                  # Downloaded base models (HF format)
│   ├── lora_adapters/         # Trained LoRA weights
│   ├── merged/                # Full merged models (base + LoRA)
│   └── gguf/                  # Final quantized GGUF files for deployment
├── logs/                      # Training logs, TensorBoard, W&B
└── notebooks/                 # Jupyter notebooks for experimentation
```

## Quick Start

```bash
# 1. Setup environment (creates conda env with CUDA + MLX support)
./setup_env.sh

# 2. Prepare your data
python scripts/01_prepare_data.py --input data/raw/ --output data/processed/

# 3. Run full pipeline (all models)
./run_pipeline.sh --all

# Or train individual models:
./run_pipeline.sh --model smolvlm2
./run_pipeline.sh --model moondream2
./run_pipeline.sh --model qwen3vl
./run_pipeline.sh --model minilm

# 4. Deploy to model server
./scripts/09_deploy.sh --target http://192.168.3.14:8080
```

## Data Format

See `scripts/01_prepare_data.py` for details. Training data should be in:

**Chat/RAG models** (SmolVLM2, Moondream, Qwen3):
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Vision models** (with images):
```json
{"messages": [...], "images": ["path/to/image.jpg"]}
```

**Embedding model** (MiniLM):
```json
{"query": "search query", "positive": "relevant passage", "negative": "irrelevant passage"}
```

## Pipeline Flow

```
Raw Data → Prepare → Train (LoRA) → Merge → Quantize (GGUF) → Evaluate → Deploy
                         ↓
                    LoRA Adapters → Merged Model → Q4_K_M / Q8_0 GGUF
```

## Script Reference

| # | Script | Purpose |
|---|--------|---------|
| 01 | `01_prepare_data.py` | Download & convert datasets to ChatML/triplet format |
| 02 | `02_train_smolvlm2.py` | Fine-tune SmolVLM2-500M (LoRA, CUDA+MLX) |
| 03 | `03_train_moondream2.py` | Fine-tune Moondream 2 (QLoRA 4-bit, CUDA+MLX) |
| 04 | `04_train_qwen3vl.py` | Fine-tune Qwen3-VL-2B (QLoRA 4-bit, CUDA+MLX) |
| 05 | `05_train_minilm.py` | Fine-tune MiniLM-L6-v2 embeddings (full) |
| 06 | `06_merge_lora.py` | Merge LoRA adapters into base model weights |
| 07 | `07_quantize_gguf.py` | Convert merged models → GGUF via llama.cpp |
| 08 | `08_evaluate.py` | Benchmark perplexity, speed, quality, retrieval |
| 09 | `09_deploy.sh` | Deploy GGUF to local/SSH/HTTP model server |

## Orchestrator

```bash
# Full pipeline — all 4 models end-to-end
./run_pipeline.sh --all

# Single model
./run_pipeline.sh --model qwen3vl --backend mlx

# Resume from a specific stage
./run_pipeline.sh --model moondream2 --from merge

# Cherry-pick stages
./run_pipeline.sh --stages train,merge,quantize --model smolvlm2

# Preview without executing
./run_pipeline.sh --all --dry-run
```
