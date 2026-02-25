# Ocula Fine-Tuning (CUDA Server)

This folder is now organized around the active Ocula family only:

- `Ocula Lite`  → `Qwen2.5-1.5B-Instruct` (text)
- `Ocula Plus`  → `Qwen3-VL-2B-Instruct` (vision)
- `Ocula Pro`   → `Qwen2.5-VL-7B-Instruct` (vision reasoning)
- `Embed`       → `Qwen3-Embedding-0.6B` (RAG embeddings)

## One Command (Fully Automated)

From `fine_tune_models/`:

```bash
./run_pipeline.sh
# or
./run_cuda_server.sh
```

This does:
1. Environment setup
2. Dataset download/prep for active Ocula models
3. Initial Ocula family build (`lite/plus/pro/embed`)
4. Continuous refinement cycles with benchmark/promotion gates

## Common Modes

```bash
# Initial families only
./run_pipeline.sh --mode init

# Continuous refinement only
./run_pipeline.sh --mode continuous --cycles 4

# Lower memory profile
./run_pipeline.sh --compress aggressive
```

## Compression Profiles

- `balanced`: `Q4_K_M`, `Q5_K_M` (recommended)
- `aggressive`: `Q3_K_M`, `Q4_0`
- `extreme`: `Q2_K`, `Q3_K_S`

## Key Files

- Entry point: `run_pipeline.sh`
- Family trainer: `run_ocula_family.sh`
- Continuous loop: `scripts/10_continuous_mlops.py`
- Data prep: `scripts/01_prepare_data.py`
- Quantization: `scripts/07_quantize_gguf.py`
- Deploy: `scripts/09_deploy.sh`

## Continuous State

Promotion state/history is saved to:

- `logs/mlops/state.json`
