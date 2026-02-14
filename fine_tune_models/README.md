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
│   ├── 02_train_smolvlm2.py   # Fine-tune SmolVLM2-500M (text-only)
│   ├── 02a_prepare_vision_data.py  # ★ Download vision+document datasets
│   ├── 02b_train_smolvlm2_vision.py  # ★ Multimodal VLM fine-tuning
│   ├── 03a_prepare_moondream_data.py  # ★ Moondream2 vision data prep
│   ├── 03b_train_moondream2_vision.py  # ★ Moondream2 multimodal SFT
│   ├── 04a_prepare_qwen3vl_data.py  # ★ Qwen3-VL vision data prep
│   ├── 04b_train_qwen3vl_vision.py  # ★ Qwen3-VL multimodal SFT
│   ├── 07b_export_moondream2_gguf.py  # ★ Export fine-tuned Moondream2 GGUF
│   ├── 07c_export_qwen3vl_gguf.py  # ★ Export fine-tuned Qwen3-VL GGUF
│   ├── 03_train_moondream2.py  # Fine-tune Moondream 2
│   ├── 04_train_qwen3vl.py    # Fine-tune Qwen3-VL-2B
│   ├── 05_train_minilm.py     # Fine-tune MiniLM-L6-v2
│   ├── 06_merge_lora.py       # Merge LoRA adapters → full model
│   ├── 07_quantize_gguf.py    # Convert to GGUF + quantize
│   ├── 07a_export_smolvlm2_gguf.py  # ★ Export fine-tuned SmolVLM2 GGUF
│   ├── 08_evaluate.py         # Benchmark before/after
│   └── 09_deploy.sh           # Copy GGUF to model server
├── run_vision_finetune.sh     # ★ SmolVLM2 vision fine-tune pipeline
├── run_moondream_finetune.sh  # ★ Moondream2 vision fine-tune pipeline
├── run_qwen3vl_finetune.sh    # ★ Qwen3-VL vision fine-tune pipeline
├── data/
│   ├── raw/                   # Your raw training data (JSON, CSV, images)
│   ├── processed/             # Formatted datasets ready for training
│   ├── vision/                # ★ Vision training data (images + JSONL)
│   │   ├── images/            #   Downloaded images by dataset
│   │   ├── smolvlm2_vision_train.jsonl  # SmolVLM2 train set
│   │   ├── smolvlm2_vision_val.jsonl    # SmolVLM2 val set
│   │   ├── moondream2_vision_train.jsonl  # Moondream2 train set
│   │   ├── moondream2_vision_val.jsonl    # Moondream2 val set
│   │   ├── qwen3vl_vision_train.jsonl     # Qwen3-VL train set
│   │   └── qwen3vl_vision_val.jsonl       # Qwen3-VL val set
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

## 🔥 Vision Fine-Tuning (Image + Document)


All three VLMs (SmolVLM2, Moondream2, Qwen3-VL) support dedicated vision fine-tuning pipelines to improve image understanding and document summarization with real image+text training data.

### Model Comparison

| Model         | Params | Method         | VRAM (CUDA) | RAM (MLX) | Training Time (50K) |
|---------------|--------|---------------|-------------|-----------|---------------------|
| SmolVLM2-500M | 500M   | LoRA/QLoRA    | ~4 GB       | ~3 GB     | ~6–8h (MLX)         |
| Moondream2    | 1.8B   | QLoRA (4-bit) | ~8 GB       | ~6 GB     | ~8–10h (CUDA)       |
| Qwen3-VL-2B   | 2B     | QLoRA (4-bit) | ~10 GB      | ~8 GB     | ~10–12h (CUDA)      |

---

## SmolVLM2 Vision Fine-Tuning


#### Quick Start (Mac MLX)

```bash
# One-command pipeline: download data → train → merge → GGUF
./run_vision_finetune.sh --mlx

# Or step-by-step:
cd scripts
python 02a_prepare_vision_data.py              # Download 7 vision/doc datasets
python 02b_train_smolvlm2_vision.py --backend mlx  # LoRA fine-tune on Apple Silicon
python 06_merge_lora.py --model smolvlm2       # Merge LoRA → full model
python 07a_export_smolvlm2_gguf.py --deploy    # GGUF + copy to app
```

#### Quick Start (CUDA)

```bash
./run_vision_finetune.sh --cuda
# Or with 4-bit quantized training (saves VRAM):
./run_vision_finetune.sh --cuda --4bit
```

---

## Moondream2 Vision Fine-Tuning

#### Quick Start

```bash
# One-command pipeline: download data → train → merge → GGUF
./run_moondream_finetune.sh --cuda

# Or step-by-step:
cd scripts
python 03a_prepare_moondream_data.py           # Prepare Moondream2 vision data
python 03b_train_moondream2_vision.py --backend cuda  # QLoRA fine-tune
python 06_merge_lora.py --model moondream2     # Merge LoRA → full model
python 07b_export_moondream2_gguf.py --deploy  # GGUF + copy to app
```

---

## Qwen3-VL Vision Fine-Tuning

#### Quick Start

```bash
# One-command pipeline: download data → train → merge → GGUF
./run_qwen3vl_finetune.sh --cuda

# Or step-by-step:
cd scripts
python 04a_prepare_qwen3vl_data.py             # Prepare Qwen3-VL vision data
python 04b_train_qwen3vl_vision.py --backend cuda  # QLoRA fine-tune
python 06_merge_lora.py --model qwen3vl        # Merge LoRA → full model
python 07c_export_qwen3vl_gguf.py --deploy     # GGUF + copy to app
```

### Training Datasets

| Dataset | Task | Examples | What it teaches |
|---------|------|----------|-----------------|
| VQAv2 | Image QA | 10K | Answer questions about image content |
| TextCaps | OCR + Caption | 10K | Read text in images |
| Paragraph Captions | Description | 10K | Long-form image descriptions |
| DocVQA | Document QA | 10K | Read forms, invoices, reports |
| ChartQA | Chart QA | 10K | Interpret charts and graphs |
| InfoVQA | Infographic QA | 10K | Understand infographics |
| AI2D | Diagram QA | 10K | Science diagram understanding |
| CNN/DailyMail | Summarization | 5K | Text document summarisation |

### Hardware Requirements

| Backend | Hardware | Training time (50K samples) |
|---------|----------|---------------------------|
| MLX | M1 Pro 16GB | ~6–8 hours |
| MLX | M2 Ultra 192GB | ~1–2 hours |
| CUDA | RTX 4070 16GB | ~2–3 hours |
| CUDA + 4bit | RTX 3060 12GB | ~4–5 hours |
| MPS | M1 16GB (fallback) | ~10–12 hours |

### New Scripts

| Script | Purpose |
|--------|---------|
| `02a_prepare_vision_data.py` | Download & convert 7 vision/document datasets (SmolVLM2) |
| `02b_train_smolvlm2_vision.py` | Multimodal LoRA SFT (SmolVLM2) |
| `07a_export_smolvlm2_gguf.py` | Export fine-tuned SmolVLM2 GGUF + mmproj |
| `03a_prepare_moondream_data.py` | Prepare Moondream2 vision data |
| `03b_train_moondream2_vision.py` | Multimodal QLoRA SFT (Moondream2) |
| `07b_export_moondream2_gguf.py` | Export fine-tuned Moondream2 GGUF + mmproj |
| `04a_prepare_qwen3vl_data.py` | Prepare Qwen3-VL vision data |
| `04b_train_qwen3vl_vision.py` | Multimodal QLoRA SFT (Qwen3-VL) |
| `07c_export_qwen3vl_gguf.py` | Export fine-tuned Qwen3-VL GGUF + mmproj |
| `run_vision_finetune.sh` | SmolVLM2 vision pipeline orchestrator |
| `run_moondream_finetune.sh` | Moondream2 vision pipeline orchestrator |
| `run_qwen3vl_finetune.sh` | Qwen3-VL vision pipeline orchestrator |

---

## Script Reference

| # | Script | Purpose |
|---|--------|---------|
| 01 | `01_prepare_data.py` | Download & convert datasets to ChatML/triplet format |
| 02 | `02_train_smolvlm2.py` | Fine-tune SmolVLM2-500M text-only (LoRA, CUDA+MLX) |
| 02a | `02a_prepare_vision_data.py` | Download vision+document training data (SmolVLM2) |
| 02b | `02b_train_smolvlm2_vision.py` | Multimodal VLM fine-tuning (SmolVLM2) |
| 03a | `03a_prepare_moondream_data.py` | Prepare Moondream2 vision data |
| 03b | `03b_train_moondream2_vision.py` | Multimodal VLM fine-tuning (Moondream2) |
| 04a | `04a_prepare_qwen3vl_data.py` | Prepare Qwen3-VL vision data |
| 04b | `04b_train_qwen3vl_vision.py` | Multimodal VLM fine-tuning (Qwen3-VL) |
| 07a | `07a_export_smolvlm2_gguf.py` | Export fine-tuned SmolVLM2 GGUF + mmproj |
| 07b | `07b_export_moondream2_gguf.py` | Export fine-tuned Moondream2 GGUF + mmproj |
| 07c | `07c_export_qwen3vl_gguf.py` | Export fine-tuned Qwen3-VL GGUF + mmproj |
| run_vision_finetune.sh | SmolVLM2 vision pipeline orchestrator |
| run_moondream_finetune.sh | Moondream2 vision pipeline orchestrator |
| run_qwen3vl_finetune.sh | Qwen3-VL vision pipeline orchestrator |
| 05 | `05_train_minilm.py` | Fine-tune MiniLM-L6-v2 embeddings (full) |
| 06 | `06_merge_lora.py` | Merge LoRA adapters into base model weights |
| 07 | `07_quantize_gguf.py` | Convert merged models → GGUF via llama.cpp |
| 07a | `07a_export_smolvlm2_gguf.py` | **NEW** — Export fine-tuned SmolVLM2 GGUF + mmproj |
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
