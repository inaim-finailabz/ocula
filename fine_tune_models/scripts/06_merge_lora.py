#!/usr/bin/env python3
"""
LoRA Adapter Merge Script
Merges trained LoRA adapters back into base models to create full-weight checkpoints.

Supports: CUDA (PEFT merge) + MLX (mlx_lm.fuse)

Usage:
    # Merge all active models with trained adapters
    python 06_merge_lora.py

    # Merge a specific model
    python 06_merge_lora.py --model ocula_lite

    # Custom paths
    python 06_merge_lora.py --model ocula_lite \
        --base-path ../models/base/qwen3-1.7b \
        --adapter-path ../models/lora_adapters/ocula-lite-qwen3-1_7b-qlora \
        --output-path ../models/merged/ocula-lite-qwen3-1_7b-merged

    # MLX merge
    python 06_merge_lora.py --model ocula_lite --backend mlx
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ── Model registry ──────────────────────────────────────────────
MODELS = {
    "ocula_lite": {
        "config": "../configs/qwen25_1_5b_text.yaml",
        "base_path": "../models/base/qwen3-1.7b",
        "adapter_glob": "../models/lora_adapters/ocula-lite-qwen3-1_7b-*",
        "output": "../models/merged/ocula-lite-qwen3-1_7b-merged",
    },
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def detect_backend():
    """Auto-detect best available backend."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mlx"
    return "cpu"


def find_best_adapter(adapter_glob):
    """Find the most recent adapter directory matching the glob."""
    import glob
    candidates = sorted(glob.glob(adapter_glob), key=os.path.getmtime, reverse=True)
    if not candidates:
        return None
    return candidates[0]


# ─────────────────────────────────────────────────────────────────
# CUDA / PyTorch Merge
# ─────────────────────────────────────────────────────────────────

def merge_cuda(base_path, adapter_path, output_path, config):
    """Merge LoRA adapter into base model using PEFT."""
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    print(f"\n{'='*60}")
    print(f"  CUDA/PyTorch LoRA Merge")
    print(f"  Base:    {base_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")
    print(f"{'='*60}\n")

    model_type = config.get("model", {}).get("type", "llm")
    is_vlm = model_type == "vlm"

    # Load tokenizer / processor
    print("[1/4] Loading tokenizer/processor...")
    if is_vlm:
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    # Load base model
    print("[2/4] Loading base model...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    base_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"  Base model: {base_params:.1f}M parameters")

    # Load and merge LoRA
    print("[3/4] Loading and merging LoRA adapter...")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  LoRA trainable params: {lora_params:.1f}M")
    print(f"  LoRA rank: {peft_config.r}, alpha: {peft_config.lora_alpha}")

    # Merge weights — this modifies the base model in place
    model = model.merge_and_unload()
    merged_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Merged model: {merged_params:.1f}M parameters")

    # Save
    print(f"[4/4] Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    if is_vlm:
        processor.save_pretrained(output_path)

    # Copy any extra config files from adapter
    for extra in ["adapter_config.json", "training_args.json"]:
        src = os.path.join(adapter_path, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, f"original_{extra}"))

    size_gb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / 1024**3
    print(f"\n[OK] Merged model saved: {output_path} ({size_gb:.2f} GB)")
    return output_path


# ─────────────────────────────────────────────────────────────────
# MLX Merge (Apple Silicon)
# ─────────────────────────────────────────────────────────────────

def merge_mlx(base_path, adapter_path, output_path, config):
    """Merge LoRA adapter using mlx_lm.fuse."""
    print(f"\n{'='*60}")
    print(f"  MLX LoRA Fuse")
    print(f"  Base:    {base_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {output_path}")
    print(f"{'='*60}\n")

    os.makedirs(output_path, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", base_path,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--de-quantize",  # Fuse into full precision for GGUF conversion
    ]

    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] MLX fuse failed:\n{result.stderr}")
        sys.exit(1)

    print(result.stdout)

    size_gb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / 1024**3
    print(f"\n[OK] Fused model saved: {output_path} ({size_gb:.2f} GB)")
    return output_path


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────

def merge_one(model_name, backend, base_override=None, adapter_override=None,
              output_override=None):
    """Merge a single model's LoRA adapter."""
    info = MODELS[model_name]
    config = load_config(info["config"])

    base_path = base_override or info["base_path"]
    output_path = output_override or info["output"]

    # Find adapter
    if adapter_override:
        adapter_path = adapter_override
    else:
        adapter_path = find_best_adapter(info["adapter_glob"])

    if not adapter_path or not os.path.isdir(adapter_path):
        print(f"[!] No trained adapter found for {model_name}")
        print(f"    Searched: {info['adapter_glob']}")
        print(f"    Run the training script first.")
        return None

    if not os.path.isdir(base_path):
        print(f"[!] Base model not found: {base_path}")
        print(f"    Run setup_env.sh to download base models.")
        return None

    if backend == "mlx":
        return merge_mlx(base_path, adapter_path, output_path, config)
    else:
        return merge_cuda(base_path, adapter_path, output_path, config)


def main():
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base models")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to merge (default: all)")
    parser.add_argument("--backend", choices=["cuda", "mlx", "auto"], default="auto",
                        help="Compute backend (default: auto-detect)")
    parser.add_argument("--base-path", type=str, default=None,
                        help="Override base model path")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Override adapter path")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Override output path")
    args = parser.parse_args()

    backend = args.backend if args.backend != "auto" else detect_backend()
    print(f"[*] Backend: {backend}")

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = {}

    for model_name in targets:
        print(f"\n{'─'*60}")
        print(f"  Merging: {model_name}")
        print(f"{'─'*60}")
        result = merge_one(
            model_name, backend,
            base_override=args.base_path if len(targets) == 1 else None,
            adapter_override=args.adapter_path if len(targets) == 1 else None,
            output_override=args.output_path if len(targets) == 1 else None,
        )
        results[model_name] = "OK" if result else "SKIPPED"

    print(f"\n{'='*60}")
    print("  Merge Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✓" if status == "OK" else "⊘"
        print(f"  {icon} {name:15s} {status}")
    print()


if __name__ == "__main__":
    main()
