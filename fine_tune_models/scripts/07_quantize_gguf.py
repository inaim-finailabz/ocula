#!/usr/bin/env python3
"""
GGUF Quantization Script
Converts merged HuggingFace models → GGUF format for llama.cpp inference.

Uses llama.cpp's convert_hf_to_gguf.py + llama-quantize binary.

Usage:
    # Quantize all merged models
    python 07_quantize_gguf.py

    # Quantize a specific model
    python 07_quantize_gguf.py --model smolvlm2
    python 07_quantize_gguf.py --model moondream2
    python 07_quantize_gguf.py --model qwen3vl

    # Custom quantization type
    python 07_quantize_gguf.py --model smolvlm2 --quant-types Q4_K_M Q8_0

    # Skip conversion (already have F16 GGUF)
    python 07_quantize_gguf.py --model smolvlm2 --quantize-only
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ── Paths ────────────────────────────────────────────────────────
LLAMA_CPP_DIR = Path("../../llama.cpp")
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

GGUF_OUTPUT_DIR = Path("../models/gguf")

# ── Model registry ──────────────────────────────────────────────
MODELS = {
    "smolvlm2": {
        "config": "../configs/smolvlm2.yaml",
        "merged_path": "../models/merged/smolvlm2-merged",
        "gguf_name": "SmolVLM2-500M-finetuned",
    },
    "moondream2": {
        "config": "../configs/moondream2.yaml",
        "merged_path": "../models/merged/moondream2-merged",
        "gguf_name": "moondream2-text-model-finetuned",
    },
    "qwen3vl": {
        "config": "../configs/qwen3vl.yaml",
        "merged_path": "../models/merged/qwen3vl-merged",
        "gguf_name": "Qwen3VL-2B-Thinking-finetuned",
    },
    "minilm": {
        "config": "../configs/minilm.yaml",
        "merged_path": "../models/base/minilm-l6-v2-finetuned",
        "gguf_name": "minilm-l6-v2-finetuned",
    },
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_prerequisites():
    """Verify llama.cpp tools are available."""
    errors = []
    if not CONVERT_SCRIPT.exists():
        errors.append(f"  convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}")
    if not QUANTIZE_BIN.exists():
        errors.append(f"  llama-quantize binary not found at {QUANTIZE_BIN}")
    if errors:
        print("[!] Prerequisites missing:")
        for e in errors:
            print(e)
        print("\n    Run: cd ../../llama.cpp && mkdir -p build && cd build && "
              "cmake .. && cmake --build . --target llama-quantize -j")
        sys.exit(1)
    print(f"[OK] llama.cpp tools found")
    print(f"  convert:  {CONVERT_SCRIPT}")
    print(f"  quantize: {QUANTIZE_BIN}")


# ─────────────────────────────────────────────────────────────────
# Step 1: HuggingFace → F16 GGUF
# ─────────────────────────────────────────────────────────────────

def convert_to_f16_gguf(merged_path, output_name):
    """Convert HuggingFace safetensors model to F16 GGUF."""
    f16_path = GGUF_OUTPUT_DIR / f"{output_name}-F16.gguf"

    if f16_path.exists():
        size_gb = f16_path.stat().st_size / 1024**3
        print(f"[*] F16 GGUF already exists: {f16_path} ({size_gb:.2f} GB)")
        return f16_path

    print(f"\n[*] Converting HF → F16 GGUF...")
    print(f"  Input:  {merged_path}")
    print(f"  Output: {f16_path}")

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(merged_path),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ]

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Conversion failed:\n{result.stderr}")
        # Try with --vocab-only fallback for unusual architectures
        if "unknown model architecture" in result.stderr.lower():
            print("[*] Retrying with --verbose flag...")
            cmd.append("--verbose")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[!] Retry also failed:\n{result.stderr}")
                return None

    if result.stdout:
        # Print just the last few lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  {line}")

    size_gb = f16_path.stat().st_size / 1024**3
    print(f"[OK] F16 GGUF created: {f16_path} ({size_gb:.2f} GB)")
    return f16_path


# ─────────────────────────────────────────────────────────────────
# Step 2: F16 GGUF → Quantized GGUF
# ─────────────────────────────────────────────────────────────────

def quantize_gguf(f16_path, output_name, quant_type):
    """Quantize F16 GGUF to a specific quantization type."""
    quant_path = GGUF_OUTPUT_DIR / f"{output_name}-{quant_type}.gguf"

    if quant_path.exists():
        size_gb = quant_path.stat().st_size / 1024**3
        print(f"[*] {quant_type} GGUF already exists: {quant_path} ({size_gb:.2f} GB)")
        return quant_path

    print(f"\n[*] Quantizing → {quant_type}...")
    print(f"  Input:  {f16_path}")
    print(f"  Output: {quant_path}")

    cmd = [
        str(QUANTIZE_BIN),
        str(f16_path),
        str(quant_path),
        quant_type,
    ]

    # Add importance matrix for K-quants if available
    imatrix = GGUF_OUTPUT_DIR / f"{output_name}-imatrix.dat"
    if imatrix.exists() and "_K_" in quant_type:
        cmd.extend(["--imatrix", str(imatrix)])
        print(f"  Using importance matrix: {imatrix}")

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Quantization failed:\n{result.stderr}")
        return None

    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  {line}")

    size_gb = quant_path.stat().st_size / 1024**3
    f16_gb = f16_path.stat().st_size / 1024**3
    ratio = size_gb / f16_gb * 100

    print(f"[OK] {quant_type}: {quant_path} ({size_gb:.2f} GB, {ratio:.0f}% of F16)")
    return quant_path


# ─────────────────────────────────────────────────────────────────
# Step 3: Embedding model → GGUF (sentence-transformers)
# ─────────────────────────────────────────────────────────────────

def convert_embedding_to_gguf(model_path, output_name):
    """Convert sentence-transformers model to GGUF for llama.cpp embedding."""
    f32_path = GGUF_OUTPUT_DIR / f"{output_name}-F32.gguf"

    if f32_path.exists():
        size_mb = f32_path.stat().st_size / 1024**2
        print(f"[*] Embedding GGUF already exists: {f32_path} ({size_mb:.1f} MB)")
        return f32_path

    print(f"\n[*] Converting embedding model → GGUF...")
    print(f"  Input:  {model_path}")
    print(f"  Output: {f32_path}")

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

    # sentence-transformers stores model inside a subdirectory
    actual_path = model_path
    if (Path(model_path) / "1_Pooling").exists():
        # It's a sentence-transformers format, convert via the HF path
        pass

    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(actual_path),
        "--outfile", str(f32_path),
        "--outtype", "f32",  # Embeddings need precision
    ]

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Embedding conversion failed:\n{result.stderr}")
        print("[*] Trying with safetensors...")
        return None

    size_mb = f32_path.stat().st_size / 1024**2
    print(f"[OK] Embedding GGUF created: {f32_path} ({size_mb:.1f} MB)")
    return f32_path


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────

def quantize_one(model_name, quant_types=None, quantize_only=False):
    """Quantize a single model."""
    info = MODELS[model_name]
    config = load_config(info["config"])

    merged_path = info["merged_path"]
    output_name = info["gguf_name"]

    # Get quantization types from config or CLI override
    if quant_types is None:
        quant_types = config.get("quantization", {}).get("methods", ["Q4_K_M"])

    if not os.path.isdir(merged_path):
        print(f"[!] Merged model not found: {merged_path}")
        print(f"    Run 06_merge_lora.py --model {model_name} first.")
        return {}

    is_embedding = config.get("model", {}).get("type") == "embedding"
    results = {}

    if is_embedding:
        # Embedding models: F32 → Q8_0 only
        f32_path = convert_embedding_to_gguf(merged_path, output_name)
        if f32_path and "Q8_0" in quant_types:
            q8_path = quantize_gguf(f32_path, output_name, "Q8_0")
            results["Q8_0"] = str(q8_path) if q8_path else "FAILED"
        results["F32"] = str(f32_path) if f32_path else "FAILED"
    else:
        # Language / VLM models: HF → F16 → quantized
        if not quantize_only:
            f16_path = convert_to_f16_gguf(merged_path, output_name)
        else:
            f16_path = GGUF_OUTPUT_DIR / f"{output_name}-F16.gguf"
            if not f16_path.exists():
                print(f"[!] F16 GGUF not found: {f16_path}")
                return {}

        if f16_path:
            for qt in quant_types:
                qpath = quantize_gguf(f16_path, output_name, qt)
                results[qt] = str(qpath) if qpath else "FAILED"

    return results


def main():
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(description="Convert and quantize models to GGUF")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to quantize (default: all)")
    parser.add_argument("--quant-types", nargs="+", default=None,
                        help="Quantization types (e.g., Q4_K_M Q8_0 Q5_K_M)")
    parser.add_argument("--quantize-only", action="store_true",
                        help="Skip HF→F16 conversion, just quantize existing F16 GGUF")
    parser.add_argument("--skip-prerequisites", action="store_true",
                        help="Skip llama.cpp tool checks")
    args = parser.parse_args()

    if not args.skip_prerequisites:
        check_prerequisites()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_name in targets:
        print(f"\n{'─'*60}")
        print(f"  Quantizing: {model_name}")
        print(f"{'─'*60}")
        results = quantize_one(model_name, args.quant_types, args.quantize_only)
        all_results[model_name] = results

    # Summary
    print(f"\n{'='*60}")
    print("  Quantization Summary")
    print(f"{'='*60}")
    for model_name, results in all_results.items():
        print(f"\n  {model_name}:")
        if not results:
            print(f"    ⊘ SKIPPED (no merged model)")
        for qt, path in results.items():
            if path == "FAILED":
                print(f"    ✗ {qt}: FAILED")
            else:
                size = Path(path).stat().st_size / 1024**3
                unit = "GB" if size >= 0.1 else "MB"
                val = size if unit == "GB" else size * 1024
                print(f"    ✓ {qt}: {path} ({val:.2f} {unit})")
    print()

    # List final GGUF files
    if GGUF_OUTPUT_DIR.exists():
        ggufs = sorted(GGUF_OUTPUT_DIR.glob("*-finetuned-*.gguf"))
        if ggufs:
            print(f"  Ready for deployment ({len(ggufs)} files):")
            for g in ggufs:
                size_gb = g.stat().st_size / 1024**3
                print(f"    {g.name:50s} {size_gb:.2f} GB")
            print()


if __name__ == "__main__":
    main()
