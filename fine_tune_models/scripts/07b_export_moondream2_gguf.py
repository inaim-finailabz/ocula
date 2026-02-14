#!/usr/bin/env python3
"""
Moondream2 GGUF Export
======================
Converts the fine-tuned Moondream2 (HuggingFace format) → GGUF for llama.cpp.

Handles:
  • CUDA path: merged HF model → convert_hf_to_gguf.py → llama-quantize
  • MLX path:  fused MLX model → convert to HF → convert_hf_to_gguf.py → quantize

The vision projector (mmproj) is exported separately since llama.cpp
loads the text model and vision encoder as two files.

Usage:
    python 07b_export_moondream2_gguf.py
    python 07b_export_moondream2_gguf.py --from-mlx ../models/merged/moondream2-vision-merged
    python 07b_export_moondream2_gguf.py --quant-types Q4_K_M Q8_0
    python 07b_export_moondream2_gguf.py --deploy
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
MERGED_MODEL_DIR = Path("../models/merged/moondream2-vision-merged")

DEFAULT_QUANT_TYPES = ["Q4_K_M", "Q8_0"]


def check_prerequisites():
    errors = []
    if not CONVERT_SCRIPT.exists():
        errors.append(f"  convert_hf_to_gguf.py not found: {CONVERT_SCRIPT}")
        errors.append(f"  → cd ../../llama.cpp && git pull")
    if not QUANTIZE_BIN.exists():
        errors.append(f"  llama-quantize not found: {QUANTIZE_BIN}")
        errors.append(f"  → cd ../../llama.cpp && mkdir -p build && cd build && cmake .. && make -j")
    if errors:
        print("[!] Missing prerequisites:")
        for e in errors:
            print(e)
        sys.exit(1)
    print("[OK] llama.cpp tools found")


def convert_hf_to_gguf(model_dir: Path, output_path: Path):
    print(f"\n[*] Converting HF → GGUF F16")
    print(f"    Input:  {model_dir}")
    print(f"    Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]

    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Conversion failed:")
        print(result.stderr[-2000:] if result.stderr else "No error output")
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    [OK] F16 GGUF: {output_path} ({size_mb:.1f} MB)")
    return True


def quantize_gguf(input_path: Path, output_path: Path, quant_type: str = "Q4_K_M"):
    print(f"\n[*] Quantizing F16 → {quant_type}")
    print(f"    Input:  {input_path}")
    print(f"    Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(QUANTIZE_BIN),
        str(input_path),
        str(output_path),
        quant_type,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Quantization failed:")
        print(result.stderr[-2000:] if result.stderr else "No error output")
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    [OK] {quant_type}: {output_path} ({size_mb:.1f} MB)")
    return True


def export_projector(model_dir: Path, output_path: Path):
    """
    Export the vision projector (mmproj) as a separate GGUF file.

    Moondream2 uses a SigLIP vision encoder. The projector maps vision
    embeddings into the language model's embedding space.
    """
    print(f"\n[*] Exporting vision projector (mmproj)")

    mmproj_script = LLAMA_CPP_DIR / "examples" / "llava" / "convert_hf_to_gguf_llava.py"
    alt_mmproj_script = LLAMA_CPP_DIR / "tools" / "mtmd" / "convert_hf_to_mtmd_gguf.py"

    if alt_mmproj_script.exists():
        cmd = [
            sys.executable,
            str(alt_mmproj_script),
            str(model_dir),
            "--outfile", str(output_path),
        ]
    elif mmproj_script.exists():
        cmd = [
            sys.executable,
            str(mmproj_script),
            str(model_dir),
            "--mmproj-outfile", str(output_path),
        ]
    else:
        print("  [WARN] No mmproj converter found in llama.cpp")
        print(f"  Checked: {mmproj_script}")
        print(f"  Checked: {alt_mmproj_script}")
        print("  → Use the base mmproj GGUF:")
        print(f"    cp ../../models/mmproj-moondream2-*.gguf {output_path}")
        return False

    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [WARN] Projector export failed — using base projector")
        print(result.stderr[-1000:] if result.stderr else "")
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    [OK] mmproj: {output_path} ({size_mb:.1f} MB)")
    return True


def deploy_to_app(gguf_path: Path, mmproj_path: Path, app_models_dir: Path):
    """Copy GGUFs to the Ocula app's model directories."""
    print(f"\n[*] Deploying to app...")

    # Moondream2 is plus tier → goes into server downloadable
    server_dir = app_models_dir.parent / "models_server"
    server_dir.mkdir(parents=True, exist_ok=True)

    target_model = server_dir / gguf_path.name
    target_mmproj = server_dir / mmproj_path.name

    print(f"    Model:   {gguf_path} → {target_model}")
    shutil.copy2(gguf_path, target_model)
    if mmproj_path.exists():
        print(f"    mmproj:  {mmproj_path} → {target_mmproj}")
        shutil.copy2(mmproj_path, target_mmproj)

    # Also place in main models dir for local serving
    app_models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gguf_path, app_models_dir / gguf_path.name)
    if mmproj_path.exists():
        shutil.copy2(mmproj_path, app_models_dir / mmproj_path.name)

    print(f"    [OK] Deployed to {server_dir} and {app_models_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned Moondream2 to GGUF for llama.cpp")
    parser.add_argument("--model-dir", type=Path, default=MERGED_MODEL_DIR)
    parser.add_argument("--from-mlx", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=GGUF_OUTPUT_DIR)
    parser.add_argument("--quant-types", nargs="+", default=DEFAULT_QUANT_TYPES)
    parser.add_argument("--projector-only", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--skip-projector", action="store_true")
    args = parser.parse_args()

    check_prerequisites()

    model_dir = args.from_mlx or args.model_dir

    if not model_dir.exists():
        print(f"[!] Model directory not found: {model_dir}")
        print("    Run these first:")
        print("    1. python 03b_train_moondream2_vision.py")
        print("    2. python 06_merge_lora.py --model moondream2")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = "moondream2-finetuned"

    if not args.projector_only:
        # Step 1: Convert to F16 GGUF
        f16_path = output_dir / f"{base_name}-F16.gguf"
        if not convert_hf_to_gguf(model_dir, f16_path):
            sys.exit(1)

        # Step 2: Quantize
        for quant_type in args.quant_types:
            quant_path = output_dir / f"{base_name}-{quant_type}.gguf"
            quantize_gguf(f16_path, quant_path, quant_type)

        print(f"\n  F16 GGUF kept at: {f16_path}")
        print(f"  Delete to save space: rm {f16_path}")

    # Step 3: Export vision projector
    if not args.skip_projector:
        mmproj_path = output_dir / f"mmproj-{base_name}-F16.gguf"
        if not export_projector(model_dir, mmproj_path):
            # Fallback: copy existing base projector
            base_mmproj = Path("../../models/mmproj-moondream2-*.gguf")
            import glob
            candidates = glob.glob(str(base_mmproj))
            if candidates:
                print(f"  Copying base projector: {candidates[0]}")
                shutil.copy2(candidates[0], mmproj_path)
            else:
                print("  [WARN] No projector available — vision may not work")

    # Step 4: Deploy
    if args.deploy:
        best_quant = args.quant_types[0]
        gguf_path = output_dir / f"{base_name}-{best_quant}.gguf"
        mmproj_path = output_dir / f"mmproj-{base_name}-F16.gguf"
        app_dir = Path("../../ocula_app/assets/models")

        if gguf_path.exists():
            deploy_to_app(gguf_path, mmproj_path, app_dir)
        else:
            print(f"[!] Quantized model not found: {gguf_path}")

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  MOONDREAM2 GGUF EXPORT COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Output: {output_dir}")
    for f in sorted(output_dir.glob(f"{base_name}*.gguf")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    • {f.name} ({size_mb:.1f} MB)")
    for f in sorted(output_dir.glob("mmproj-moondream2*.gguf")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    • {f.name} ({size_mb:.1f} MB)")
    print(f"\n  To deploy: python 07b_export_moondream2_gguf.py --deploy")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
