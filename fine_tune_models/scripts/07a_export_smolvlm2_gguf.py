#!/usr/bin/env python3
"""
SmolVLM2 GGUF Export
====================
Converts the fine-tuned SmolVLM2 (HuggingFace format) → GGUF for llama.cpp.

Handles both:
  • CUDA path:  merged HF model → convert_hf_to_gguf.py → llama-quantize
  • MLX path:   fused MLX model → convert to HF → convert_hf_to_gguf.py → quantize

The vision projector (mmproj) is exported separately since llama.cpp
loads the text model and vision encoder as two files.

Usage:
    # Export merged model (after 06_merge_lora.py)
    python 07a_export_smolvlm2_gguf.py

    # Export from MLX fused model
    python 07a_export_smolvlm2_gguf.py --from-mlx ../models/merged/smolvlm2-vision-merged

    # Custom quantization
    python 07a_export_smolvlm2_gguf.py --quant-types Q8_0 Q4_K_M

    # Export projector only
    python 07a_export_smolvlm2_gguf.py --projector-only
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ── Paths ────────────────────────────────────────────────────────
def find_llama_cpp():
    """Find llama.cpp in common locations relative to scripts dir."""
    candidates = [
        Path("../../llama.cpp"),   # Ocula/fine_tune_models/scripts/ → Ocula/llama.cpp
        Path("../llama.cpp"),      # ocula/scripts/ → ocula/llama.cpp (flat layout)
    ]
    for p in candidates:
        if (p / "convert_hf_to_gguf.py").exists():
            return p
    return candidates[0]

LLAMA_CPP_DIR = find_llama_cpp()
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

GGUF_OUTPUT_DIR = Path("../models/gguf")
MERGED_MODEL_DIRS = [
    Path("../models/merged/smolvlm2-vision-merged"),
    Path("../models/lora_adapters/smolvlm2-vision-merged"),
    Path("../models/lora_adapters/ocula-base-vision-merged"),
]

DEFAULT_QUANT_TYPES = ["Q8_0"]


def detect_cuda():
    """Check if CUDA is available for building llama.cpp with GPU support."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def setup_llama_cpp():
    """Clone and build llama.cpp automatically if not present."""
    repo_dir = LLAMA_CPP_DIR.resolve()

    if not repo_dir.exists():
        print(f"[*] Cloning llama.cpp into {repo_dir}...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(repo_dir)],
            check=True,
        )
    elif (repo_dir / ".git").exists():
        print(f"[*] Updating llama.cpp...")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)

    if not QUANTIZE_BIN.resolve().exists():
        print(f"[*] Building llama.cpp...")
        build_dir = (LLAMA_CPP_DIR / "build").resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_cmd = ["cmake", ".."]
        if detect_cuda():
            cmake_cmd += ["-DGGML_CUDA=ON"]
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if not cuda_home:
                for p in ["/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12"]:
                    if os.path.isdir(p):
                        cuda_home = p
                        break
            if cuda_home:
                cmake_cmd += [f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc"]
            print(f"    (with CUDA support{f', toolkit: {cuda_home}' if cuda_home else ''})")

        subprocess.run(cmake_cmd, cwd=str(build_dir), check=True)
        nproc = os.cpu_count() or 4
        subprocess.run(["make", f"-j{nproc}"], cwd=str(build_dir), check=True)

    print("[OK] llama.cpp ready")


def check_prerequisites():
    """Verify llama.cpp tools are available, auto-setup if missing."""
    if not CONVERT_SCRIPT.exists() or not QUANTIZE_BIN.exists():
        print("[*] llama.cpp tools not found — setting up automatically...")
        setup_llama_cpp()

    # Final check
    errors = []
    if not CONVERT_SCRIPT.exists():
        errors.append(f"  convert_hf_to_gguf.py not found: {CONVERT_SCRIPT}")
    if not QUANTIZE_BIN.exists():
        errors.append(f"  llama-quantize not found: {QUANTIZE_BIN}")
    if errors:
        print("[!] Auto-setup failed:")
        for e in errors:
            print(e)
        sys.exit(1)
    print("[OK] llama.cpp tools found")


def convert_hf_to_gguf(model_dir: Path, output_path: Path, model_type: str = "auto"):
    """Convert HuggingFace model to F16 GGUF."""
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


def quantize_gguf(input_path: Path, output_path: Path, quant_type: str = "Q8_0"):
    """Quantize F16 GGUF to target quantization."""
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

    SmolVLM2 uses a SigLIP vision encoder. The projector maps vision
    embeddings to the LM's embedding space. llama.cpp loads it separately.
    """
    print(f"\n[*] Exporting vision projector (mmproj)")

    # The convert script should handle mmproj if the model has one
    # For SmolVLM2, we need to check if there's a dedicated mmproj converter
    mmproj_script = LLAMA_CPP_DIR / "examples" / "llava" / "convert_hf_to_gguf_llava.py"
    alt_mmproj_script = LLAMA_CPP_DIR / "tools" / "mtmd" / "convert_hf_to_mtmd_gguf.py"

    # Try the newer mtmd converter first (llama.cpp commit after March 2025)
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
        print("  → The base mmproj GGUF should still work. Copy it manually:")
        print(f"  cp ../../ocula_app/assets/models/mmproj-SmolVLM2-*.gguf {output_path}")
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

    # For bundled models (free tier)
    bundled_dir = app_models_dir
    # For server-downloadable models
    server_dir = app_models_dir.parent / "models_server"

    bundled_dir.mkdir(parents=True, exist_ok=True)
    server_dir.mkdir(parents=True, exist_ok=True)

    # SmolVLM2 is free tier → goes into bundled assets
    target_model = bundled_dir / gguf_path.name
    target_mmproj = bundled_dir / mmproj_path.name

    print(f"    Model:   {gguf_path} → {target_model}")
    shutil.copy2(gguf_path, target_model)
    print(f"    mmproj:  {mmproj_path} → {target_mmproj}")
    shutil.copy2(mmproj_path, target_mmproj)

    print(f"    [OK] Deployed to {bundled_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned SmolVLM2 to GGUF for llama.cpp")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to merged HuggingFace model")
    parser.add_argument("--from-mlx", type=Path, default=None,
                        help="Path to MLX fused model (will convert via HF)")
    parser.add_argument("--output-dir", type=Path, default=GGUF_OUTPUT_DIR,
                        help="Output directory for GGUF files")
    parser.add_argument("--quant-types", nargs="+", default=DEFAULT_QUANT_TYPES,
                        help="Quantization types (e.g., Q8_0 Q4_K_M)")
    parser.add_argument("--projector-only", action="store_true",
                        help="Only export the vision projector")
    parser.add_argument("--deploy", action="store_true",
                        help="Copy GGUFs to ocula_app/assets/models/")
    parser.add_argument("--skip-projector", action="store_true",
                        help="Skip projector export (use base mmproj)")
    args = parser.parse_args()

    check_prerequisites()

    model_dir = args.from_mlx or args.model_dir
    if model_dir is None:
        for candidate in MERGED_MODEL_DIRS:
            if candidate.exists():
                model_dir = candidate
                print(f"[*] Found merged model at: {model_dir}")
                break

    if model_dir is None or not model_dir.exists():
        print(f"[!] Model directory not found. Searched:")
        for d in MERGED_MODEL_DIRS:
            print(f"    - {d}")
        print("    Run these first:")
        print("    1. python 02b_train_smolvlm2_vision.py")
        print("    2. python 06_merge_lora.py --model smolvlm2")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = "SmolVLM2-500M-Video-Instruct-finetuned"

    if not args.projector_only:
        # Step 1: Convert to F16 GGUF
        f16_path = output_dir / f"{base_name}-F16.gguf"
        if not convert_hf_to_gguf(model_dir, f16_path):
            sys.exit(1)

        # Step 2: Quantize
        for quant_type in args.quant_types:
            quant_path = output_dir / f"{base_name}-{quant_type}.gguf"
            quantize_gguf(f16_path, quant_path, quant_type)

        # Optionally clean up F16 (large)
        print(f"\n  F16 GGUF kept at: {f16_path}")
        print(f"  Delete it to save space: rm {f16_path}")

    # Step 3: Export vision projector
    if not args.skip_projector:
        mmproj_path = output_dir / f"mmproj-{base_name}-F16.gguf"
        if not export_projector(model_dir, mmproj_path):
            # Fallback: copy existing base projector
            base_mmproj = Path("../../ocula_app/assets/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
            if base_mmproj.exists():
                print(f"  Copying base projector: {base_mmproj}")
                shutil.copy2(base_mmproj, mmproj_path)
            else:
                print("  [WARN] No projector available — vision may not work")

    # Step 4: Deploy to app
    if args.deploy:
        # Use the best quantized model
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
    print(f"  GGUF EXPORT COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Output: {output_dir}")
    for f in sorted(output_dir.glob(f"{base_name}*.gguf")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    • {f.name} ({size_mb:.1f} MB)")
    for f in sorted(output_dir.glob("mmproj-*.gguf")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    • {f.name} ({size_mb:.1f} MB)")
    print(f"\n  To deploy: python 07a_export_smolvlm2_gguf.py --deploy")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
