#!/usr/bin/env python3
"""
SmolVLM2-500M Vision-Language Fine-tuning
==========================================
Proper multimodal SFT using the processor (image + text jointly).

Supports:
  • CUDA  — HuggingFace Transformers + PEFT LoRA + SFTTrainer (TRL)
  • MLX   — mlx-vlm LoRA fine-tuning on Apple Silicon
  • MPS   — PyTorch MPS fallback (slower than MLX on Apple Silicon)

Key difference from 02_train_smolvlm2.py:
  This script uses **AutoProcessor** (not just tokenizer) so the model
  learns from image pixels + text together.  The old script was text-only.

Prerequisites:
  pip install transformers>=4.46 peft>=0.13 trl>=0.12 \
              datasets accelerate bitsandbytes pillow
  # For MLX:
  pip install mlx-vlm>=0.1.2

Usage:
    # Auto-detect best backend
    python 02b_train_smolvlm2_vision.py

    # Force backend
    python 02b_train_smolvlm2_vision.py --backend cuda
    python 02b_train_smolvlm2_vision.py --backend mlx

    # Custom data
    python 02b_train_smolvlm2_vision.py \
        --train-data ../data/vision/smolvlm2_vision_train.jsonl \
        --val-data   ../data/vision/smolvlm2_vision_val.jsonl

    # Resume from checkpoint
    python 02b_train_smolvlm2_vision.py --resume ../models/lora_adapters/smolvlm2-vision/checkpoint-500
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import yaml

# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_CONFIG  = "../configs/smolvlm2_vision.yaml"
DEFAULT_TRAIN   = "../data/vision/smolvlm2_vision_train.jsonl"
DEFAULT_VAL     = "../data/vision/smolvlm2_vision_val.jsonl"
DEFAULT_OUTPUT  = "../models/lora_adapters/ocula-base-vision"


def load_config(path: str) -> dict:
    """Load YAML config, returning empty dict if file missing."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def detect_backend() -> str:
    """Auto-detect the best available training backend."""
    # Check CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"[*] CUDA detected: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
    except ImportError:
        pass

    # Apple Silicon — prefer MLX
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            import mlx_vlm   # noqa: F401
            print("[*] Apple Silicon + mlx-vlm detected → MLX backend")
            return "mlx"
        except ImportError:
            pass
        # Fallback: PyTorch MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                print("[*] Apple Silicon MPS detected (install mlx-vlm for faster training)")
                return "mps"
        except (ImportError, AttributeError):
            pass

    print("[!] No GPU detected — CPU training (will be slow)")
    return "cpu"


# ═════════════════════════════════════════════════════════════════
# CUDA / MPS Training (HuggingFace Transformers + TRL SFTTrainer)
# ═════════════════════════════════════════════════════════════════

def train_cuda_mps(config: dict, args):
    """
    Vision-language LoRA fine-tuning with SFTTrainer.

    This properly handles multimodal inputs:
    - Images are processed through the vision encoder
    - Text is tokenized alongside image tokens
    - LoRA adapters are applied to the language model layers
    """
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name

    # MLX uses special "-mlx" model repos for compatible weights.
    if isinstance(model_path, str) and model_path == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
        model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx"

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    device = "cuda" if args.backend == "cuda" else "mps"

    # ── Quantization for limited VRAM ──
    bnb_config = None
    if args.backend == "cuda" and args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("[*] Using 4-bit QLoRA (saves VRAM)")

    # ── Load model + processor ──
    print(f"\n[*] Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to("mps")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")

    # ── LoRA config ──
    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ── Load dataset ──
    print(f"[*] Loading training data: {args.train_data}")
    train_ds = load_dataset("json", data_files=args.train_data, split="train")
    val_ds = None
    if args.val_data and Path(args.val_data).exists():
        val_ds = load_dataset("json", data_files=args.val_data, split="train")
        print(f"  Validation: {len(val_ds)} examples")

    max_samples = train_cfg.get("max_samples", args.max_samples)
    if max_samples and len(train_ds) > max_samples:
        train_ds = train_ds.shuffle(seed=42).select(range(max_samples))
    print(f"  Training: {len(train_ds)} examples")

    # ── Collator for multimodal data ──
    class VisionCollator:
        """Process image + text pairs into model inputs."""
        def __init__(self, processor, max_length=2048):
            self.processor = processor
            self.max_length = max_length

        def __call__(self, examples):
            texts = []
            images_list = []

            for ex in examples:
                messages = ex.get("messages", [])
                image_paths = ex.get("images", [])

                # Build conversation text via processor's chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)

                # Load images
                imgs = []
                for img_path in image_paths:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        imgs.append(img)
                    except Exception:
                        # Create tiny placeholder if image missing
                        imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
                images_list.append(imgs if imgs else None)

            # Process with the multimodal processor
            if any(imgs is not None for imgs in images_list):
                # Flatten images for batch processing
                batch = self.processor(
                    text=texts,
                    images=[imgs[0] if imgs else None for imgs in images_list],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                batch = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )

            # Labels = input_ids (standard causal LM)
            labels = batch["input_ids"].clone()
            # Mask padding tokens in labels
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch

    collator = VisionCollator(
        processor,
        max_length=train_cfg.get("max_seq_length", 2048),
    )

    # ── Training config ──
    epochs = train_cfg.get("epochs", 3)
    batch_size = train_cfg.get("batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    lr = train_cfg.get("learning_rate", 2e-4)

    if device == "mps":
        # MPS has limited memory — reduce batch size
        batch_size = min(batch_size, 2)
        grad_accum = max(grad_accum, 8)

    output_dir = args.output or DEFAULT_OUTPUT

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=(device == "cuda"),
        fp16=False,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=500 if val_ds else None,
        report_to=["tensorboard"],
        logging_dir=f"../logs/ocula-base-vision",
        dataloader_num_workers=4 if device == "cuda" else 0,
        remove_unused_columns=False,
        dataset_text_field=None,          # We use a custom collator
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ── Train ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        peft_config=lora_config,
    )

    print(f"\n[*] Starting vision SFT on {device.upper()}")
    print(f"    Epochs: {epochs}, Batch: {batch_size} × {grad_accum} accum, LR: {lr}")
    print(f"    Output: {output_dir}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ──
    print(f"\n[*] Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("[OK] SmolVLM2 vision fine-tuning complete!")
    print(f"     Next: python 06_merge_lora.py --model ocula-base")
    print(f"     Then: python 07_quantize_gguf.py --model ocula-base")


# ═════════════════════════════════════════════════════════════════
# MLX Training (Apple Silicon — recommended for Mac)
# ═════════════════════════════════════════════════════════════════

def _patch_smolvlm2_config(model_path: str) -> str:
    """
    Patch SmolVLM2-500M config.json to include num_hidden_layers in vision_config.

    mlx_vlm's SmolVLM VisionConfig defaults to 27 layers (from the larger model),
    but SmolVLM2-500M only has 12 vision encoder layers. The HuggingFace config.json
    omits num_hidden_layers, so we need to add it explicitly.

    Returns the (possibly updated) model_path pointing to the patched local copy.
    """
    import json
    from huggingface_hub import snapshot_download

    # Download model to a local cache we can modify
    local_dir = Path("../models/base/smolvlm2-500m-patched")
    if not local_dir.exists():
        print(f"[*] Downloading {model_path} to {local_dir} ...")
        snapshot_download(model_path, local_dir=str(local_dir))

    config_path = local_dir / "config.json"
    if not config_path.exists():
        print(f"[!] No config.json found in {local_dir}")
        return model_path

    with open(config_path) as f:
        cfg = json.load(f)

    vision_cfg = cfg.get("vision_config", {})
    has_layers = vision_cfg.get("num_hidden_layers") is not None
    has_intermediate = vision_cfg.get("intermediate_size") is not None

    if not has_layers or not has_intermediate:
        # Infer missing values from actual weight shapes
        from safetensors import safe_open
        import glob
        safetensors_files = glob.glob(str(local_dir / "*.safetensors"))
        layer_nums = set()
        intermediate_size = None
        for sf_path in safetensors_files:
            with safe_open(sf_path, framework="numpy") as sf:
                for key in sf.keys():
                    if "vision_model.encoder.layers." in key:
                        num = int(key.split("vision_model.encoder.layers.")[1].split(".")[0])
                        layer_nums.add(num)
                    if "vision_model.encoder.layers.0.mlp.fc1.weight" in key:
                        intermediate_size = sf.get_tensor(key).shape[0]

        if not has_layers:
            num_layers = len(layer_nums) if layer_nums else 12
            vision_cfg["num_hidden_layers"] = num_layers
            print(f"[*] Patched vision_config.num_hidden_layers = {num_layers}")

        if not has_intermediate and intermediate_size:
            vision_cfg["intermediate_size"] = intermediate_size
            print(f"[*] Patched vision_config.intermediate_size = {intermediate_size}")

        cfg["vision_config"] = vision_cfg
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[*] Config saved to {config_path}")
    else:
        print(f"[*] vision_config OK: num_hidden_layers={vision_cfg['num_hidden_layers']}, "
              f"intermediate_size={vision_cfg['intermediate_size']}")

    return str(local_dir)


def train_mlx(config: dict, args):
    """
    Fine-tune SmolVLM2 using mlx-vlm on Apple Silicon.

    mlx-vlm provides native VLM LoRA training that:
    - Uses unified memory (no CPU↔GPU copies)
    - Handles image+text jointly
    - Is significantly faster than MPS PyTorch on Apple Silicon
    - Supports LoRA out of the box

    Requires: pip install mlx-vlm>=0.1.2
    """
    import subprocess
    import tempfile

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name

    # Patch config.json to fix missing num_hidden_layers in vision_config
    # (mlx_vlm defaults to 27 layers but SmolVLM2-500M only has 12)
    model_path = _patch_smolvlm2_config(model_path)

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    mlx_cfg = config.get("mlx", {})

    output_dir = args.output or DEFAULT_OUTPUT + "-mlx"
    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare data in mlx-vlm format ──
    # mlx-vlm expects JSONL with:
    #   {"messages": [...], "images": ["path/to/img.jpg"]}
    # This is already our format from 02a_prepare_vision_data.py!

    train_file = args.train_data
    val_file = args.val_data

    if not Path(train_file).exists():
        print(f"[!] Training data not found: {train_file}")
        print("    Run: python 02a_prepare_vision_data.py")
        sys.exit(1)

    # Count examples
    with open(train_file) as f:
        n_train = sum(1 for _ in f)
    print(f"[*] Training examples: {n_train:,}")

    # ── Determine iterations ──
    batch_size = mlx_cfg.get("batch_size", 4)
    epochs = train_cfg.get("epochs", 3)
    iters = (n_train * epochs) // batch_size
    # Apply max_steps from config (prevents 50+ hour runs when loss converges early)
    max_steps = train_cfg.get("max_steps")
    if max_steps:
        iters = min(iters, max_steps)
    if args.max_iters:
        iters = min(iters, args.max_iters)

    # ── Create mlx-vlm config ──
    mlx_train_config = {
        "model": model_path,
        "data": str(Path(train_file).parent),
        "train_file": Path(train_file).name,
        "adapter_path": output_dir,
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": train_cfg.get("learning_rate", 2e-4),
        "lora_layers": mlx_cfg.get("lora_layers", 16),
        "lora_rank": lora_cfg.get("rank", 32),
        "val_batches": 25,
        "steps_per_report": 10,
        "steps_per_eval": 200,
        "save_every": 500,
        "seed": 42,
    }

    if val_file and Path(val_file).exists():
        mlx_train_config["valid_file"] = Path(val_file).name

    config_file = Path(output_dir) / "train_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(mlx_train_config, f, default_flow_style=False)

    print(f"\n[*] MLX-VLM LoRA Training Config:")
    print(f"    Model:        {model_path}")
    print(f"    Epochs:       {epochs}")
    print(f"    Steps:        {iters}")
    print(f"    Batch size:   {batch_size}")
    print(f"    LoRA rank:    {mlx_train_config['lora_rank']}")
    print(f"    LoRA alpha:   {mlx_train_config['lora_rank'] * 2}")
    print(f"    LR:           {mlx_train_config['learning_rate']}")
    print(f"    Output:       {output_dir}")

    # ── Run mlx-vlm training ──
    # mlx-vlm >=0.2 changed CLI: --model-path, --dataset, --steps, --output-path
    # (dropped: --train, --lora-layers, --iters, --steps-per-report, etc.)
    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", model_path,
        "--dataset", str(Path(train_file).parent),
        "--output-path", output_dir,
        "--batch-size", str(batch_size),
        "--lora-rank", str(mlx_train_config["lora_rank"]),
        "--lora-alpha", str(mlx_train_config["lora_rank"] * 2),  # alpha = 2*rank
        "--learning-rate", str(mlx_train_config["learning_rate"]),
        "--epochs", str(epochs),
        "--steps", str(iters),
        "--print-every", str(mlx_train_config["steps_per_report"]),
        "--save-after-epoch",
        # Our data already has correct message format with {"type": "image"} entries.
        # Passing --apply-chat-template disables mlx-vlm's own template pre-processing
        # (action="store_false" in argparse) which would otherwise strip image tokens.
        "--apply-chat-template",
    ]

    # Resume from existing adapter if present
    adapter_weights = Path(output_dir) / "adapters.safetensors"
    if adapter_weights.exists():
        cmd.extend(["--adapter-path", output_dir])

    print(f"\n[*] Command: {' '.join(cmd)}")
    print(f"\n{'═' * 60}")
    print(f"  TRAINING START")
    print(f"{'═' * 60}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[!] mlx-vlm training exited with code {result.returncode}")
        print("    Check that mlx-vlm is installed: pip install mlx-vlm>=0.1.2")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"\n[OK] MLX LoRA adapters saved to: {output_dir}")
    print(f"\n  Next steps:")
    print(f"  1. Fuse:     python -m mlx_vlm.fuse --model {model_path} \\")
    print(f"                 --adapter-path {output_dir} \\")
    print(f"                 --save-path ../models/merged/ocula-base-vision-merged")
    print(f"  2. Convert:  python 07a_export_smolvlm2_gguf.py")
    print(f"  3. Deploy:   Copy GGUF to ocula_app/assets/models/")


# ═════════════════════════════════════════════════════════════════
# MLX Inference Test
# ═════════════════════════════════════════════════════════════════

def test_model(config: dict, args):
    """Quick inference test with the fine-tuned model."""
    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    adapter_path = args.output or DEFAULT_OUTPUT

    print(f"\n[*] Testing fine-tuned model")
    print(f"    Base:    {model_name}")
    print(f"    Adapter: {adapter_path}")

    backend = args.backend if args.backend != "auto" else detect_backend()

    if backend == "mlx":
        _test_mlx(model_name, adapter_path)
    else:
        _test_transformers(model_name, adapter_path, backend)


def _test_mlx(model_name, adapter_path):
    from mlx_vlm import load as mlx_load, generate as mlx_generate
    from PIL import Image

    model, processor = mlx_load(model_name, adapter_path=adapter_path)

    test_prompts = [
        ("Describe this image in detail.", None),
        ("What text can you see in this image?", None),
        ("Summarize the content of this document.", None),
    ]

    for prompt, img_path in test_prompts:
        print(f"\n  Q: {prompt}")
        if img_path:
            image = Image.open(img_path)
            output = mlx_generate(model, processor, prompt, image, max_tokens=200)
        else:
            output = mlx_generate(model, processor, prompt, max_tokens=200)
        print(f"  A: {output}")


def _test_transformers(model_name, adapter_path, backend):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

    device = "cuda" if backend == "cuda" else "mps" if backend == "mps" else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()

    test_prompt = "Describe this image in detail."
    inputs = processor(text=test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    result = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n  Q: {test_prompt}")
    print(f"  A: {result}")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SmolVLM2-500M Vision-Language Fine-tuning (MLX + CUDA)")

    parser.add_argument("--backend", choices=["cuda", "mlx", "mps", "cpu", "auto"],
                        default="auto", help="Training backend")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="YAML config file")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN,
                        help="Training JSONL file")
    parser.add_argument("--val-data", default=DEFAULT_VAL,
                        help="Validation JSONL file")
    parser.add_argument("--output", default=None,
                        help="Output directory for adapters")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Cap MLX iterations")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit QLoRA (CUDA only, saves VRAM)")
    parser.add_argument("--test", action="store_true",
                        help="Run inference test instead of training")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.test:
        test_model(config, args)
        return

    backend = args.backend if args.backend != "auto" else detect_backend()
    args.backend = backend  # Store resolved backend

    if backend == "mlx":
        train_mlx(config, args)
    elif backend in ("cuda", "mps", "cpu"):
        train_cuda_mps(config, args)
    else:
        print(f"[!] Unknown backend: {backend}")
        sys.exit(1)


if __name__ == "__main__":
    main()
