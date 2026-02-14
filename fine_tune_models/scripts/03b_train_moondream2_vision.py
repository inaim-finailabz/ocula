#!/usr/bin/env python3
"""
Moondream2 Vision-Language Fine-tuning
=======================================
Proper multimodal SFT using the processor (image + text jointly).

Supports:
  • CUDA  — HuggingFace Transformers + PEFT LoRA + SFTTrainer (TRL)
  • MLX   — mlx-vlm LoRA fine-tuning on Apple Silicon
  • MPS   — PyTorch MPS fallback (slower than MLX on Apple Silicon)

Key difference from 03_train_moondream2.py:
  This script uses **AutoProcessor** (not just tokenizer) so the model
  learns from image pixels + text together.  The old script was text-only.

Moondream2 architecture notes:
  - Uses SigLIP vision encoder (378×378 images)
  - Custom Phi-1.5 based language model
  - Model class: AutoModelForCausalLM with integrated vision support
  - Processor handles image + text jointly via moondream's generate()

Prerequisites:
  pip install transformers>=4.46 peft>=0.13 trl>=0.12 \
              datasets accelerate bitsandbytes pillow einops
  # For MLX:
  pip install mlx-vlm>=0.1.2

Usage:
    python 03b_train_moondream2_vision.py
    python 03b_train_moondream2_vision.py --backend cuda --use-4bit
    python 03b_train_moondream2_vision.py --backend mlx
    python 03b_train_moondream2_vision.py --resume ../models/lora_adapters/moondream2-vision/checkpoint-500
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

DEFAULT_MODEL   = "vikhyatk/moondream2"
DEFAULT_CONFIG  = "../configs/moondream2_vision.yaml"
DEFAULT_TRAIN   = "../data/vision_moondream/moondream2_vision_train.jsonl"
DEFAULT_VAL     = "../data/vision_moondream/moondream2_vision_val.jsonl"
DEFAULT_OUTPUT  = "../models/lora_adapters/moondream2-vision"


def load_config(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def detect_backend() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"[*] CUDA detected: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
    except ImportError:
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core   # noqa: F401
            import mlx_vlm    # noqa: F401
            print("[*] Apple Silicon + mlx-vlm detected → MLX backend")
            return "mlx"
        except ImportError:
            pass
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
    Moondream2 vision LoRA fine-tuning with SFTTrainer.

    Moondream2 uses a custom architecture:
      • SigLIP vision encoder (378×378)
      • Phi-1.5 based language model
      • Custom projection between vision & language

    We use AutoModelForCausalLM with trust_remote_code=True
    since Moondream2 registers its own model class.
    The processor handles image tokenization internally.
    """
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name
    revision = config.get("model", {}).get("revision", "2025-01-09")

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    device = "cuda" if args.backend == "cuda" else "mps"

    # ── Quantization ──
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
    print(f"\n[*] Loading model: {model_path} (revision: {revision})")
    processor = AutoProcessor.from_pretrained(
        model_path, revision=revision, trust_remote_code=True
    )

    # Moondream2 uses AutoModelForCausalLM with trust_remote_code
    # (it registers its own model class internally)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
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
    # Moondream2's Phi-1.5 backbone uses these projection names
    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "dense",
        "fc1", "fc2",
    ])
    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
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
    class MoondreamVisionCollator:
        """
        Process image + text pairs for Moondream2.

        Moondream2's processor:
          - Accepts images through its encode_image() method
          - Text is processed with the tokenizer
          - We use the processor's chat template to format conversations
        """
        def __init__(self, processor, max_length=2048):
            self.processor = processor
            self.max_length = max_length

        def __call__(self, examples):
            texts = []
            images_list = []

            for ex in examples:
                messages = ex.get("messages", [])
                image_paths = ex.get("images", [])

                # Build conversation text
                # Moondream2 uses a simple format:
                #   <image>\n\nQuestion: {question}\n\nAnswer: {answer}
                user_text = ""
                assistant_text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for c in content:
                            if c.get("type") == "text":
                                if role == "user":
                                    user_text = c["text"]
                                elif role == "assistant":
                                    assistant_text = c["text"]
                    elif isinstance(content, str):
                        if role == "user":
                            user_text = content
                        elif role == "assistant":
                            assistant_text = content

                # Moondream2's prompt format
                text = f"<image>\n\nQuestion: {user_text}\n\nAnswer: {assistant_text}"
                texts.append(text)

                # Load images
                imgs = []
                for img_path in image_paths:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        imgs.append(img)
                    except Exception:
                        imgs.append(Image.new("RGB", (378, 378), (128, 128, 128)))
                images_list.append(imgs if imgs else None)

            # Process with the multimodal processor
            try:
                if any(imgs is not None for imgs in images_list):
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
            except Exception:
                # Fallback: tokenize text only (for debugging)
                batch = self.processor.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )

            # Labels = input_ids
            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch

    collator = MoondreamVisionCollator(
        processor,
        max_length=train_cfg.get("max_seq_length", 2048),
    )

    # ── Training config ──
    epochs = train_cfg.get("epochs", 2)
    batch_size = train_cfg.get("batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    lr = train_cfg.get("learning_rate", 1e-4)

    if device == "mps":
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
        logging_dir=f"../logs/moondream2-vision",
        dataloader_num_workers=4 if device == "cuda" else 0,
        remove_unused_columns=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        gradient_checkpointing=True,  # Moondream2 is 1.8B — needs this
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

    print(f"\n[*] Starting Moondream2 vision SFT on {device.upper()}")
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

    print("[OK] Moondream2 vision fine-tuning complete!")
    print(f"     Next: python 06_merge_lora.py --model moondream2")
    print(f"     Then: python 07b_export_moondream2_gguf.py")


# ═════════════════════════════════════════════════════════════════
# MLX Training (Apple Silicon — recommended for Mac)
# ═════════════════════════════════════════════════════════════════

def train_mlx(config: dict, args):
    """
    Fine-tune Moondream2 using mlx-vlm on Apple Silicon.

    mlx-vlm handles Moondream2's custom architecture natively,
    including the SigLIP vision encoder.

    Requires: pip install mlx-vlm>=0.1.2
    """
    import subprocess

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    mlx_cfg = config.get("mlx", {})

    output_dir = args.output or DEFAULT_OUTPUT + "-mlx"
    os.makedirs(output_dir, exist_ok=True)

    train_file = args.train_data
    val_file = args.val_data

    if not Path(train_file).exists():
        print(f"[!] Training data not found: {train_file}")
        print("    Run: python 03a_prepare_moondream_data.py")
        sys.exit(1)

    with open(train_file) as f:
        n_train = sum(1 for _ in f)
    print(f"[*] Training examples: {n_train:,}")

    batch_size = mlx_cfg.get("batch_size", 2)
    epochs = train_cfg.get("epochs", 2)
    iters = (n_train * epochs) // batch_size
    max_steps = train_cfg.get("max_steps")
    if max_steps:
        iters = min(iters, max_steps)
    if args.max_iters:
        iters = min(iters, args.max_iters)

    mlx_train_config = {
        "model": model_path,
        "data": str(Path(train_file).parent),
        "train_file": Path(train_file).name,
        "adapter_path": output_dir,
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": train_cfg.get("learning_rate", 1e-4),
        "lora_layers": mlx_cfg.get("lora_layers", 12),
        "lora_rank": lora_cfg.get("rank", 16),
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
    print(f"    Iterations:   {iters}")
    print(f"    Batch size:   {batch_size}")
    print(f"    LoRA layers:  {mlx_train_config['lora_layers']}")
    print(f"    LoRA rank:    {mlx_train_config['lora_rank']}")
    print(f"    LR:           {mlx_train_config['learning_rate']}")
    print(f"    Output:       {output_dir}")

    # mlx-vlm >=0.2 CLI: --model-path, --dataset, --steps, --output-path
    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", model_path,
        "--dataset", str(Path(train_file).parent),
        "--output-path", output_dir,
        "--batch-size", str(batch_size),
        "--lora-rank", str(mlx_train_config["lora_rank"]),
        "--lora-alpha", str(mlx_train_config["lora_rank"] * 2),
        "--learning-rate", str(mlx_train_config["learning_rate"]),
        "--epochs", str(epochs),
        "--steps", str(iters),
        "--print-every", str(mlx_train_config["steps_per_report"]),
        "--save-after-epoch",
        # Data already has correct {"type": "image"} format — disable mlx-vlm's
        # own template pre-processing which strips image tokens (num_images=0).
        "--apply-chat-template",
    ]

    # Resume from existing adapter
    adapter_weights = Path(output_dir) / "adapters.safetensors"
    if adapter_weights.exists():
        cmd.extend(["--adapter-path", output_dir])

    print(f"\n[*] Command: {' '.join(cmd)}")
    print(f"\n{'═' * 60}")
    print(f"  MOONDREAM2 TRAINING START")
    print(f"{'═' * 60}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[!] mlx-vlm training exited with code {result.returncode}")
        print("    Check: pip install mlx-vlm>=0.1.2")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"\n[OK] MLX LoRA adapters saved to: {output_dir}")
    print(f"\n  Next steps:")
    print(f"  1. Fuse:     python -m mlx_vlm.fuse --model {model_path} \\")
    print(f"                 --adapter-path {output_dir} \\")
    print(f"                 --save-path ../models/merged/moondream2-vision-merged")
    print(f"  2. Convert:  python 07b_export_moondream2_gguf.py")


# ═════════════════════════════════════════════════════════════════
# Inference Test
# ═════════════════════════════════════════════════════════════════

def test_model(config: dict, args):
    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    adapter_path = args.output or DEFAULT_OUTPUT

    print(f"\n[*] Testing fine-tuned Moondream2")
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
        "Describe this image in detail.",
        "What objects can you see?",
        "How many items are in this image?",
    ]

    for prompt in test_prompts:
        print(f"\n  Q: {prompt}")
        output = mlx_generate(model, processor, prompt, max_tokens=200)
        print(f"  A: {output}")


def _test_transformers(model_name, adapter_path, backend):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    device = "cuda" if backend == "cuda" else "mps" if backend == "mps" else "cpu"
    revision = "2025-01-09"

    processor = AutoProcessor.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.bfloat16, trust_remote_code=True
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
        description="Moondream2 Vision-Language Fine-tuning (MLX + CUDA)")

    parser.add_argument("--backend", choices=["cuda", "mlx", "mps", "cpu", "auto"],
                        default="auto")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN)
    parser.add_argument("--val-data", default=DEFAULT_VAL)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--use-4bit", action="store_true",
                        help="4-bit QLoRA (CUDA only)")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.test:
        test_model(config, args)
        return

    backend = args.backend if args.backend != "auto" else detect_backend()
    args.backend = backend

    if backend == "mlx":
        train_mlx(config, args)
    elif backend in ("cuda", "mps", "cpu"):
        train_cuda_mps(config, args)
    else:
        print(f"[!] Unknown backend: {backend}")
        sys.exit(1)


if __name__ == "__main__":
    main()
