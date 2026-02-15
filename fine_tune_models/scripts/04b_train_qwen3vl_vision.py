#!/usr/bin/env python3
"""
Qwen3-VL-2B Vision-Language Fine-tuning
=========================================
Proper multimodal SFT using the processor (image + text jointly).

Supports:
  • CUDA  — Unsloth FastVisionModel + SFTTrainer (fast, handles 4-bit natively)
  • MLX   — mlx-vlm LoRA fine-tuning on Apple Silicon
  • MPS   — PyTorch MPS fallback (slower than MLX on Apple Silicon)

Key difference from 04_train_qwen3vl.py:
  This script uses **AutoProcessor** (not just tokenizer) so the model
  learns from image pixels + text together.  The old script was text-only.

Qwen3-VL architecture notes:
  - Uses Qwen2VL's vision encoder with dynamic resolution
  - Rotary position embeddings for images (RoPE-2D)
  - Native ChatML template with <|im_start|>/<|im_end|>
  - Thinking mode with /think and /no_think tags

Prerequisites:
  # For CUDA (recommended):
  pip install unsloth trl datasets pillow
  # For MLX:
  pip install mlx-vlm>=0.1.2

Usage:
    python 04b_train_qwen3vl_vision.py
    python 04b_train_qwen3vl_vision.py --backend cuda --use-4bit
    python 04b_train_qwen3vl_vision.py --backend mlx
    python 04b_train_qwen3vl_vision.py --resume checkpoint-500
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

DEFAULT_MODEL   = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_CONFIG  = "../configs/qwen3vl_vision.yaml"
DEFAULT_TRAIN   = "../data/vision_qwen3vl/qwen3vl_vision_train.jsonl"
DEFAULT_VAL     = "../data/vision_qwen3vl/qwen3vl_vision_val.jsonl"
DEFAULT_OUTPUT  = "../models/lora_adapters/qwen3vl-vision"


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
# CUDA Training via Unsloth (handles 4-bit, LoRA, peft internally)
# ═════════════════════════════════════════════════════════════════

def train_cuda(config: dict, args):
    """
    Qwen3-VL-2B vision LoRA fine-tuning using Unsloth + SFTTrainer.

    Unsloth handles:
      - 4-bit quantization (no manual BitsAndBytesConfig)
      - LoRA injection (no manual get_peft_model)
      - Gradient checkpointing
      - 2x faster training with fused kernels
    """
    import torch
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name

    # Unsloth needs the unsloth/ prefixed name for optimized loading
    # Fall back to original path if unsloth variant isn't available
    unsloth_model = f"unsloth/{model_name.split('/')[-1]}"

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    # ── Load model with Unsloth ──
    print(f"\n[*] Loading model via Unsloth: {unsloth_model}")
    print(f"    (fallback: {model_path})")

    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=unsloth_model,
            load_in_4bit=args.use_4bit,
            use_gradient_checkpointing="unsloth",
        )
    except Exception as e:
        print(f"[!] Unsloth model not found ({e}), using HF path: {model_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=args.use_4bit,
            use_gradient_checkpointing="unsloth",
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")

    # ── Configure LoRA via Unsloth ──
    r = lora_cfg.get("rank", 16)
    lora_alpha = lora_cfg.get("alpha", 32)
    lora_dropout = lora_cfg.get("dropout", 0)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_modules=True,
        finetune_language_modules=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable LoRA parameters: {trainable / 1e6:.1f}M ({100 * trainable / total_params:.1f}%)")

    # ── Load & convert dataset to Unsloth vision format ──
    # Unsloth expects a list of {"messages": [...]} dicts where images are PIL objects
    from unsloth.trainer import UnslothVisionDataCollator

    print(f"\n[*] Loading training data: {args.train_data}")

    def load_jsonl(path):
        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    raw_train = load_jsonl(args.train_data)

    max_samples = train_cfg.get("max_samples", args.max_samples)
    if max_samples and len(raw_train) > max_samples:
        import random
        random.seed(42)
        random.shuffle(raw_train)
        raw_train = raw_train[:max_samples]

    raw_val = []
    if args.val_data and Path(args.val_data).exists():
        raw_val = load_jsonl(args.val_data)

    def convert_to_conversation(sample):
        """Convert our JSONL format to Unsloth vision format with PIL images."""
        messages = sample.get("messages", [])
        image_paths = sample.get("images", [])

        # Load image as PIL
        img = None
        if image_paths:
            try:
                img = Image.open(image_paths[0]).convert("RGB")
                img = img.resize((512, 512))
            except Exception:
                img = Image.new("RGB", (512, 512), (128, 128, 128))

        converted_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, list):
                new_content = []
                for c in content:
                    if c.get("type") == "image" and img is not None:
                        new_content.append({"type": "image", "image": img})
                    elif c.get("type") == "text":
                        new_content.append({"type": "text", "text": c["text"]})
                converted_messages.append({"role": role, "content": new_content})
            elif isinstance(content, str):
                if role == "user" and img is not None:
                    converted_messages.append({
                        "role": role,
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": content},
                        ],
                    })
                else:
                    converted_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}],
                    })

        return {"messages": converted_messages}

    print(f"  Converting {len(raw_train)} training examples to vision format...")
    train_ds = [convert_to_conversation(s) for s in raw_train]
    print(f"  Training: {len(train_ds)} examples")

    val_ds = None
    if raw_val:
        val_ds = [convert_to_conversation(s) for s in raw_val]
        print(f"  Validation: {len(val_ds)} examples")

    # ── Training config ──
    epochs = train_cfg.get("epochs", 1)
    batch_size = train_cfg.get("batch_size", 2)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    lr = train_cfg.get("learning_rate", 2e-4)
    max_steps = train_cfg.get("max_steps", -1)
    if args.max_iters:
        max_steps = args.max_iters

    output_dir = args.output or DEFAULT_OUTPUT

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=5,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_steps=500,
        save_total_limit=3,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        report_to=["tensorboard"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        optim="adamw_8bit",
        max_grad_norm=1.0,
        max_seq_length=train_cfg.get("max_seq_length", 4096),
    )

    # ── Train with Unsloth's vision collator ──
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    print(f"\n[*] Starting Qwen3-VL vision SFT via Unsloth on CUDA")
    print(f"    Epochs: {epochs}, Batch: {batch_size} × {grad_accum} accum, LR: {lr}")
    print(f"    Max steps: {max_steps if max_steps and max_steps > 0 else 'unlimited'}")
    print(f"    4-bit: {args.use_4bit}")
    print(f"    Output: {output_dir}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ──
    print(f"\n[*] Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[OK] Qwen3-VL vision fine-tuning complete!")
    print(f"     Next: python 07c_export_qwen3vl_gguf.py")


# ═════════════════════════════════════════════════════════════════
# MPS Training (fallback for Apple Silicon without MLX)
# ═════════════════════════════════════════════════════════════════

def train_mps(config: dict, args):
    """
    MPS fallback — uses HuggingFace Transformers + PEFT directly.
    Slower than MLX on Apple Silicon. Use train_mlx() instead when possible.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as QwenVLModel

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    print(f"\n[*] Loading model: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = QwenVLModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to("mps")
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")

    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    print(f"[*] Loading training data: {args.train_data}")
    train_ds = load_dataset("json", data_files=args.train_data, split="train")
    val_ds = None
    if args.val_data and Path(args.val_data).exists():
        val_ds = load_dataset("json", data_files=args.val_data, split="train")

    max_samples = train_cfg.get("max_samples", args.max_samples)
    if max_samples and len(train_ds) > max_samples:
        train_ds = train_ds.shuffle(seed=42).select(range(max_samples))
    print(f"  Training: {len(train_ds)} examples")

    class QwenVisionCollator:
        def __init__(self, processor, max_length=4096):
            self.processor = processor
            self.max_length = max_length

        def __call__(self, examples):
            texts = []
            images_list = []
            for ex in examples:
                messages = ex.get("messages", [])
                image_paths = ex.get("images", [])
                qwen_messages = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        qwen_content = []
                        for c in content:
                            if c.get("type") == "image" and image_paths:
                                qwen_content.append({"type": "image", "image": image_paths[0]})
                            elif c.get("type") == "text":
                                qwen_content.append({"type": "text", "text": c["text"]})
                        qwen_messages.append({"role": role, "content": qwen_content})
                    elif isinstance(content, str):
                        qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)
                imgs = []
                for img_path in image_paths:
                    try:
                        imgs.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (448, 448), (128, 128, 128)))
                images_list.append(imgs if imgs else None)

            try:
                if any(imgs is not None for imgs in images_list):
                    batch = self.processor(
                        text=texts,
                        images=[imgs[0] if imgs else None for imgs in images_list],
                        return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
                    )
                else:
                    batch = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            except Exception as e:
                print(f"[!] Processor error: {e}")
                batch = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            return batch

    collator = QwenVisionCollator(processor, max_length=train_cfg.get("max_seq_length", 4096))

    epochs = train_cfg.get("epochs", 1)
    batch_size = 1
    grad_accum = max(train_cfg.get("gradient_accumulation", 8), 16)
    lr = train_cfg.get("learning_rate", 5e-5)
    max_steps = train_cfg.get("max_steps", -1)
    if args.max_iters:
        max_steps = args.max_iters
    output_dir = args.output or DEFAULT_OUTPUT

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=100,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=False,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        eval_strategy="steps" if val_ds else "no",
        report_to=["tensorboard"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        peft_config=lora_config,
    )

    print(f"\n[*] Starting Qwen3-VL vision SFT on MPS")
    print(f"    Epochs: {epochs}, Batch: {batch_size} × {grad_accum} accum, LR: {lr}")
    print(f"    Output: {output_dir}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    print(f"\n[*] Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("[OK] Qwen3-VL vision fine-tuning complete!")


# ═════════════════════════════════════════════════════════════════
# MLX Training (Apple Silicon)
# ═════════════════════════════════════════════════════════════════

def train_mlx(config: dict, args):
    """
    Fine-tune Qwen3-VL using mlx-vlm on Apple Silicon.

    mlx-vlm supports Qwen2-VL architecture natively.
    Uses unified memory — can handle 2B model well on 32GB+ Macs.

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
        print("    Run: python 04a_prepare_qwen3vl_data.py")
        sys.exit(1)

    with open(train_file) as f:
        n_train = sum(1 for _ in f)
    print(f"[*] Training examples: {n_train:,}")

    batch_size = mlx_cfg.get("batch_size", 1)
    epochs = train_cfg.get("epochs", 1)
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
        "learning_rate": train_cfg.get("learning_rate", 5e-5),
        "lora_layers": mlx_cfg.get("lora_layers", 8),
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
    print(f"  QWEN3-VL TRAINING START")
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
    print(f"                 --save-path ../models/merged/qwen3vl-vision-merged")
    print(f"  2. Convert:  python 07c_export_qwen3vl_gguf.py")


# ═════════════════════════════════════════════════════════════════
# Inference Test
# ═════════════════════════════════════════════════════════════════

def test_model(config: dict, args):
    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    adapter_path = args.output or DEFAULT_OUTPUT

    print(f"\n[*] Testing fine-tuned Qwen3-VL")
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
        "What text can you read in this document?",
        "Analyze the data shown in this chart.",
    ]

    for prompt in test_prompts:
        print(f"\n  Q: {prompt}")
        output = mlx_generate(model, processor, prompt, max_tokens=200)
        print(f"  A: {output}")


def _test_transformers(model_name, adapter_path, backend):
    import torch
    from transformers import AutoProcessor
    from peft import PeftModel

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as QwenVLModel

    device = "cuda" if backend == "cuda" else "mps" if backend == "mps" else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = QwenVLModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()

    test_messages = [
        {"role": "user", "content": [{"type": "text", "text": "Describe this image."}]}
    ]
    text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    result = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n  Q: Describe this image.")
    print(f"  A: {result}")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-2B Vision-Language Fine-tuning (Unsloth + MLX)")

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
    elif backend == "cuda":
        train_cuda(config, args)
    elif backend in ("mps", "cpu"):
        train_mps(config, args)
    else:
        print(f"[!] Unknown backend: {backend}")
        sys.exit(1)


if __name__ == "__main__":
    main()
