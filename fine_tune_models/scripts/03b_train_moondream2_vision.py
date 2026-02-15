#!/usr/bin/env python3
"""
Moondream2 Vision-Language Fine-tuning
=======================================
Proper multimodal SFT using the processor (image + text jointly).

Supports:
  • CUDA  — Unsloth FastVisionModel + SFTTrainer (fast, handles 4-bit natively)
  • MLX   — mlx-vlm LoRA fine-tuning on Apple Silicon
  • MPS   — PyTorch MPS fallback (slower than MLX on Apple Silicon)

Moondream2 architecture notes:
  - Uses SigLIP vision encoder (378×378 images)
  - Custom Phi-1.5 based language model
  - Model class: AutoModelForCausalLM with integrated vision support

Prerequisites:
  # For CUDA (recommended):
  pip install unsloth trl datasets pillow einops
  # For MLX:
  pip install mlx-vlm>=0.1.2

Usage:
    python 03b_train_moondream2_vision.py
    python 03b_train_moondream2_vision.py --backend cuda --use-4bit
    python 03b_train_moondream2_vision.py --backend mlx
    python 03b_train_moondream2_vision.py --resume checkpoint-500
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
# CUDA Training via Unsloth (handles 4-bit, LoRA, peft internally)
# ═════════════════════════════════════════════════════════════════

def train_cuda(config: dict, args):
    """
    Moondream2 vision LoRA fine-tuning using Unsloth + SFTTrainer.

    Unsloth handles:
      - 4-bit quantization (no manual BitsAndBytesConfig)
      - LoRA injection (no manual get_peft_model)
      - Gradient checkpointing
      - 2x faster training with fused kernels
    """
    import torch
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name
    revision = config.get("model", {}).get("revision", "2025-01-09")

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    # ── Load model with Unsloth ──
    # Try unsloth-optimized variant first, fall back to HF path
    unsloth_model = f"unsloth/{model_name.split('/')[-1]}"
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
        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_path,
                load_in_4bit=args.use_4bit,
                use_gradient_checkpointing="unsloth",
            )
        except Exception as e2:
            print(f"[!] FastVisionModel fallback also failed ({e2}).")
            print("[*] Falling back to Transformers+PEFT CUDA training path.")
            return train_cuda_hf(config, args)

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
        """Convert our JSONL format to Unsloth vision format with file paths.
        Unsloth natively supports local file paths — no need to pre-load PIL images."""
        messages = sample.get("messages", [])
        image_paths = sample.get("images", [])

        img = image_paths[0] if image_paths else None

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
    batch_size = train_cfg.get("batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    lr = train_cfg.get("learning_rate", 2e-4)
    max_steps = train_cfg.get("max_steps", -1)
    if args.max_iters:
        max_steps = args.max_iters

    output_dir = args.output or DEFAULT_OUTPUT

    # ── Checkpoint settings ──
    save_steps = train_cfg.get("save_steps", 100)
    save_total_limit = train_cfg.get("save_total_limit", 5)

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
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy="steps",
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        report_to=["tensorboard"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        optim="adamw_8bit",
        max_grad_norm=1.0,
        max_seq_length=train_cfg.get("max_seq_length", 2048),
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

    print(f"\n[*] Starting Moondream2 vision SFT via Unsloth on CUDA")
    print(f"    Epochs: {epochs}, Batch: {batch_size} × {grad_accum} accum, LR: {lr}")
    print(f"    Max steps: {max_steps if max_steps and max_steps > 0 else 'unlimited'}")
    print(f"    4-bit: {args.use_4bit}")
    print(f"    Checkpoints: every {save_steps} steps, keep {save_total_limit}")
    print(f"    Output: {output_dir}")

    # ── Auto-resume from latest checkpoint if available ──
    resume_ckpt = args.resume
    if not resume_ckpt:
        ckpt_dir = Path(output_dir)
        if ckpt_dir.exists():
            checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=os.path.getmtime)
            if checkpoints:
                resume_ckpt = str(checkpoints[-1])
                print(f"    Auto-resuming from: {resume_ckpt}")

    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    # ── Save LoRA adapters ──
    print(f"\n[*] Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Save merged model (pre-quantization) for future fine-tuning ──
    merged_dir = str(Path(output_dir).parent / "moondream2-vision-merged")
    print(f"\n[*] Saving merged model (base + LoRA) to {merged_dir}")
    print(f"    This allows future fine-tuning without retraining from scratch.")

    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    print("[OK] Moondream2 vision fine-tuning complete!")
    print(f"     LoRA adapters:  {output_dir}")
    print(f"     Merged model:   {merged_dir}")
    print(f"     Next: python 07b_export_moondream2_gguf.py")


def train_cuda_hf(config: dict, args):
    """CUDA fallback using Transformers + PEFT when Unsloth loader is unavailable."""
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name
    revision = config.get("model", {}).get("revision", "main")

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    print(f"\n[*] Loading model via Transformers: {model_path} (revision: {revision})")
    processor = AutoProcessor.from_pretrained(
        model_path, revision=revision, trust_remote_code=True
    )

    model_kwargs = {
        "revision": revision,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
    }
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        except Exception as e:
            print(f"[WARN] 4-bit unavailable ({e}), continuing in 16-bit.")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")

    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2",
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

    class MoondreamVisionCollator:
        def __init__(self, processor, max_length=2048):
            self.processor = processor
            self.max_length = max_length

        def __call__(self, examples):
            texts = []
            images_list = []
            for ex in examples:
                messages = ex.get("messages", [])
                image_paths = ex.get("images", [])
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
                text = f"<image>\n\nQuestion: {user_text}\n\nAnswer: {assistant_text}"
                texts.append(text)
                imgs = []
                for img_path in image_paths:
                    try:
                        imgs.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (378, 378), (128, 128, 128)))
                images_list.append(imgs if imgs else None)

            try:
                if any(imgs is not None for imgs in images_list):
                    batch = self.processor(
                        text=texts,
                        images=[imgs[0] if imgs else None for imgs in images_list],
                        return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
                    )
                else:
                    batch = self.processor(
                        text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
                    )
            except Exception:
                batch = self.processor.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
                )
            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            return batch

    collator = MoondreamVisionCollator(processor, max_length=train_cfg.get("max_seq_length", 2048))

    epochs = train_cfg.get("epochs", 1)
    batch_size = train_cfg.get("batch_size", 2)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    lr = train_cfg.get("learning_rate", 1e-4)
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
        warmup_steps=50,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 5),
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

    print(f"\n[*] Starting Moondream2 vision SFT on CUDA (Transformers fallback)")
    print(f"    Epochs: {epochs}, Batch: {batch_size} x {grad_accum} accum, LR: {lr}")
    print(f"    Output: {output_dir}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    print(f"\n[*] Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("[OK] Moondream2 vision fine-tuning complete (Transformers fallback)!")


# ═════════════════════════════════════════════════════════════════
# MPS Training (fallback for Apple Silicon without MLX)
# ═════════════════════════════════════════════════════════════════

def train_mps(config: dict, args):
    """MPS fallback — uses HuggingFace Transformers + PEFT directly."""
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    model_name = config.get("model", {}).get("name", DEFAULT_MODEL)
    local_path = config.get("model", {}).get("local_path", "")
    model_path = local_path if local_path and Path(local_path).exists() else model_name
    revision = config.get("model", {}).get("revision", "2025-01-09")

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    print(f"\n[*] Loading model: {model_path} (revision: {revision})")
    processor = AutoProcessor.from_pretrained(
        model_path, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to("mps")
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")

    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2",
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

    class MoondreamVisionCollator:
        def __init__(self, processor, max_length=2048):
            self.processor = processor
            self.max_length = max_length

        def __call__(self, examples):
            texts = []
            images_list = []
            for ex in examples:
                messages = ex.get("messages", [])
                image_paths = ex.get("images", [])
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
                text = f"<image>\n\nQuestion: {user_text}\n\nAnswer: {assistant_text}"
                texts.append(text)
                imgs = []
                for img_path in image_paths:
                    try:
                        imgs.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (378, 378), (128, 128, 128)))
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

    collator = MoondreamVisionCollator(processor, max_length=train_cfg.get("max_seq_length", 2048))

    epochs = train_cfg.get("epochs", 1)
    batch_size = min(train_cfg.get("batch_size", 4), 2)
    grad_accum = max(train_cfg.get("gradient_accumulation", 4), 8)
    lr = train_cfg.get("learning_rate", 1e-4)
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

    print(f"\n[*] Starting Moondream2 vision SFT on MPS")
    print(f"    Epochs: {epochs}, Batch: {batch_size} × {grad_accum} accum, LR: {lr}")
    print(f"    Output: {output_dir}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    print(f"\n[*] Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("[OK] Moondream2 vision fine-tuning complete!")


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
        "--apply-chat-template",
    ]

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
        description="Moondream2 Vision-Language Fine-tuning (Unsloth + MLX)")

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
