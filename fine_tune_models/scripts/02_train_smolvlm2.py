#!/usr/bin/env python3
"""
SmolVLM2-500M Fine-tuning Script
Supports: CUDA (RTX 4070 Super Ti) + MLX (Apple Silicon M1+)

Usage:
    # CUDA (default)
    python 02_train_smolvlm2.py

    # MLX (Apple Silicon)
    python 02_train_smolvlm2.py --backend mlx

    # Custom config
    python 02_train_smolvlm2.py --config ../configs/smolvlm2.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path="../configs/smolvlm2.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_backend():
    """Auto-detect best available backend."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[*] CUDA: {gpu_name} ({vram_gb:.1f} GB)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[*] MPS (Apple Silicon) detected — using MLX for training")
        return "mlx"
    print("[!] No GPU — falling back to CPU (will be slow)")
    return "cpu"


# ─────────────────────────────────────────────────────────────────
# CUDA / PyTorch Training Path
# ─────────────────────────────────────────────────────────────────

def train_cuda(config):
    """Fine-tune SmolVLM2-500M using HuggingFace Transformers + PEFT."""
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    model_path = config["model"]["local_path"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    print(f"\n[*] Loading model: {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For full fine-tune of 500M model
    if train_cfg["method"] == "full":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"  [*] Full fine-tune: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    else:
        # LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  [*] LoRA: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total "
              f"({100 * trainable / total:.1f}%)")

    # Load and tokenize dataset
    print("[*] Loading training data...")
    data_pattern = config["data"]["train"]
    data_dir = str(Path(data_pattern).parent)
    data_files = list(Path(data_dir).glob(Path(data_pattern).name))

    if not data_files:
        print(f"[!] No training data found matching: {data_pattern}")
        print("    Run: python 01_prepare_data.py --download-datasets first")
        sys.exit(1)

    dataset = load_dataset("json", data_files=[str(f) for f in data_files], split="train")
    max_samples = config["data"].get("max_samples")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    print(f"  [*] {len(dataset)} training examples")

    def tokenize_fn(examples):
        """Convert ChatML messages to tokenized input."""
        texts = []
        for messages in examples["messages"]:
            # Build ChatML string
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    # Training arguments
    model_name = config["model"].get("short_name", "smolvlm2")
    output_dir = f"../models/lora_adapters/{model_name}-{train_cfg['method']}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to=["tensorboard"],
        logging_dir=f"../logs/{model_name}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\n[*] Starting training → {output_dir}")
    print(f"    Epochs: {train_cfg['epochs']}, Batch: {train_cfg['batch_size']}, "
          f"LR: {train_cfg['learning_rate']}")
    trainer.train()

    # Save
    print(f"\n[*] Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[OK] SmolVLM2-500M training complete!")


# ─────────────────────────────────────────────────────────────────
# MLX Training Path (Apple Silicon)
# ─────────────────────────────────────────────────────────────────

def train_mlx(config):
    """Fine-tune SmolVLM2-500M using MLX on Apple Silicon."""
    import subprocess

    model_path = config["model"]["local_path"]
    mlx_cfg = config.get("mlx", {})
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    model_name = config["model"].get("short_name", "smolvlm2")
    output_dir = f"../models/lora_adapters/{model_name}-mlx-lora"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data file for mlx-lm
    data_pattern = config["data"]["train"]
    data_dir = str(Path(data_pattern).parent)
    data_files = list(Path(data_dir).glob(Path(data_pattern).name))

    if not data_files:
        print(f"[!] No training data found matching: {data_pattern}")
        sys.exit(1)

    # MLX-LM expects a single JSONL with {"text": "..."} format
    mlx_data_file = "../data/processed/smolvlm2_mlx_train.jsonl"
    print("[*] Preparing MLX training data...")
    with open(mlx_data_file, "w") as out:
        for data_file in data_files:
            with open(data_file) as f:
                for line in f:
                    row = json.loads(line)
                    # Convert ChatML to plain text
                    text = ""
                    for msg in row.get("messages", []):
                        role = msg["role"]
                        content = msg["content"]
                        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    if text:
                        out.write(json.dumps({"text": text}) + "\n")

    # Run MLX LoRA training
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", mlx_data_file,
        "--adapter-path", output_dir,
        "--batch-size", str(mlx_cfg.get("batch_size", 4)),
        "--lora-layers", str(mlx_cfg.get("lora_layers", 16)),
        "--iters", str(train_cfg["epochs"] * 1000),  # Rough conversion
        "--learning-rate", str(train_cfg["learning_rate"]),
    ]

    print(f"\n[*] Running MLX LoRA training:")
    print(f"    {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"\n[OK] MLX training complete! Adapters saved to: {output_dir}")
    print(f"     Next: python 06_merge_lora.py --model {model_name} --backend mlx")


def main():
    parser = argparse.ArgumentParser(description="SmolVLM2-500M Fine-tuning")
    parser.add_argument("--config", default="../configs/smolvlm2.yaml")
    parser.add_argument("--backend", choices=["cuda", "mlx", "auto"], default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    backend = args.backend if args.backend != "auto" else detect_backend()

    if backend == "mlx":
        train_mlx(config)
    else:
        train_cuda(config)


if __name__ == "__main__":
    main()
