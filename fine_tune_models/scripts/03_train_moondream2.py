#!/usr/bin/env python3
"""
Moondream 2 (Specialist) Fine-tuning Script
1.8B params — QLoRA (4-bit) for CUDA, LoRA for MLX

Usage:
    python 03_train_moondream2.py                  # Auto-detect backend
    python 03_train_moondream2.py --backend mlx    # Force MLX
    python 03_train_moondream2.py --backend cuda   # Force CUDA
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path="../configs/moondream2.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_backend():
    import torch
    if torch.cuda.is_available():
        print(f"[*] CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mlx"
    return "cpu"


def train_cuda(config):
    """QLoRA fine-tune Moondream 2 on CUDA. ~8GB VRAM."""
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    model_path = config["model"]["local_path"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    print(f"\n[*] Loading Moondream 2 with 4-bit quantization...")

    # 4-bit quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=train_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
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
    print(f"  [*] QLoRA: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total")

    # Load data
    print("[*] Loading training data...")
    data_files = list(Path(config["data"]["train"]).parent.glob(
        Path(config["data"]["train"]).name))

    if not data_files:
        print("[!] No training data. Run: python 01_prepare_data.py --download-datasets")
        sys.exit(1)

    dataset = load_dataset("json", data_files=[str(f) for f in data_files], split="train")
    max_samples = config["data"].get("max_samples")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    print(f"  [*] {len(dataset)} training examples")

    def tokenize_fn(examples):
        texts = []
        for messages in examples["messages"]:
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            texts.append(text)

        tokenized = tokenizer(
            texts, truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names, num_proc=4,
    )

    model_name = config["model"].get("short_name", "moondream2")
    output_dir = f"../models/lora_adapters/{model_name}-qlora"
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
        gradient_checkpointing=True,  # Save VRAM
        optim="paged_adamw_8bit",     # Memory-efficient optimizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\n[*] Starting QLoRA training → {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[OK] Moondream 2 training complete!")


def train_mlx(config):
    """LoRA fine-tune Moondream 2 on MLX (Apple Silicon)."""
    import subprocess

    model_path = config["model"]["local_path"]
    mlx_cfg = config.get("mlx", {})
    train_cfg = config["training"]
    model_name = config["model"].get("short_name", "moondream2")
    output_dir = f"../models/lora_adapters/{model_name}-mlx-lora"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare MLX data
    data_files = list(Path(config["data"]["train"]).parent.glob(
        Path(config["data"]["train"]).name))
    mlx_data = "../data/processed/moondream2_mlx_train.jsonl"

    with open(mlx_data, "w") as out:
        for f in data_files:
            with open(f) as inp:
                for line in inp:
                    row = json.loads(line)
                    text = ""
                    for msg in row.get("messages", []):
                        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                    if text:
                        out.write(json.dumps({"text": text}) + "\n")

    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", mlx_data,
        "--adapter-path", output_dir,
        "--batch-size", str(mlx_cfg.get("batch_size", 2)),
        "--lora-layers", str(mlx_cfg.get("lora_layers", 12)),
        "--iters", str(train_cfg["epochs"] * 1000),
        "--learning-rate", str(train_cfg["learning_rate"]),
    ]

    print(f"\n[*] MLX LoRA training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[OK] MLX training complete! → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Moondream 2 Fine-tuning")
    parser.add_argument("--config", default="../configs/moondream2.yaml")
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
