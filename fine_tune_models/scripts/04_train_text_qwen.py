#!/usr/bin/env python3
"""
Qwen Text LLM Fine-tuning Script

Default target is Ocula Lite (Qwen3-1.7B) via config.

Usage:
    python 04_train_text_qwen.py
    python 04_train_text_qwen.py --backend mlx
    python 04_train_text_qwen.py --backend cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import Unsloth as early as possible (before transformers/peft) when available.
# This avoids partial patching warnings and inconsistent fast-forward hooks.
try:
    import unsloth  # noqa: F401
except Exception:
    pass

import yaml


def load_config(config_path="../configs/qwen25_1_5b_text.yaml"):
    config_file = Path(config_path).resolve()
    with open(config_file) as f:
        return yaml.safe_load(f), config_file.parent


def resolve_path(base_dir: Path, path_value: str) -> str:
    p = Path(path_value)
    return str(p if p.is_absolute() else (base_dir / p).resolve())


def resolve_training_files(train_pattern: str) -> list[Path]:
    files = list(Path(train_pattern).parent.glob(Path(train_pattern).name))
    if files:
        return files

    # Backward compatibility: some prep runs wrote to ../data/processed at repo root.
    legacy_marker = "/fine_tune_models/data/"
    if legacy_marker in train_pattern:
        legacy_pattern = train_pattern.replace(legacy_marker, "/data/", 1)
        legacy_files = list(Path(legacy_pattern).parent.glob(Path(legacy_pattern).name))
        if legacy_files:
            print(f"[*] Using legacy processed-data path: {Path(legacy_pattern).parent}")
            return legacy_files

    return []


def detect_backend():
    import torch
    if torch.cuda.is_available():
        print(f"[*] CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mlx"
    return "cpu"


def detect_acceleration_stack():
    stack = {"unsloth": False, "flash_attn": False}
    try:
        import unsloth  # noqa: F401
        stack["unsloth"] = True
    except Exception:
        pass
    try:
        import flash_attn  # noqa: F401
        stack["flash_attn"] = True
    except Exception:
        pass
    print(f"[*] Acceleration: unsloth={stack['unsloth']} flash_attn={stack['flash_attn']}")
    return stack


def should_use_unsloth(config: dict, accel: dict) -> bool:
    """Decide whether to use Unsloth for this run.

    training.use_unsloth:
      - true  => force Unsloth path
      - false => force Transformers Trainer path
      - auto  => choose safe default by model family
    """
    if not accel.get("unsloth", False):
        return False

    train_cfg = config.get("training", {})
    pref = str(train_cfg.get("use_unsloth", "auto")).strip().lower()
    if pref in ("true", "1", "yes", "on"):
        return True
    if pref in ("false", "0", "no", "off"):
        return False

    # Auto mode: Qwen3 text currently has known Unsloth compatibility breakages
    # in some version combos (missing Qwen3Attention.apply_qkv).
    model_name = str(config.get("model", {}).get("name", "")).lower()
    model_short = str(config.get("model", {}).get("short_name", "")).lower()
    model_path = str(config.get("model", {}).get("local_path", "")).lower()
    if "qwen3" in model_name or "qwen3" in model_short or "qwen3" in model_path:
        print(
            "[*] Unsloth auto-mode: disabled for Qwen3 model "
            "(known apply_qkv compatibility issue); using Transformers Trainer."
        )
        return False

    return True


def find_latest_checkpoint(output_dir: str) -> str | None:
    ckpt_root = Path(output_dir)
    if not ckpt_root.exists():
        return None
    checkpoints = sorted(ckpt_root.glob("checkpoint-*"), key=os.path.getmtime)
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def build_loss_stop_callback(train_cfg: dict):
    target_loss = train_cfg.get("early_stop_loss")
    target_pct = train_cfg.get("early_stop_loss_pct")
    min_steps = int(train_cfg.get("early_stop_min_steps", 0) or 0)

    if target_loss is None and target_pct is None:
        return None

    from transformers import TrainerCallback

    class LossStopCallback(TrainerCallback):
        def __init__(self, abs_target, pct_target, min_steps_):
            self.abs_target = float(abs_target) if abs_target is not None else None
            self.pct_target = float(pct_target) if pct_target is not None else None
            self.min_steps = int(min_steps_)
            self.initial_loss = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return control
            if state.global_step < self.min_steps:
                return control
            try:
                loss = float(logs["loss"])
            except Exception:
                return control

            if self.initial_loss is None and loss > 0:
                self.initial_loss = loss

            stop = False
            reason = None

            if self.abs_target is not None and loss <= self.abs_target:
                stop = True
                reason = f"loss {loss:.6f} <= early_stop_loss {self.abs_target:.6f}"

            if not stop and self.pct_target is not None and self.initial_loss:
                pct_threshold = self.initial_loss * (self.pct_target / 100.0)
                if loss <= pct_threshold:
                    stop = True
                    reason = (
                        f"loss {loss:.6f} <= {self.pct_target:.2f}% of initial loss "
                        f"({pct_threshold:.6f})"
                    )

            if stop:
                print(f"[EARLY-STOP] Step {state.global_step}: {reason}")
                control.should_training_stop = True
            return control

    return LossStopCallback(target_loss, target_pct, min_steps)


def train_cuda_unsloth(config, config_dir: Path, resume: str | None = None):
    """QLoRA fine-tune using Unsloth for faster CUDA training when available."""
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    model_path = resolve_path(config_dir, config["model"]["local_path"])
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    print(f"\n[*] Loading model with Unsloth (4-bit): {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=train_cfg["max_seq_length"],
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("[*] Loading training data...")
    train_pattern = resolve_path(config_dir, config["data"]["train"])
    data_files = resolve_training_files(train_pattern)
    if not data_files:
        print("[!] No training data. Run: python 01_prepare_data.py --download-datasets")
        sys.exit(1)

    dataset = load_dataset("json", data_files=[str(f) for f in data_files], split="train")
    max_samples = config["data"].get("max_samples")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    def format_chatml(ex):
        text = ""
        for msg in ex.get("messages", []):
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        return {"text": text}

    dataset = dataset.map(format_chatml)

    model_name = config["model"].get("short_name", "qwen_text")
    output_dir = str((config_dir / "../models/lora_adapters" / f"{model_name}-qlora").resolve())
    logging_dir = str((config_dir / "../logs" / model_name).resolve())
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    save_steps = train_cfg.get("save_steps", 500)
    save_total_limit = train_cfg.get("save_total_limit", 3)
    loss_stop_cb = build_loss_stop_callback(train_cfg)
    callbacks = [loss_stop_cb] if loss_stop_cb else []

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        callbacks=callbacks,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            save_strategy="steps",
            report_to=["tensorboard"],
            logging_dir=logging_dir,
            bf16=True,
            max_seq_length=train_cfg["max_seq_length"],
        ),
    )

    print(f"\n[*] Starting Unsloth QLoRA training → {output_dir}")
    resume_ckpt = resume or find_latest_checkpoint(output_dir)
    print(f"    Checkpoints: every {save_steps} steps, keep {save_total_limit}")
    if train_cfg.get("early_stop_loss") is not None or train_cfg.get("early_stop_loss_pct") is not None:
        print(
            "    Early-stop: "
            f"loss<={train_cfg.get('early_stop_loss')} "
            f"or loss<={train_cfg.get('early_stop_loss_pct')}% initial "
            f"(min_steps={int(train_cfg.get('early_stop_min_steps', 0) or 0)})"
        )
    if resume_ckpt:
        print(f"    Resuming from: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[OK] Unsloth training complete!")


def train_cuda(config, config_dir: Path, resume: str | None = None):
    """QLoRA fine-tune Qwen text model on CUDA."""
    import torch
    from datasets import load_dataset

    model_path = resolve_path(config_dir, config["model"]["local_path"])
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    accel = detect_acceleration_stack()

    if should_use_unsloth(config, accel):
        try:
            return train_cuda_unsloth(config, config_dir, resume=resume)
        except Exception as e:
            if "apply_qkv" in str(e):
                print(
                    "[WARN] Unsloth Qwen3 patch failed (missing apply_qkv). "
                    "Falling back to Transformers Trainer automatically."
                )
            else:
                print(f"[WARN] Unsloth path failed, falling back to Transformers Trainer: {e}")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    print(f"\n[*] Loading Qwen text model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=train_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if accel["flash_attn"] else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
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
    train_pattern = resolve_path(config_dir, config["data"]["train"])
    data_files = resolve_training_files(train_pattern)

    if not data_files:
        print("[!] No training data. Run: python 01_prepare_data.py --download-datasets")
        sys.exit(1)

    dataset = load_dataset("json", data_files=[str(f) for f in data_files], split="train")
    max_samples = config["data"].get("max_samples")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    print(f"  [*] {len(dataset)} training examples")

    def tokenize_fn(examples):
        """Qwen3 uses ChatML format natively."""
        texts = []
        for messages in examples["messages"]:
            # Qwen3's native ChatML
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

    model_name = config["model"].get("short_name", "qwen3vl")
    output_dir = str((config_dir / "../models/lora_adapters" / f"{model_name}-qlora").resolve())
    logging_dir = str((config_dir / "../logs" / model_name).resolve())
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    save_steps = train_cfg.get("save_steps", 500)
    save_total_limit = train_cfg.get("save_total_limit", 3)
    loss_stop_cb = build_loss_stop_callback(train_cfg)
    callbacks = [loss_stop_cb] if loss_stop_cb else []
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
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy="steps",
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )

    print(f"\n[*] Starting QLoRA training → {output_dir}")
    print(f"    Epochs: {train_cfg['epochs']}, Batch: {train_cfg['batch_size']}, "
          f"Grad Accum: {train_cfg['gradient_accumulation']}")
    print(f"    Effective batch: {train_cfg['batch_size'] * train_cfg['gradient_accumulation']}")
    print(f"    Checkpoints: every {save_steps} steps, keep {save_total_limit}")
    if train_cfg.get("early_stop_loss") is not None or train_cfg.get("early_stop_loss_pct") is not None:
        print(
            "    Early-stop: "
            f"loss<={train_cfg.get('early_stop_loss')} "
            f"or loss<={train_cfg.get('early_stop_loss_pct')}% initial "
            f"(min_steps={int(train_cfg.get('early_stop_min_steps', 0) or 0)})"
        )
    resume_ckpt = resume or find_latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"    Resuming from: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[OK] Qwen text model training complete!")


def train_mlx(config, config_dir: Path):
    """LoRA fine-tune Qwen text model on MLX."""
    import subprocess

    model_path = resolve_path(config_dir, config["model"]["local_path"])
    mlx_cfg = config.get("mlx", {})
    train_cfg = config["training"]
    model_name = config["model"].get("short_name", "qwen3vl")
    output_dir = str((config_dir / "../models/lora_adapters" / f"{model_name}-mlx-lora").resolve())
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    train_pattern = resolve_path(config_dir, config["data"]["train"])
    data_files = resolve_training_files(train_pattern)
    mlx_data = str((config_dir / "../data/processed/qwen_text_mlx_train.jsonl").resolve())

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
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", mlx_data,
        "--adapter-path", output_dir,
        "--batch-size", str(mlx_cfg.get("batch_size", 1)),
        "--lora-layers", str(mlx_cfg.get("lora_layers", 8)),
        "--iters", str(train_cfg["epochs"] * 1500),
        "--learning-rate", str(train_cfg["learning_rate"]),
    ]

    print(f"\n[*] MLX LoRA training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[OK] MLX training complete! → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Qwen Text LLM Fine-tuning")
    parser.add_argument("--config", default="../configs/qwen25_1_5b_text.yaml")
    parser.add_argument("--backend", choices=["cuda", "mlx", "auto"], default="auto")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume from. If omitted, auto-resumes from latest checkpoint in output dir.")
    args = parser.parse_args()

    config, config_dir = load_config(args.config)
    backend = args.backend if args.backend != "auto" else detect_backend()

    if backend == "mlx":
        train_mlx(config, config_dir)
    else:
        train_cuda(config, config_dir, resume=args.resume)


if __name__ == "__main__":
    main()
