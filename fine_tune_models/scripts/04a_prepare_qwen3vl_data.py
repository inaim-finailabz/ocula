#!/usr/bin/env python3
"""
Qwen3-VL-2B Vision Training Data Preparation
==============================================
Downloads & converts vision-language datasets into the format Qwen3-VL expects.
Qwen3-VL uses ChatML-style templates with native vision token handling.

Qwen3-VL focuses on:
  1. High-resolution image understanding (dynamic resolution)
  2. Document analysis and OCR
  3. Chart / diagram / infographic reasoning
  4. Multi-turn visual conversation

Output format (JSONL):
  {
    "messages": [
      {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ],
    "images": ["path/to/image.jpg"]
  }

Datasets:
  • VQAv2          — Visual QA (short answers)
  • TextCaps       — OCR-aware captioning
  • DocVQA         — Document question answering
  • ChartQA        — Chart understanding
  • InfoVQA        — Infographic QA
  • AI2D           — Science diagram understanding
  • ScienceQA      — Multimodal science reasoning

Usage:
    python 04a_prepare_qwen3vl_data.py                   # download all
    python 04a_prepare_qwen3vl_data.py --max-samples 10000
    python 04a_prepare_qwen3vl_data.py --datasets docvqa chartqa
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional


def save_image(image, save_path: Path) -> Optional[str]:
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(save_path), "JPEG", quality=85)
        return str(save_path)
    except Exception:
        return None


def make_vision_example(question: str, answer: str, image_path: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ],
        "images": [image_path],
    }


# ─────────────────────────────────────────────────────────────────
# Dataset Converters
# ─────────────────────────────────────────────────────────────────

def convert_vqav2(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading VQAv2...")
    ds = load_dataset("merve/vqav2-small", split="validation")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "vqav2"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        question = row.get("question", "")
        answer = row.get("multiple_choice_answer", "")
        if not answer:
            answers = row.get("answers", [])
            if answers:
                answer = max(set(a.get("answer", "") for a in answers),
                             key=lambda x: sum(1 for a in answers if a.get("answer") == x))
        if question and answer:
            examples.append(make_vision_example(question, answer, img_path))
    return examples


def convert_textcaps(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading TextCaps...")
    try:
        ds = load_dataset("HuggingFaceM4/TextCaps", split="train")
    except Exception:
        try:
            ds = load_dataset("lmms-lab/TextCaps", split="train")
        except Exception:
            print("  [WARN] TextCaps not available — skipping")
            return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "textcaps"
    prompts = [
        "What text is visible in this image?",
        "Describe what you see, including any text.",
        "Read and describe the text in this image.",
        "Caption this image, paying attention to any visible text.",
    ]

    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        caption = row.get("reference_strs", row.get("caption", ""))
        if isinstance(caption, list):
            caption = caption[0] if caption else ""
        if caption:
            examples.append(make_vision_example(prompts[i % len(prompts)], caption, img_path))
    return examples


def convert_docvqa(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading DocVQA...")
    try:
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="train")
    except ValueError:
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "docvqa"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        question = row.get("question", "")
        answers = row.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and answers else row.get("answer", "")
        if question and answer:
            examples.append(make_vision_example(question, str(answer), img_path))
    return examples


def convert_chartqa(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading ChartQA...")
    ds = load_dataset("HuggingFaceM4/ChartQA", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "chartqa"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        question = row.get("query", row.get("question", ""))
        answer = row.get("label", row.get("answer", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if question and answer:
            examples.append(make_vision_example(question, str(answer), img_path))
    return examples


def convert_infovqa(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading InfoVQA...")
    try:
        ds = load_dataset("lmms-lab/InfoVQA", split="train")
    except Exception:
        try:
            ds = load_dataset("vidore/infovqa_test_subsampled", split="test")
        except Exception:
            print("  [WARN] InfoVQA not available — skipping")
            return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "infovqa"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        question = row.get("question", "")
        answers = row.get("answers", row.get("answer", []))
        if isinstance(answers, list):
            answer = answers[0] if answers else ""
        else:
            answer = str(answers)
        if question and answer:
            examples.append(make_vision_example(question, str(answer), img_path))
    return examples


def convert_ai2d(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading AI2D...")
    ds = load_dataset("lmms-lab/ai2d", split="test")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "ai2d"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        question = row.get("question", "")
        answer = row.get("answer", "")
        # AI2D has multiple choice — build context
        options = row.get("options", [])
        if options and isinstance(options, list):
            opts_text = "\n".join(f"  {chr(65+j)}. {opt}" for j, opt in enumerate(options))
            question = f"{question}\n{opts_text}"
            if isinstance(answer, int) and answer < len(options):
                answer = options[answer]
        if question and answer:
            examples.append(make_vision_example(question, str(answer), img_path))
    return examples


def convert_scienceqa(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading ScienceQA...")
    try:
        ds = load_dataset("derek-thomas/ScienceQA", split="train")
    except Exception:
        print("  [WARN] ScienceQA not available — skipping")
        return []

    # Filter to only image-based questions
    ds = ds.filter(lambda x: x.get("image") is not None)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "scienceqa"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue

        question = row.get("question", "")
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)

        if choices and isinstance(choices, list):
            opts_text = "\n".join(f"  {chr(65+j)}. {opt}" for j, opt in enumerate(choices))
            question = f"{question}\n{opts_text}"
            if isinstance(answer_idx, int) and answer_idx < len(choices):
                answer = choices[answer_idx]
            else:
                answer = str(answer_idx)
        else:
            answer = str(answer_idx)

        # Include explanation if available (Qwen3-VL is the "thinker" model)
        explanation = row.get("solution", row.get("explanation", ""))
        if explanation:
            answer = f"{answer}\n\nExplanation: {explanation}"

        if question and answer:
            examples.append(make_vision_example(question, answer, img_path))
    return examples


CONVERTERS = {
    "vqav2":      convert_vqav2,
    "textcaps":   convert_textcaps,
    "docvqa":     convert_docvqa,
    "chartqa":    convert_chartqa,
    "infovqa":    convert_infovqa,
    "ai2d":       convert_ai2d,
    "scienceqa":  convert_scienceqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare vision training data for Qwen3-VL-2B")
    parser.add_argument("--datasets", nargs="+", choices=list(CONVERTERS.keys()))
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max samples per dataset (default: 10K)")
    parser.add_argument("--output-dir", default="../data/vision_qwen3vl")
    parser.add_argument("--image-dir", default="../data/vision_qwen3vl/images")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nDatasets for Qwen3-VL vision fine-tuning:")
        for k in CONVERTERS:
            print(f"  • {k}")
        return

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = {k: v for k, v in CONVERTERS.items()
                if not args.datasets or k in args.datasets}

    all_examples = []
    for ds_name, fn in selected.items():
        print(f"\n{'─' * 60}")
        print(f"  Processing: {ds_name}")
        print(f"{'─' * 60}")
        try:
            examples = fn(args.max_samples, image_dir)
            print(f"  → {len(examples)} examples")
            ds_file = output_dir / f"{ds_name}.jsonl"
            with open(ds_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            all_examples.extend(examples)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    random.seed(42)
    random.shuffle(all_examples)
    val_size = max(1, int(len(all_examples) * args.val_ratio))

    train_file = output_dir / "qwen3vl_vision_train.jsonl"
    val_file = output_dir / "qwen3vl_vision_val.jsonl"

    with open(train_file, "w") as f:
        for ex in all_examples[val_size:]:
            f.write(json.dumps(ex) + "\n")
    with open(val_file, "w") as f:
        for ex in all_examples[:val_size]:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'═' * 60}")
    print(f"  QWEN3-VL DATA COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Total:  {len(all_examples):,}")
    print(f"  Train:  {len(all_examples) - val_size:,}")
    print(f"  Val:    {val_size:,}")
    print(f"  Output: {train_file}")
    print(f"{'═' * 60}")
    print(f"\n  Next: python 04b_train_qwen3vl_vision.py --backend auto")


if __name__ == "__main__":
    main()
