#!/usr/bin/env python3
"""
Moondream2 Vision Training Data Preparation
=============================================
Downloads & converts vision-language datasets into the format Moondream2 expects.
Moondream2 uses its own chat template (not ChatML).

Moondream2 focuses on:
  1. Concise image captioning
  2. Visual question answering
  3. Object counting / detection (text-based)

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
  • COCO Captions  — Standard image captioning (concise)
  • RefCOCO        — Region-based descriptions
  • GQA            — Compositional scene understanding

Usage:
    python 03a_prepare_moondream_data.py                  # download all
    python 03a_prepare_moondream_data.py --max-samples 5000
    python 03a_prepare_moondream_data.py --list
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

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
        # HuggingFaceM4/TextCaps is script-based and often blocked by newer datasets versions.
        ds = load_dataset("lmms-lab/TextCaps", split="train")
    except Exception as e:
        try:
            ds = load_dataset("HuggingFaceM4/TextCaps", split="train")
        except Exception as e2:
            print(f"  [WARN] TextCaps unavailable ({e}; {e2}) — skipping")
            return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "textcaps"
    prompts = [
        "What text is visible in this image?",
        "Describe what you see, including any text.",
        "Read the text in this image.",
        "Caption this image, noting any visible text.",
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


def convert_coco_captions(max_samples: int, image_dir: Path):
    """COCO captions — short, clean image descriptions."""
    from datasets import load_dataset

    print("[*] Loading COCO Captions...")
    try:
        ds = load_dataset("HuggingFaceM4/COCO", split="train")
    except Exception as e1:
        try:
            # Keep fallback to non-script dataset only. trust_remote_code is removed.
            ds = load_dataset("yerevann/coco-karpathy", split="train")
        except Exception as e2:
            print(f"  [WARN] COCO not available — skipping ({e1}; {e2})")
            return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "coco"
    prompts = [
        "Describe this image.",
        "What do you see in this image?",
        "Caption this image briefly.",
        "Provide a short description of this image.",
    ]

    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue

        # COCO can have multiple captions
        captions = row.get("sentences", row.get("captions", row.get("caption", "")))
        if isinstance(captions, list):
            if captions and isinstance(captions[0], dict):
                caption = captions[0].get("raw", captions[0].get("text", ""))
            else:
                caption = captions[0] if captions else ""
        elif isinstance(captions, str):
            caption = captions
        else:
            continue

        if caption and len(caption) > 10:
            examples.append(make_vision_example(prompts[i % len(prompts)], caption, img_path))
    return examples


def convert_gqa(max_samples: int, image_dir: Path):
    """GQA — compositional visual reasoning."""
    from datasets import load_dataset

    print("[*] Loading GQA...")
    try:
        ds = load_dataset("merve/gqa-small", split="train")
    except Exception:
        try:
            ds = load_dataset("leonardlin/GQA", split="train")
        except Exception:
            print("  [WARN] GQA not available — skipping")
            return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "gqa"
    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue

        question = row.get("question", "")
        answer = row.get("answer", row.get("fullAnswer", ""))
        if question and answer:
            examples.append(make_vision_example(question, answer, img_path))
    return examples


def convert_docvqa(max_samples: int, image_dir: Path):
    from datasets import load_dataset

    print("[*] Loading DocVQA...")
    ds = None
    split_used = None
    for cfg in ("DocVQA", "InfographicVQA"):
        for split in ("train", "validation", "test"):
            try:
                ds = load_dataset("lmms-lab/DocVQA", cfg, split=split)
                split_used = f"{cfg}/{split}"
                break
            except Exception:
                continue
        if ds is not None:
            break
    if ds is None:
        print("  [WARN] DocVQA not available — skipping")
        return []
    if not split_used.endswith("/train"):
        print(f"  [WARN] DocVQA train split unavailable, using '{split_used}'")
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "docvqa"
    for i, row in enumerate(ds):
        img = row.get("image")
        if img is None:
            continue
        img_path = save_image(img, img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue
        question = row.get("question", row.get("query", ""))
        answers = row.get("answers", [])
        answer = (
            answers[0]
            if isinstance(answers, list) and answers
            else row.get("answer", row.get("label", ""))
        )
        if question and answer:
            examples.append(make_vision_example(question, str(answer), img_path))
    return examples


CONVERTERS = {
    "vqav2":        convert_vqav2,
    "textcaps":     convert_textcaps,
    "coco":         convert_coco_captions,
    "gqa":          convert_gqa,
    "docvqa":       convert_docvqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare vision training data for Moondream2")
    parser.add_argument("--datasets", nargs="+", choices=list(CONVERTERS.keys()),
                        help="Specific datasets")
    parser.add_argument("--max-samples", type=int, default=8000,
                        help="Max samples per dataset (default: 8K)")
    parser.add_argument("--output-dir", default="../data/vision_moondream")
    parser.add_argument("--image-dir", default="../data/vision_moondream/images")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nDatasets for Moondream2 vision fine-tuning:")
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
    if not all_examples:
        print("\n[!] No training examples were prepared. Check dataset availability/network.")
        sys.exit(2)

    val_size = max(1, int(len(all_examples) * args.val_ratio))

    train_file = output_dir / "moondream2_vision_train.jsonl"
    val_file = output_dir / "moondream2_vision_val.jsonl"

    with open(train_file, "w") as f:
        for ex in all_examples[val_size:]:
            f.write(json.dumps(ex) + "\n")
    with open(val_file, "w") as f:
        for ex in all_examples[:val_size]:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'═' * 60}")
    print(f"  MOONDREAM2 DATA COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Total:  {len(all_examples):,}")
    print(f"  Train:  {len(all_examples) - val_size:,}")
    print(f"  Val:    {val_size:,}")
    print(f"  Output: {train_file}")
    print(f"{'═' * 60}")
    print(f"\n  Next: python 03b_train_moondream2_vision.py --backend auto")


if __name__ == "__main__":
    main()
