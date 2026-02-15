#!/usr/bin/env python3
"""
SmolVLM2 Vision Training Data Preparation
==========================================
Downloads & converts vision-language datasets into the format SmolVLM2 expects
for multimodal SFT (supervised fine-tuning).

Targets two weak areas:
  1. Image detection / captioning / visual QA
  2. Document understanding / OCR / summarisation

Output format (JSONL):
  {
    "messages": [
      {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ],
    "images": ["path/to/image.jpg"]
  }

Datasets downloaded:
  ── Image Understanding ──
  • LLaVA-Instruct-150K  (150K image+instruction pairs, CC BY 4.0)
  • VQAv2-small          (visual QA, CC BY 4.0)
  • TextCaps             (OCR-infused captioning, CC BY 4.0)
  • RefCOCO              (object detection & grounding)

  ── Document Understanding ──
  • DocVQA               (document visual QA, key for form/invoice reading)
  • ChartQA              (chart interpretation)
  • InfographicVQA       (infographic understanding)

Usage:
    python 02a_prepare_vision_data.py                       # download all
    python 02a_prepare_vision_data.py --task image           # image datasets only
    python 02a_prepare_vision_data.py --task document        # document datasets only
    python 02a_prepare_vision_data.py --max-samples 10000    # cap per dataset
    python 02a_prepare_vision_data.py --list                 # show available datasets
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────────────────────────────

VISION_DATASETS = {
    # ── Image Understanding ──
    "llava_instruct_150k": {
        "hf_name": "liuhaotian/LLaVA-Instruct-150K",
        "task": "image",
        "description": "150K GPT-4V generated image+instruction pairs (uses COCO images)",
        "size": "~200 MB (text) + COCO images",
        "license": "CC BY 4.0",
        "needs_coco": True,
    },
    "vqav2": {
        "hf_name": "merve/vqav2-small",
        "task": "image",
        "description": "Visual question answering — short answers about image content",
        "size": "~500 MB",
        "license": "CC BY 4.0",
        "needs_coco": False,
    },
    "textcaps": {
        "hf_name": "HuggingFaceM4/TextCaps",
        "task": "image",
        "description": "Captions that include text visible in images (OCR + captioning)",
        "size": "~300 MB",
        "license": "CC BY 4.0",
        "needs_coco": False,
    },
    "image_paragraph_captioning": {
        "hf_name": "merve/image-paragraph-captioning",
        "task": "image",
        "description": "Long-form paragraph captions for images (multi-sentence description)",
        "size": "~100 MB",
        "license": "CC BY 4.0",
        "needs_coco": False,
    },
    # ── Document Understanding ──
    "docvqa": {
        "hf_name": "lmms-lab/DocVQA",
        "task": "document",
        "description": "12K+ document images with questions about content (forms, invoices, reports)",
        "size": "~2 GB",
        "license": "Apache 2.0",
        "needs_coco": False,
    },
    "chartqa": {
        "hf_name": "HuggingFaceM4/ChartQA",
        "task": "document",
        "description": "Chart/graph understanding — questions about plotted data",
        "size": "~500 MB",
        "license": "GPL-3.0",
        "needs_coco": False,
    },
    "infovqa": {
        "hf_name": "lmms-lab/InfoVQA",
        "task": "document",
        "description": "Infographic understanding — complex visual documents",
        "size": "~1 GB",
        "license": "Apache 2.0",
        "needs_coco": False,
    },
    "ai2d": {
        "hf_name": "lmms-lab/ai2d",
        "task": "document",
        "description": "Science diagrams — questions about labeled diagrams",
        "size": "~500 MB",
        "license": "Apache 2.0",
        "needs_coco": False,
    },
}

# COCO images for LLaVA-Instruct (train2017)
COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"


def list_datasets():
    """Print available datasets."""
    print("\n" + "=" * 80)
    print("VISION-LANGUAGE DATASETS FOR SmolVLM2 FINE-TUNING")
    print("=" * 80)
    for task in ["image", "document"]:
        print(f"\n{'─' * 80}")
        print(f"  {task.upper()} UNDERSTANDING")
        print(f"{'─' * 80}")
        for key, ds in VISION_DATASETS.items():
            if ds["task"] == task:
                print(f"\n  {key}")
                print(f"    HF: {ds['hf_name']}")
                print(f"    {ds['description']}")
                print(f"    Size: {ds['size']} | License: {ds['license']}")
    print("\n" + "=" * 80)


def save_image(image, save_path: Path) -> Optional[str]:
    """Save a PIL Image and return the path, or None on failure."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(save_path), "JPEG", quality=85)
        return str(save_path)
    except Exception as e:
        return None


def make_vision_example(question: str, answer: str, image_path: str) -> dict:
    """Create a SmolVLM2-compatible multimodal training example."""
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


def make_text_example(question: str, answer: str) -> dict:
    """Fallback text-only example (no image)."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# ─────────────────────────────────────────────────────────────────
# Dataset Converters
# ─────────────────────────────────────────────────────────────────

def convert_vqav2(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert VQAv2 small dataset."""
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
        # VQAv2 can have multiple answers — take the most common
        answer = row.get("multiple_choice_answer", "")
        if not answer:
            answers = row.get("answers", [])
            if answers:
                answer = max(set(a.get("answer", "") for a in answers),
                             key=lambda x: sum(1 for a in answers if a.get("answer") == x))
        if not question or not answer:
            continue

        examples.append(make_vision_example(question, answer, img_path))
        if (i + 1) % 2000 == 0:
            print(f"  [{i + 1}/{len(ds)}] processed...")

    return examples


def convert_textcaps(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert TextCaps (OCR-focused captioning)."""
    from datasets import load_dataset

    print("[*] Loading TextCaps...")
    # HuggingFaceM4/TextCaps uses a legacy loading script (TextCaps.py) that
    # newer datasets versions reject. Use lmms-lab/TextCaps which is Parquet-based.
    ds = load_dataset("lmms-lab/TextCaps", split="train")

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "textcaps"
    prompts = [
        "What text can you see in this image?",
        "Describe this image, including any visible text.",
        "Read and describe the text in this image.",
        "What does the image show? Include any text you can read.",
        "Provide a detailed caption for this image, mentioning any text present.",
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
        if not caption:
            continue

        # Vary the prompt for diversity
        prompt = prompts[i % len(prompts)]
        examples.append(make_vision_example(prompt, caption, img_path))

    return examples


def convert_paragraph_captioning(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert paragraph captioning dataset (long-form descriptions)."""
    from datasets import load_dataset

    print("[*] Loading image paragraph captioning...")
    try:
        ds = load_dataset("merve/image-paragraph-captioning", split="train")
    except Exception:
        try:
            print("  [*] Trying fallback: google/docci...")
            ds = load_dataset("google/docci", split="train")
        except Exception as e:
            print(f"  [WARN] Could not load paragraph captioning: {e}")
            return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    img_dir = image_dir / "paragraph_captions"
    prompts = [
        "Describe this image in detail.",
        "Write a detailed description of what you see.",
        "Provide a comprehensive description of this image.",
        "What is happening in this image? Describe thoroughly.",
        "Summarize the content of this image in a paragraph.",
    ]

    for i, row in enumerate(ds):
        if row.get("image") is None:
            continue
        img_path = save_image(row["image"], img_dir / f"{i:06d}.jpg")
        if not img_path:
            continue

        caption = row.get("caption", row.get("text", ""))
        if not caption or len(caption) < 20:
            continue

        prompt = prompts[i % len(prompts)]
        examples.append(make_vision_example(prompt, caption, img_path))

    return examples


def convert_docvqa(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert DocVQA (document understanding)."""
    from datasets import load_dataset

    print("[*] Loading DocVQA...")
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
        # DocVQA answers can be a list
        answers = row.get("answers", [])
        if isinstance(answers, list) and answers:
            answer = answers[0]
        elif isinstance(answers, str):
            answer = answers
        else:
            answer = row.get("answer", "")

        if not question or not answer:
            continue

        examples.append(make_vision_example(question, answer, img_path))
        if (i + 1) % 2000 == 0:
            print(f"  [{i + 1}/{len(ds)}] processed...")

    return examples


def convert_chartqa(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert ChartQA (chart/graph understanding)."""
    from datasets import load_dataset

    print("[*] Loading ChartQA...")
    try:
        ds = load_dataset("HuggingFaceM4/ChartQA", split="train")
    except Exception:
        # Some versions split differently
        ds = load_dataset("ahmed-masry/ChartQA", split="train")

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

        question = row.get("question", row.get("query", ""))
        answer = row.get("answer", row.get("label", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not question or not answer:
            continue

        examples.append(make_vision_example(question, str(answer), img_path))

    return examples


def convert_infovqa(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert InfoVQA (infographic understanding)."""
    from datasets import load_dataset

    print("[*] Loading InfoVQA...")
    try:
        ds = load_dataset("lmms-lab/InfoVQA", split="train")
    except Exception:
        try:
            ds = load_dataset("lmms-lab/InfoVQA", split="validation")
        except Exception:
            try:
                ds = load_dataset("docile-benchmark/InfoVQA", split="train")
            except Exception as e:
                print(f"  [WARN] Could not load InfoVQA: {e}")
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
        answers = row.get("answers", row.get("answer", ""))
        if isinstance(answers, list) and answers:
            answer = answers[0]
        elif isinstance(answers, str):
            answer = answers
        else:
            continue

        if not question or not answer:
            continue

        examples.append(make_vision_example(question, str(answer), img_path))

    return examples


def convert_ai2d(max_samples: int, output_dir: Path, image_dir: Path):
    """Convert AI2D (science diagrams)."""
    from datasets import load_dataset

    print("[*] Loading AI2D...")
    try:
        ds = load_dataset("lmms-lab/ai2d", split="train")
    except Exception:
        try:
            ds = load_dataset("lmms-lab/ai2d", split="test")
        except Exception as e:
            print(f"  [WARN] Could not load AI2D: {e}")
            return []

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
        if isinstance(answer, int):
            # Multiple choice — get the text of the chosen option
            options = row.get("options", [])
            if options and answer < len(options):
                answer = options[answer]
            else:
                answer = str(answer)
        if not question or not answer:
            continue

        examples.append(make_vision_example(question, str(answer), img_path))

    return examples


# ─────────────────────────────────────────────────────────────────
# Synthetic Data: Document Summarization
# ─────────────────────────────────────────────────────────────────

def generate_doc_summarization_examples(max_samples: int = 2000):
    """
    Generate synthetic document-like training examples for
    text summarization capability.

    These are pure-text (no image) examples that teach the model
    to summarize content — complementing the vision + doc datasets.
    """
    from datasets import load_dataset

    print("[*] Loading document summarization data (CNN/DailyMail)...")
    try:
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train",
                          trust_remote_code=False)
    except Exception:
        print("  [WARN] Could not load CNN/DailyMail — skipping text summarization")
        return []

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    examples = []
    prompts = [
        "Summarize this document:",
        "Provide a brief summary of the following text:",
        "What are the key points in this document?",
        "Give a concise summary:",
        "Summarize the following:",
    ]

    for i, row in enumerate(ds):
        article = row.get("article", "")
        highlights = row.get("highlights", "")
        if not article or not highlights:
            continue

        # Truncate long articles (SmolVLM2 has limited context)
        if len(article) > 3000:
            article = article[:3000] + "..."

        prompt = prompts[i % len(prompts)]
        examples.append(make_text_example(
            f"{prompt}\n\n{article}",
            highlights,
        ))

    return examples


# ─────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────

CONVERTERS = {
    "vqav2":                      ("image",    convert_vqav2),
    "textcaps":                   ("image",    convert_textcaps),
    "image_paragraph_captioning": ("image",    convert_paragraph_captioning),
    "docvqa":                     ("document", convert_docvqa),
    "chartqa":                    ("document", convert_chartqa),
    "infovqa":                    ("document", convert_infovqa),
    "ai2d":                       ("document", convert_ai2d),
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare vision training data for SmolVLM2 fine-tuning")
    parser.add_argument("--task", choices=["image", "document", "all"], default="all",
                        help="Dataset category to download")
    parser.add_argument("--datasets", nargs="+", choices=list(CONVERTERS.keys()),
                        help="Specific datasets to download")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max samples per dataset (default: 10K)")
    parser.add_argument("--output-dir", default="../data/vision",
                        help="Output directory for processed data")
    parser.add_argument("--image-dir", default="../data/vision/images",
                        help="Directory to save extracted images")
    parser.add_argument("--include-summarization", action="store_true",
                        help="Include text-only document summarization data")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction held out for validation")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Decide which datasets to process
    if args.datasets:
        selected = {k: v for k, v in CONVERTERS.items() if k in args.datasets}
    elif args.task == "all":
        selected = CONVERTERS
    else:
        selected = {k: v for k, v in CONVERTERS.items() if v[0] == args.task}

    all_vision_examples = []
    all_text_examples = []

    for ds_name, (task, converter_fn) in selected.items():
        print(f"\n{'─' * 60}")
        print(f"  Processing: {ds_name} ({task})")
        print(f"{'─' * 60}")
        try:
            examples = converter_fn(args.max_samples, output_dir, image_dir)
            print(f"  → {len(examples)} examples")

            # Write per-dataset JSONL
            ds_file = output_dir / f"{ds_name}.jsonl"
            with open(ds_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            all_vision_examples.extend(examples)
        except Exception as e:
            print(f"  [ERROR] Failed to process {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Optional text summarization
    if args.include_summarization:
        print(f"\n{'─' * 60}")
        print(f"  Processing: document summarization (text-only)")
        print(f"{'─' * 60}")
        sum_examples = generate_doc_summarization_examples(
            max_samples=min(args.max_samples, 5000))
        print(f"  → {len(sum_examples)} examples")

        sum_file = output_dir / "summarization.jsonl"
        with open(sum_file, "w") as f:
            for ex in sum_examples:
                f.write(json.dumps(ex) + "\n")
        all_text_examples.extend(sum_examples)

    # ── Combined output ──
    import random
    random.seed(42)

    # Keep text-only examples in a separate file — mlx-vlm requires
    # a consistent schema (all rows must have "images" + array content).
    all_examples = list(all_vision_examples)  # vision-only for the main file
    random.shuffle(all_examples)

    # Train / val split
    val_size = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]

    # Write combined files
    train_file = output_dir / "smolvlm2_vision_train.jsonl"
    val_file = output_dir / "smolvlm2_vision_val.jsonl"

    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(val_file, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    # Write text-only examples separately (not compatible with mlx-vlm vision training)
    if all_text_examples:
        random.shuffle(all_text_examples)
        text_val_size = max(1, int(len(all_text_examples) * args.val_ratio))
        text_train = all_text_examples[text_val_size:]
        text_val = all_text_examples[:text_val_size]

        text_train_file = output_dir / "smolvlm2_text_train.jsonl"
        text_val_file = output_dir / "smolvlm2_text_val.jsonl"
        with open(text_train_file, "w") as f:
            for ex in text_train:
                f.write(json.dumps(ex) + "\n")
        with open(text_val_file, "w") as f:
            for ex in text_val:
                f.write(json.dumps(ex) + "\n")
        print(f"  Text-only:  {text_train_file} ({len(text_train):,} train, {len(text_val):,} val)")

    # Summary stats
    vision_count = len(all_vision_examples)
    text_count = len(all_text_examples)
    print(f"\n{'═' * 60}")
    print(f"  DATASET PREPARATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Vision examples:  {vision_count:,}")
    print(f"  Text examples:    {text_count:,} (separate file)")
    print(f"  Vision train:     {len(train_examples):,}")
    print(f"  Vision val:       {len(val_examples):,}")
    print(f"")
    print(f"  Output:  {train_file}")
    print(f"           {val_file}")
    print(f"  Images:  {image_dir}")
    print(f"{'═' * 60}")
    print(f"\n  Next: python 02b_train_smolvlm2_vision.py --backend auto")


if __name__ == "__main__":
    main()
