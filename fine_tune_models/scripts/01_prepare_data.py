#!/usr/bin/env python3
"""
Ocula Data Preparation Pipeline

Converts raw datasets into training-ready formats for each model.
Supports both custom data and public HuggingFace datasets.

Usage:
    # Download recommended public datasets
    python 01_prepare_data.py --download-datasets

    # Prepare your own data
    python 01_prepare_data.py --input ../data/raw/ --output ../data/processed/

    # Prepare for specific model only
    python 01_prepare_data.py --download-datasets --model ocula_plus
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    # Always resolve relative paths from fine_tune_models root, not shell cwd.
    return (PROJECT_DIR / p).resolve()

# ─────────────────────────────────────────────────────────────────
# RECOMMENDED DATASETS
# ─────────────────────────────────────────────────────────────────
# These are high-quality, permissively-licensed datasets suitable
# for fine-tuning Ocula's on-device models.

RECOMMENDED_DATASETS = {
    # ── Chat / RAG / Assistant ──
    "chat": [
        {
            "name": "OpenAssistant/oasst2",
            "description": "35K human-written multi-turn conversations. Best for general assistant behavior.",
            "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
            "size": "~50 MB",
            "license": "Apache 2.0",
            "use_for": ["ocula_lite"],
        },
        {
            "name": "HuggingFaceH4/ultrachat_200k",
            "description": "200K synthetic multi-turn conversations. Great for instruction following.",
            "url": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
            "size": "~500 MB",
            "license": "MIT",
            "use_for": ["ocula_lite"],
        },
        {
            "name": "teknium/OpenHermes-2.5",
            "description": "1M+ instruction-response pairs. Diverse, high quality.",
            "url": "https://huggingface.co/datasets/teknium/OpenHermes-2.5",
            "size": "~1.5 GB",
            "license": "Apache 2.0",
            "use_for": ["ocula_lite"],
        },
        {
            "name": "argilla/distilabel-capybara-dpo-7k-binarized",
            "description": "7K DPO preference pairs. Good for alignment.",
            "url": "https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized",
            "size": "~15 MB",
            "license": "Apache 2.0",
            "use_for": ["ocula_lite"],
        },
    ],
    # ── Vision / Image Understanding ──
    "vision": [
        {
            "name": "HuggingFaceM4/the_cauldron",
            "description": "50+ vision-language tasks. Image captioning, VQA, OCR, charts.",
            "url": "https://huggingface.co/datasets/HuggingFaceM4/the_cauldron",
            "size": "~20 GB (subset recommended)",
            "license": "Mixed (mostly permissive)",
            "use_for": ["ocula_plus", "ocula_pro"],
            "split": "train",
        },
        {
            "name": "lmms-lab/LLaVA-OneVision-Data",
            "description": "LLaVA training data. Image+text instruction following.",
            "url": "https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data",
            "size": "~10 GB",
            "license": "Apache 2.0",
            "use_for": ["ocula_plus", "ocula_pro"],
            "split": "train",
            "config": "CLEVR-Math(MathV360K)",
        },
        {
            "name": "liuhaotian/LLaVA-Instruct-150K",
            "description": "150K image-instruction pairs from GPT-4V. Classic VLM training set.",
            "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K",
            "size": "~200 MB (text only, images from COCO)",
            "license": "CC BY 4.0",
            "use_for": ["ocula_plus", "ocula_pro"],
            "split": "train",
        },
        {
            "name": "merve/vqav2-small",
            "description": "Visual QA dataset. Good for image question answering.",
            "url": "https://huggingface.co/datasets/merve/vqav2-small",
            "size": "~500 MB",
            "license": "CC BY 4.0",
            "use_for": ["ocula_plus", "ocula_pro"],
            "split": "validation",
        },
    ],
    # ── RAG / Retrieval / Search ──
    "retrieval": [
        {
            "name": "sentence-transformers/all-nli",
            "description": "275K NLI sentence pairs. Foundation for semantic search tuning.",
            "url": "https://huggingface.co/datasets/sentence-transformers/all-nli",
            "size": "~50 MB",
            "license": "Apache 2.0",
            "use_for": ["qwen3embed"],
            "split": "train",
            "config": "pair",
        },
        {
            "name": "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
            "description": "500K query-passage pairs from MS MARCO. Best for RAG retrieval.",
            "url": "https://huggingface.co/datasets/sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
            "size": "~200 MB",
            "license": "MIT",
            "use_for": ["qwen3embed"],
            "split": "train",
        },
        {
            "name": "sentence-transformers/natural-questions",
            "description": "100K real Google search queries with Wikipedia passages.",
            "url": "https://huggingface.co/datasets/sentence-transformers/natural-questions",
            "size": "~300 MB",
            "license": "Apache 2.0",
            "use_for": ["qwen3embed"],
            "split": "train",
        },
        {
            "name": "BeIR/hotpotqa",
            "description": "Multi-hop QA. Tests complex retrieval reasoning.",
            "url": "https://huggingface.co/datasets/BeIR/hotpotqa",
            "size": "~100 MB",
            "license": "CC BY-SA 4.0",
            "use_for": ["qwen3embed"],
        },
    ],
    # ── Domain-Specific (Ocula use cases) ──
    "domain": [
        {
            "name": "Open-Orca/SlimOrca",
            "description": "518K GPT-4 verified instructions. High quality, diverse tasks.",
            "url": "https://huggingface.co/datasets/Open-Orca/SlimOrca",
            "size": "~400 MB",
            "license": "MIT",
            "use_for": ["ocula_lite"],
        },
        {
            "name": "TIGER-Lab/MathInstruct",
            "description": "262K math reasoning problems. Good for chain-of-thought training.",
            "url": "https://huggingface.co/datasets/TIGER-Lab/MathInstruct",
            "size": "~100 MB",
            "license": "MIT",
            "use_for": ["ocula_lite"],
        },
        {
            "name": "ccdv/pubmed-summarization",
            "description": "Medical/scientific summarization. Good for Ocula's document analysis.",
            "url": "https://huggingface.co/datasets/ccdv/pubmed-summarization",
            "size": "~1 GB",
            "license": "Apache 2.0",
            "use_for": ["ocula_lite"],
        },
    ],
}


def print_datasets():
    """Print all recommended datasets in a nice table."""
    print("\n" + "=" * 80)
    print("RECOMMENDED DATASETS FOR OCULA FINE-TUNING")
    print("=" * 80)

    for category, datasets in RECOMMENDED_DATASETS.items():
        print(f"\n{'─' * 80}")
        print(f"  {category.upper()}")
        print(f"{'─' * 80}")
        for ds in datasets:
            models = ", ".join(ds["use_for"])
            print(f"\n  {ds['name']}")
            print(f"    {ds['description']}")
            print(f"    Size: {ds['size']} | License: {ds['license']}")
            print(f"    Models: {models}")
            print(f"    URL: {ds['url']}")

    print(f"\n{'=' * 80}")
    print("To download: python 01_prepare_data.py --download-datasets")
    print("To download specific: python 01_prepare_data.py --download-datasets --model qwen3embed")
    print("=" * 80 + "\n")


def download_datasets(model_filter=None, output_dir=None):
    """Download recommended datasets from HuggingFace."""
    from datasets import load_dataset

    output_dir = resolve_project_path(output_dir or "data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, datasets in RECOMMENDED_DATASETS.items():
        for ds_info in datasets:
            # Filter by model if specified
            if model_filter and model_filter not in ds_info["use_for"]:
                continue

            name = ds_info["name"]
            target_dir = output_dir / name.replace("/", "__")

            if target_dir.exists():
                print(f"  [skip] {name} — already downloaded")
                continue

            print(f"\n  [download] {name} ({ds_info['size']})")
            print(f"    → {target_dir}")

            try:
                split = ds_info.get("split", "train")
                config = ds_info.get("config")

                if config:
                    ds = load_dataset(name, config, split=split)
                else:
                    ds = load_dataset(name, split=split)
                ds.save_to_disk(str(target_dir))
                print(f"    [OK] {len(ds)} examples saved")
            except Exception as e:
                # Fallback: try first available split if default split is missing.
                try:
                    ds_all = load_dataset(name, ds_info.get("config")) if ds_info.get("config") else load_dataset(name)
                    first_split = next(iter(ds_all.keys()))
                    ds = ds_all[first_split]
                    ds.save_to_disk(str(target_dir))
                    print(f"    [OK] {len(ds)} examples saved (fallback split={first_split})")
                    continue
                except Exception:
                    pass
                print(f"    [WARN] Failed: {e}")
                print(f"    Try manually: huggingface-cli download {name}")


def convert_oasst2_to_chat(input_dir, output_file):
    """Convert OpenAssistant OASST2 dataset to ChatML format."""
    from datasets import load_from_disk

    ds = load_from_disk(str(input_dir))
    conversations = []

    # OASST2 has a tree structure — extract linear conversations
    # Group messages by conversation tree
    trees = {}
    for row in ds:
        tree_id = row.get("message_tree_id", row.get("parent_id", "unknown"))
        if tree_id not in trees:
            trees[tree_id] = []
        trees[tree_id].append(row)

    for tree_id, messages in trees.items():
        # Sort by created_date or index
        messages.sort(key=lambda x: x.get("created_date", ""))

        chat = {"messages": []}
        for msg in messages:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            text = msg.get("text", "")
            if text.strip():
                chat["messages"].append({"role": role, "content": text})

        if len(chat["messages"]) >= 2:
            conversations.append(chat)

    # Write JSONL
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"  [OK] {len(conversations)} conversations → {output_file}")
    return conversations


def convert_retrieval_to_triplets(input_dir, output_file):
    """Convert retrieval dataset to query/positive/negative triplets for MiniLM."""
    from datasets import load_from_disk

    ds = load_from_disk(str(input_dir))
    triplets = []

    for row in ds:
        triplet = {}
        # Handle different column naming conventions
        if "query" in row and "positive" in row:
            triplet = {
                "query": row["query"],
                "positive": row["positive"],
                "negative": row.get("negative", ""),
            }
        elif "anchor" in row and "positive" in row:
            triplet = {
                "query": row["anchor"],
                "positive": row["positive"],
                "negative": row.get("negative", ""),
            }
        elif "sentence1" in row and "sentence2" in row:
            # NLI-style: entailment = positive, contradiction = negative
            label = row.get("label", 0)
            if label == 0:  # entailment
                triplet = {
                    "query": row["sentence1"],
                    "positive": row["sentence2"],
                    "negative": "",
                }
        if triplet:
            triplets.append(triplet)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"  [OK] {len(triplets)} triplets → {output_file}")
    return triplets


def convert_vision_to_chatml(input_dir, output_file):
    """Convert vision dataset to ChatML with image paths."""
    from datasets import load_from_disk

    ds = load_from_disk(str(input_dir))
    examples = []

    img_dir = Path(output_file).parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        # Save image if present
        img_path = None
        if "image" in row and row["image"] is not None:
            img_path = str(img_dir / f"{i:06d}.jpg")
            try:
                row["image"].save(img_path)
            except Exception:
                img_path = None

        # Extract question/answer
        question = row.get("question", row.get("query", row.get("prompt", "")))
        answer = row.get("answer", row.get("response", row.get("caption", "")))

        if not question or not answer:
            continue

        example = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        if img_path:
            example["images"] = [img_path]

        examples.append(example)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  [OK] {len(examples)} vision examples → {output_file}")


def convert_custom_data(input_dir, output_dir):
    """
    Convert custom data in data/raw/ to training format.

    Expected input formats:
    - *.jsonl with {"prompt": "...", "response": "..."} or ChatML format
    - *.csv with prompt,response columns
    - *.txt files (will be chunked for embedding training)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process JSONL files
    for jsonl_file in input_dir.glob("*.jsonl"):
        print(f"  Processing {jsonl_file.name}...")
        chat_examples = []
        retrieval_examples = []

        with open(jsonl_file) as f:
            for line in f:
                row = json.loads(line.strip())

                # Already in ChatML format
                if "messages" in row:
                    chat_examples.append(row)
                # Simple prompt/response
                elif "prompt" in row and "response" in row:
                    chat_examples.append({
                        "messages": [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]},
                        ]
                    })
                # Retrieval triplets
                elif "query" in row and "positive" in row:
                    retrieval_examples.append(row)

        if chat_examples:
            out_file = output_dir / f"chat_{jsonl_file.stem}.jsonl"
            with open(out_file, "w") as f:
                for ex in chat_examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"    → {len(chat_examples)} chat examples → {out_file.name}")

        if retrieval_examples:
            out_file = output_dir / f"retrieval_{jsonl_file.stem}.jsonl"
            with open(out_file, "w") as f:
                for ex in retrieval_examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"    → {len(retrieval_examples)} retrieval examples → {out_file.name}")

    # Process text files (chunk for embedding training)
    for txt_file in input_dir.glob("*.txt"):
        print(f"  Chunking {txt_file.name} for embedding training...")
        text = txt_file.read_text()
        # Simple sentence-aware chunking (same logic as Ocula's RAG)
        import re
        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > 800 and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap: keep last 2 sentences
                current_chunk = current_chunk[-2:]
                current_len = sum(len(s) for s in current_chunk)
            current_chunk.append(sent)
            current_len += len(sent)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Create self-supervised pairs (adjacent chunks are positive pairs)
        pairs = []
        for i in range(len(chunks) - 1):
            pairs.append({
                "query": chunks[i][:200],  # Use first part as query
                "positive": chunks[i + 1],
                "negative": chunks[max(0, i - 3)] if i > 3 else "",
            })

        out_file = output_dir / f"retrieval_{txt_file.stem}.jsonl"
        with open(out_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"    → {len(pairs)} retrieval pairs → {out_file.name}")


def _extract_text_pair(row):
    """Try to extract a user prompt + assistant response from many schema variants."""
    prompt_keys = ["prompt", "instruction", "question", "query", "input", "user", "context"]
    response_keys = ["response", "answer", "output", "assistant", "completion", "target", "chosen"]

    prompt = ""
    response = ""

    for k in prompt_keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            prompt = row[k].strip()
            break
    for k in response_keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            response = row[k].strip()
            break

    # Handle chat-style rows
    if not prompt and "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        user_msgs = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        asst_msgs = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        if user_msgs:
            prompt = str(user_msgs[-1]).strip()
        if asst_msgs:
            response = str(asst_msgs[-1]).strip()

    if prompt and response:
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}
    return None


def convert_downloaded_hf_datasets(raw_dir, output_dir):
    """Convert datasets downloaded via save_to_disk into Ocula training formats."""
    from datasets import load_from_disk

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_dir in sorted(raw_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        if "__" not in ds_dir.name:
            continue

        ds_name = ds_dir.name.replace("__", "/")
        print(f"  Converting downloaded dataset: {ds_name}")

        try:
            ds = load_from_disk(str(ds_dir))
        except Exception as e:
            print(f"    [WARN] Could not load {ds_dir.name}: {e}")
            continue

        # Retrieval datasets
        if "sentence-transformers/" in ds_name or "BeIR/" in ds_name:
            out_file = output_dir / f"retrieval_{ds_dir.name}.jsonl"
            convert_retrieval_to_triplets(ds_dir, out_file)
            continue

        # Vision datasets
        if any(tag in ds_name for tag in ["LLaVA", "vqav2", "the_cauldron"]):
            out_file = output_dir / f"vision_{ds_dir.name}.jsonl"
            convert_vision_to_chatml(ds_dir, out_file)
            continue

        # OASST2 has specific tree structure
        if ds_name == "OpenAssistant/oasst2":
            out_file = output_dir / "chat_oasst2.jsonl"
            convert_oasst2_to_chat(ds_dir, out_file)
            continue

        # Generic chat/dataset conversion
        chat_rows = []
        for row in ds:
            chat_ex = _extract_text_pair(row)
            if chat_ex:
                chat_rows.append(chat_ex)

        if chat_rows:
            out_file = output_dir / f"chat_{ds_dir.name}.jsonl"
            with open(out_file, "w") as f:
                for ex in chat_rows:
                    f.write(json.dumps(ex) + "\n")
            print(f"    [OK] {len(chat_rows)} chat examples → {out_file.name}")
        else:
            print(f"    [WARN] No usable examples found in {ds_name}")


def create_train_val_split(processed_dir, val_ratio=0.05):
    """Split processed data into train/val sets."""
    processed_dir = Path(processed_dir)

    for jsonl_file in processed_dir.glob("*.jsonl"):
        if jsonl_file.stem.endswith("_train") or jsonl_file.stem.endswith("_val"):
            continue

        with open(jsonl_file) as f:
            lines = f.readlines()

        import random
        random.shuffle(lines)

        val_size = max(1, int(len(lines) * val_ratio))
        val_lines = lines[:val_size]
        train_lines = lines[val_size:]

        train_file = processed_dir / f"{jsonl_file.stem}_train.jsonl"
        val_file = processed_dir / f"{jsonl_file.stem}_val.jsonl"

        with open(train_file, "w") as f:
            f.writelines(train_lines)
        with open(val_file, "w") as f:
            f.writelines(val_lines)

        print(f"  {jsonl_file.stem}: {len(train_lines)} train / {len(val_lines)} val")


def main():
    parser = argparse.ArgumentParser(description="Ocula Data Preparation Pipeline")
    parser.add_argument("--input", default="data/raw", help="Raw data directory (relative to fine_tune_models/)")
    parser.add_argument("--output", default="data/processed", help="Output directory (relative to fine_tune_models/)")
    parser.add_argument("--download-datasets", action="store_true", help="Download recommended HF datasets")
    parser.add_argument("--list-datasets", action="store_true", help="List recommended datasets")
    parser.add_argument("--model", choices=["ocula_lite", "ocula_plus", "ocula_pro", "qwen3embed", "all"],
                       default="all", help="Prepare data for specific model")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    args = parser.parse_args()

    if args.list_datasets:
        print_datasets()
        return

    if args.download_datasets:
        model_filter = None if args.model == "all" else args.model
        download_datasets(model_filter=model_filter, output_dir=args.input)
        print("\n[*] Downloaded. Now converting to training format...")

    # Resolve canonical paths (fine_tune_models/data/*)
    raw_dir = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output)
    legacy_data_root = PROJECT_DIR.parent / "data"
    legacy_raw_dir = legacy_data_root / "raw"
    legacy_processed_dir = legacy_data_root / "processed"

    # Auto-reuse legacy raw data location from older script versions.
    if args.input == "data/raw" and not raw_dir.exists() and legacy_raw_dir.exists():
        print(f"[*] Using legacy raw data path: {legacy_raw_dir}")
        raw_dir = legacy_raw_dir

    # Auto-migrate legacy processed files into canonical output location.
    if args.output == "data/processed" and legacy_processed_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        if not any(output_dir.glob("*.jsonl")) and any(legacy_processed_dir.glob("*.jsonl")):
            print(f"[*] Importing legacy processed data from: {legacy_processed_dir}")
            for src in legacy_processed_dir.glob("*.jsonl"):
                dst = output_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

    if raw_dir.exists():
        print("\n[*] Converting raw data to training format...")
        convert_downloaded_hf_datasets(raw_dir, output_dir)
        convert_custom_data(raw_dir, output_dir)

    # Create train/val splits
    if output_dir.exists() and any(output_dir.glob("*.jsonl")):
        print("\n[*] Creating train/val splits...")
        create_train_val_split(output_dir, args.val_ratio)

    print("\n[OK] Data preparation complete!")
    print(f"     Processed data in: {output_dir}")
    print(f"     Next: run training scripts (02-05)")


if __name__ == "__main__":
    main()
