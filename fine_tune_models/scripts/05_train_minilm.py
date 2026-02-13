#!/usr/bin/env python3
"""
MiniLM-L6-v2 Embedding Fine-tuning Script
22M params — full fine-tune (tiny model, no LoRA needed)

Improves RAG retrieval quality by training on domain-specific data.

Usage:
    python 05_train_minilm.py
    python 05_train_minilm.py --config ../configs/minilm.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path="../configs/minilm.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config):
    """Full fine-tune MiniLM-L6-v2 for semantic search."""
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
        evaluation,
    )
    from torch.utils.data import DataLoader

    model_path = config["model"]["local_path"]
    train_cfg = config["training"]

    print(f"\n[*] Loading MiniLM-L6-v2 from: {model_path}")
    model = SentenceTransformer(model_path)
    print(f"  [*] Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"  [*] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load retrieval training data
    print("[*] Loading training data...")
    data_files = list(Path(config["data"]["train"]).parent.glob(
        Path(config["data"]["train"]).name))

    if not data_files:
        print("[!] No training data. Run: python 01_prepare_data.py --download-datasets --model minilm")
        sys.exit(1)

    train_examples = []
    for data_file in data_files:
        with open(data_file) as f:
            for line in f:
                row = json.loads(line.strip())
                query = row.get("query", "")
                positive = row.get("positive", "")
                negative = row.get("negative", "")

                if query and positive:
                    if negative:
                        train_examples.append(InputExample(
                            texts=[query, positive, negative]
                        ))
                    else:
                        train_examples.append(InputExample(
                            texts=[query, positive]
                        ))

    max_samples = config["data"].get("max_samples")
    if max_samples and len(train_examples) > max_samples:
        import random
        random.shuffle(train_examples)
        train_examples = train_examples[:max_samples]

    print(f"  [*] {len(train_examples)} training pairs")

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=train_cfg["batch_size"],
    )

    # Loss function — MultipleNegativesRankingLoss is the gold standard
    # for retrieval fine-tuning. Uses in-batch negatives + hard negatives.
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Load validation data for evaluation during training
    val_files = list(Path(config["data"]["val"]).parent.glob(
        Path(config["data"]["val"]).name))
    evaluator = None

    if val_files:
        print("[*] Loading validation data for evaluation...")
        queries = {}
        corpus = {}
        relevant_docs = {}
        doc_id = 0

        for val_file in val_files:
            with open(val_file) as f:
                for i, line in enumerate(f):
                    row = json.loads(line.strip())
                    query = row.get("query", "")
                    positive = row.get("positive", "")

                    if query and positive:
                        q_id = f"q_{i}"
                        d_id = f"d_{doc_id}"
                        queries[q_id] = query
                        corpus[d_id] = positive
                        relevant_docs[q_id] = {d_id}
                        doc_id += 1

                    if len(queries) >= 1000:  # Cap eval set
                        break

        if queries:
            evaluator = evaluation.InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="retrieval-eval",
            )
            print(f"  [*] {len(queries)} eval queries, {len(corpus)} eval docs")

    # Training
    output_dir = "../models/base/minilm-l6-v2-finetuned"
    warmup_steps = train_cfg.get("warmup_steps", 500)
    epochs = train_cfg["epochs"]

    print(f"\n[*] Starting training → {output_dir}")
    print(f"    Epochs: {epochs}, Batch: {train_cfg['batch_size']}, "
          f"LR: {train_cfg['learning_rate']}")
    print(f"    Loss: MultipleNegativesRankingLoss")

    model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=1000,
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True,  # Mixed precision
    )

    print(f"\n[OK] MiniLM fine-tuning complete!")
    print(f"     Model saved to: {output_dir}")
    print(f"     Next: python 07_quantize_gguf.py --model minilm")


def main():
    parser = argparse.ArgumentParser(description="MiniLM-L6-v2 Embedding Fine-tuning")
    parser.add_argument("--config", default="../configs/minilm.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
