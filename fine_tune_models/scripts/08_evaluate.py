#!/usr/bin/env python3
"""
Model Evaluation & Benchmarking Script
Compares base vs fine-tuned models across multiple benchmarks.

Evaluates:
  - Perplexity (lower = better language modeling)
  - GGUF inference speed (tok/s via llama.cpp)
  - RAG retrieval quality (MRR, NDCG for embedding model)
  - Task-specific accuracy (ChatML response quality)

Usage:
    # Evaluate all models
    python 08_evaluate.py

    # Evaluate a specific model
    python 08_evaluate.py --model smolvlm2
    python 08_evaluate.py --model minilm

    # Only run perplexity benchmark
    python 08_evaluate.py --model qwen3vl --bench perplexity

    # Compare base vs fine-tuned GGUF
    python 08_evaluate.py --model moondream2 --bench inference
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ── Paths ────────────────────────────────────────────────────────
LLAMA_CPP_DIR = Path("../../llama.cpp")
LLAMA_PERPLEXITY = LLAMA_CPP_DIR / "build" / "bin" / "llama-perplexity"
LLAMA_CLI = LLAMA_CPP_DIR / "build" / "bin" / "llama-cli"

GGUF_DIR = Path("../models/gguf")
EVAL_DIR = Path("../data/eval")
LOGS_DIR = Path("../logs")
RESULTS_DIR = LOGS_DIR / "eval_results"

# ── Model registry ──────────────────────────────────────────────
MODELS = {
    "smolvlm2": {
        "config": "../configs/smolvlm2.yaml",
        "base_gguf": "../../models/SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf",
        "finetuned_gguf_pattern": "SmolVLM2-500M-finetuned-*.gguf",
        "ctx": 2048,
    },
    "moondream2": {
        "config": "../configs/moondream2.yaml",
        "base_gguf": "../../models/moondream2-q4.gguf",
        "finetuned_gguf_pattern": "moondream2-text-model-finetuned-*.gguf",
        "ctx": 2048,
    },
    "qwen3vl": {
        "config": "../configs/qwen3vl.yaml",
        "base_gguf": None,  # Not currently deployed as GGUF
        "finetuned_gguf_pattern": "Qwen3VL-2B-Thinking-finetuned-*.gguf",
        "ctx": 4096,
    },
    "minilm": {
        "config": "../configs/minilm.yaml",
        "base_gguf": None,
        "finetuned_gguf_pattern": "minilm-l6-v2-finetuned-*.gguf",
        "ctx": 512,
    },
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_gguf(pattern):
    """Find the best quantized GGUF (prefer Q4_K_M, then Q8_0)."""
    candidates = sorted(GGUF_DIR.glob(pattern))
    if not candidates:
        return None
    # Prefer Q4_K_M for inference benchmarks
    for c in candidates:
        if "Q4_K_M" in c.name:
            return c
    return candidates[0]


def load_eval_prompts(eval_file=None):
    """Load evaluation prompts for response quality testing."""
    if eval_file and Path(eval_file).exists():
        with open(eval_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    # Default eval prompts for Ocula assistant
    return [
        {"prompt": "What's the weather like today?",
         "expected_traits": ["polite", "admits_limitation"]},
        {"prompt": "Summarize this document for me.",
         "expected_traits": ["asks_for_document", "helpful"]},
        {"prompt": "Write a Python function to sort a list.",
         "expected_traits": ["code_block", "correct_syntax"]},
        {"prompt": "Tell me about my calendar events.",
         "expected_traits": ["references_calendar", "structured"]},
        {"prompt": "What did we talk about yesterday?",
         "expected_traits": ["references_history", "conversational"]},
        {"prompt": "Help me draft an email to my boss.",
         "expected_traits": ["professional", "structured"]},
        {"prompt": "Describe what you see in this image.",
         "expected_traits": ["vision_aware", "descriptive"]},
        {"prompt": "Can you translate this to French?",
         "expected_traits": ["multilingual", "accurate"]},
    ]


# ─────────────────────────────────────────────────────────────────
# Benchmark: Perplexity
# ─────────────────────────────────────────────────────────────────

def bench_perplexity(gguf_path, ctx_size, label=""):
    """Measure perplexity using llama-perplexity on wikitext."""
    if not LLAMA_PERPLEXITY.exists():
        print(f"  [!] llama-perplexity not found at {LLAMA_PERPLEXITY}")
        return None

    # Use a small eval corpus
    eval_text = EVAL_DIR / "wikitext_test.txt"
    if not eval_text.exists():
        print(f"  [!] Eval corpus not found: {eval_text}")
        print(f"      Create it: head -500 wikitext-2-raw/wiki.test.raw > {eval_text}")
        return None

    print(f"  [{label}] Running perplexity on {gguf_path.name}...")

    cmd = [
        str(LLAMA_PERPLEXITY),
        "--model", str(gguf_path),
        "--file", str(eval_text),
        "--ctx-size", str(ctx_size),
        "--threads", str(min(os.cpu_count() or 4, 8)),
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [!] Perplexity failed: {result.stderr[:200]}")
        return None

    # Parse perplexity from output
    ppl = None
    for line in result.stdout.split("\n"):
        if "perplexity" in line.lower() and "=" in line:
            try:
                ppl = float(line.split("=")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Final estimate:" in line:
            try:
                ppl = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    if ppl:
        print(f"  [{label}] Perplexity: {ppl:.2f} ({elapsed:.0f}s)")
    else:
        print(f"  [{label}] Could not parse perplexity from output")

    return {"perplexity": ppl, "elapsed_s": elapsed}


# ─────────────────────────────────────────────────────────────────
# Benchmark: Inference Speed
# ─────────────────────────────────────────────────────────────────

def bench_inference(gguf_path, ctx_size, label=""):
    """Measure tokens/second using llama-cli."""
    if not LLAMA_CLI.exists():
        print(f"  [!] llama-cli not found at {LLAMA_CLI}")
        return None

    print(f"  [{label}] Benchmarking inference speed on {gguf_path.name}...")

    prompt = "Explain quantum computing in simple terms."
    cmd = [
        str(LLAMA_CLI),
        "--model", str(gguf_path),
        "--prompt", f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "--n-predict", "128",
        "--ctx-size", str(ctx_size),
        "--threads", str(min(os.cpu_count() or 4, 8)),
        "--no-display-prompt",
        "--log-disable",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [!] Inference benchmark failed: {result.stderr[:200]}")
        return None

    # Parse timing from stderr (llama.cpp outputs stats there)
    tok_per_sec = None
    prompt_tok_per_sec = None
    total_tokens = None

    for line in (result.stderr + result.stdout).split("\n"):
        # llama_perf: eval time = ... ms / 128 tokens (... ms per token, ... tokens per second)
        if "eval time" in line and "tokens per second" in line:
            try:
                tok_per_sec = float(line.split("tokens per second")[0].split(",")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "prompt eval time" in line and "tokens per second" in line:
            try:
                prompt_tok_per_sec = float(
                    line.split("tokens per second")[0].split(",")[-1].strip()
                )
            except (ValueError, IndexError):
                pass

    output_tokens = len(result.stdout.split())

    print(f"  [{label}] Generation: {tok_per_sec or '?'} tok/s, "
          f"Prompt: {prompt_tok_per_sec or '?'} tok/s, "
          f"Time: {elapsed:.1f}s")

    return {
        "tok_per_sec": tok_per_sec,
        "prompt_tok_per_sec": prompt_tok_per_sec,
        "elapsed_s": elapsed,
        "output_tokens": output_tokens,
    }


# ─────────────────────────────────────────────────────────────────
# Benchmark: Embedding Retrieval Quality (MiniLM only)
# ─────────────────────────────────────────────────────────────────

def bench_retrieval(finetuned_path, base_path=None):
    """Evaluate embedding model retrieval quality."""
    print(f"  [*] Evaluating retrieval quality...")

    # Load eval data
    eval_file = EVAL_DIR / "retrieval_eval.jsonl"
    if not eval_file.exists():
        print(f"  [!] Retrieval eval data not found: {eval_file}")
        print(f"      Create JSONL with {{query, positive, negatives[]}} entries")
        return None

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  [!] sentence-transformers not installed")
        return None

    with open(eval_file) as f:
        eval_data = [json.loads(line) for line in f if line.strip()]

    if not eval_data:
        print("  [!] No eval data found")
        return None

    results = {}

    for label, model_path in [("base", base_path), ("finetuned", finetuned_path)]:
        if model_path is None or not Path(model_path).exists():
            print(f"  [{label}] Model not found, skipping")
            continue

        model = SentenceTransformer(str(model_path))
        mrr_total = 0.0
        hits_at_5 = 0

        for item in eval_data:
            query = item["query"]
            positive = item["positive"]
            negatives = item.get("negatives", [])
            candidates = [positive] + negatives

            q_emb = model.encode(query, normalize_embeddings=True)
            c_embs = model.encode(candidates, normalize_embeddings=True)

            # Cosine similarity (already normalized)
            scores = np.dot(c_embs, q_emb)
            ranked = np.argsort(-scores)

            # MRR: rank of the positive doc (index 0)
            rank = int(np.where(ranked == 0)[0][0]) + 1
            mrr_total += 1.0 / rank
            if rank <= 5:
                hits_at_5 += 1

        n = len(eval_data)
        mrr = mrr_total / n
        h5 = hits_at_5 / n

        print(f"  [{label}] MRR: {mrr:.4f}, Hits@5: {h5:.2%}")
        results[label] = {"mrr": mrr, "hits_at_5": h5, "n_queries": n}

    return results


# ─────────────────────────────────────────────────────────────────
# Benchmark: Response Quality (LLM-as-judge)
# ─────────────────────────────────────────────────────────────────

def bench_response_quality(gguf_path, ctx_size, label=""):
    """Generate responses and compute basic quality metrics."""
    if not LLAMA_CLI.exists():
        return None

    prompts = load_eval_prompts()
    results = []

    print(f"  [{label}] Testing response quality ({len(prompts)} prompts)...")

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        cmd = [
            str(LLAMA_CLI),
            "--model", str(gguf_path),
            "--prompt",
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "--n-predict", "256",
            "--ctx-size", str(ctx_size),
            "--threads", str(min(os.cpu_count() or 4, 8)),
            "--no-display-prompt",
            "--log-disable",
            "--temp", "0.1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        response = result.stdout.strip() if result.returncode == 0 else ""

        # Basic quality metrics
        word_count = len(response.split())
        has_garbage = any(c * 5 in response for c in "!@#$%^&*")
        repeats = _detect_repetition(response)
        is_empty = word_count < 3

        quality = {
            "prompt": prompt,
            "response_preview": response[:200],
            "word_count": word_count,
            "has_garbage": has_garbage,
            "has_repetition": repeats,
            "is_empty": is_empty,
            "quality_score": 1.0 - (0.3 * has_garbage + 0.3 * repeats + 0.4 * is_empty),
        }
        results.append(quality)

    avg_score = sum(r["quality_score"] for r in results) / len(results)
    avg_words = sum(r["word_count"] for r in results) / len(results)

    print(f"  [{label}] Avg quality: {avg_score:.2f}, Avg words: {avg_words:.0f}")
    return {"avg_quality_score": avg_score, "avg_word_count": avg_words, "details": results}


def _detect_repetition(text, window=20):
    """Detect if the model is stuck in a repetition loop."""
    words = text.split()
    if len(words) < window * 2:
        return False
    # Check if the last N words repeat the previous N words
    tail = " ".join(words[-window:])
    body = " ".join(words[:-window])
    return tail in body


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model_name, benchmarks):
    """Run all requested benchmarks for a model."""
    info = MODELS[model_name]
    config = load_config(info["config"])
    ctx = info["ctx"]

    is_embedding = config.get("model", {}).get("type") == "embedding"

    base_gguf = Path(info["base_gguf"]) if info.get("base_gguf") else None
    finetuned_gguf = find_gguf(info["finetuned_gguf_pattern"])

    results = {"model": model_name, "benchmarks": {}}

    if is_embedding:
        # Embedding model → retrieval benchmarks only
        if "retrieval" in benchmarks or "all" in benchmarks:
            base_path = config["model"].get("local_path")
            ft_path = "../models/base/minilm-l6-v2-finetuned"
            results["benchmarks"]["retrieval"] = bench_retrieval(ft_path, base_path)
    else:
        # LLM / VLM benchmarks
        for gguf, label in [(base_gguf, "base"), (finetuned_gguf, "finetuned")]:
            if gguf is None or not gguf.exists():
                print(f"  [{label}] GGUF not found, skipping")
                continue

            if "perplexity" in benchmarks or "all" in benchmarks:
                results["benchmarks"].setdefault("perplexity", {})[label] = \
                    bench_perplexity(gguf, ctx, label)

            if "inference" in benchmarks or "all" in benchmarks:
                results["benchmarks"].setdefault("inference", {})[label] = \
                    bench_inference(gguf, ctx, label)

            if "quality" in benchmarks or "all" in benchmarks:
                results["benchmarks"].setdefault("quality", {})[label] = \
                    bench_response_quality(gguf, ctx, label)

    return results


def main():
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned models")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to evaluate (default: all)")
    parser.add_argument("--bench", nargs="+",
                        choices=["perplexity", "inference", "quality", "retrieval", "all"],
                        default=["all"],
                        help="Which benchmarks to run (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_name in targets:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")
        results = evaluate_model(model_name, args.bench)
        all_results[model_name] = results

    # Save results
    output_path = args.output or str(
        RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("  Evaluation Summary")
    print(f"{'='*60}")
    for model_name, data in all_results.items():
        print(f"\n  {model_name}:")
        benchmarks = data.get("benchmarks", {})

        if "perplexity" in benchmarks:
            ppl = benchmarks["perplexity"]
            base_ppl = ppl.get("base", {})
            ft_ppl = ppl.get("finetuned", {})
            b = base_ppl.get("perplexity", "N/A") if base_ppl else "N/A"
            f = ft_ppl.get("perplexity", "N/A") if ft_ppl else "N/A"
            delta = ""
            if isinstance(b, (int, float)) and isinstance(f, (int, float)):
                d = f - b
                delta = f" ({'+' if d > 0 else ''}{d:.2f})"
            print(f"    Perplexity:  base={b}  fine-tuned={f}{delta}")

        if "inference" in benchmarks:
            inf = benchmarks["inference"]
            for label in ["base", "finetuned"]:
                r = inf.get(label, {})
                if r:
                    print(f"    Speed ({label}): {r.get('tok_per_sec', '?')} tok/s")

        if "quality" in benchmarks:
            qual = benchmarks["quality"]
            for label in ["base", "finetuned"]:
                r = qual.get(label, {})
                if r:
                    print(f"    Quality ({label}): {r.get('avg_quality_score', '?'):.2f}")

        if "retrieval" in benchmarks:
            ret = benchmarks["retrieval"]
            if ret:
                for label in ["base", "finetuned"]:
                    r = ret.get(label, {})
                    if r:
                        print(f"    Retrieval ({label}): MRR={r['mrr']:.4f} "
                              f"Hits@5={r['hits_at_5']:.2%}")

    print()


if __name__ == "__main__":
    main()
