#!/usr/bin/env python3
"""
Model Comparison Script — Before vs After Fine-Tuning
=====================================================
Runs identical prompts through base and fine-tuned models, shows side-by-side
responses, scores each on relevance/quality, and recommends whether to
replace or keep the current model.

Runs automatically after fine-tuning (called from pipeline scripts) or
standalone:

    # Compare a specific model
    python 08b_compare_models.py --model smolvlm2

    # Compare all models
    python 08b_compare_models.py --model all

    # Custom prompts file
    python 08b_compare_models.py --model moondream2 --prompts my_prompts.json

Output:
    logs/comparisons/compare_<model>_<timestamp>.json
    Terminal: colored side-by-side with verdicts + final recommendation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────
LLAMA_CPP_DIR = Path("../../llama.cpp")
LLAMA_CLI = LLAMA_CPP_DIR / "build" / "bin" / "llama-cli"

GGUF_DIR = Path("../models/gguf")
LOGS_DIR = Path("../logs/comparisons")

# ── ANSI Colors ──────────────────────────────────────────────────
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

# ── Model registry ──────────────────────────────────────────────
MODELS = {
    "smolvlm2": {
        "base_gguf": "../../models/SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf",
        "finetuned_gguf_pattern": "SmolVLM2-500M-*finetuned-*.gguf",
        "ctx": 2048,
        "max_tokens": 150,
        "temp": 0.1,
    },
    "moondream2": {
        "base_gguf": "../../models/moondream2-q4.gguf",
        "finetuned_gguf_pattern": "moondream2-text-model-finetuned-*.gguf",
        "ctx": 2048,
        "max_tokens": 200,
        "temp": 0.3,
    },
    "qwen3vl": {
        "base_gguf": None,
        "finetuned_gguf_pattern": "Qwen3VL-2B-Thinking-finetuned-*.gguf",
        "ctx": 4096,
        "max_tokens": 384,
        "temp": 0.5,
    },
}

# ── Ocula-specific evaluation prompts ────────────────────────────
# Each has a prompt, optional RAG context, and expected behavior traits.
# The comparison will check both base and finetuned against these.
DEFAULT_PROMPTS = [
    # ── RAG-grounded queries (should use context, not hallucinate) ──
    {
        "prompt": "What's John's phone number?",
        "context": "CONTACT: Name: John Smith\nPhone number: +1-555-0123\nEmail address: john@example.com\nWorks at: Acme Corp",
        "category": "contact_lookup",
        "expected": "+1-555-0123",
        "check": "grounded",  # Response must contain data from context
    },
    {
        "prompt": "What meetings do I have tomorrow?",
        "context": "CALENDAR EVENT: Team Standup at Conference Room B on 2026-02-15 09:00\n\nCALENDAR EVENT: 1:1 with Manager at Office on 2026-02-15 14:00",
        "category": "calendar_query",
        "expected": "Team Standup",
        "check": "grounded",
    },
    {
        "prompt": "Find my tax document",
        "context": "FILE: tax_return_2025.pdf — Documents folder, 2.3 MB, modified 2026-01-15",
        "category": "file_search",
        "expected": "tax_return_2025",
        "check": "grounded",
    },
    {
        "prompt": "What did Sarah email me about?",
        "context": "EMAIL: From: sarah@company.com — Subject: Q4 Budget Review\nHi, please review the attached Q4 budget proposal before Friday's meeting.",
        "category": "email_query",
        "expected": "budget",
        "check": "grounded",
    },
    # ── No-context queries (should NOT hallucinate) ──
    {
        "prompt": "What's my boss's phone number?",
        "context": "",
        "category": "no_data_contact",
        "expected": None,
        "check": "no_hallucinate",  # Should NOT invent a phone number
    },
    {
        "prompt": "Show me my vacation photos from Greece",
        "context": "",
        "category": "no_data_photo",
        "expected": None,
        "check": "no_hallucinate",
    },
    # ── General assistant queries ──
    {
        "prompt": "Hello, how are you?",
        "context": "",
        "category": "greeting",
        "expected": None,
        "check": "coherent",  # Just needs to be a coherent greeting response
    },
    {
        "prompt": "What can you help me with?",
        "context": "",
        "category": "capabilities",
        "expected": None,
        "check": "coherent",
    },
    # ── Edge cases (model stress tests) ──
    {
        "prompt": "Summarize this document",
        "context": "FILE: Project Proposal\n\nOcula is a privacy-first mobile AI assistant that runs entirely on-device. "
                   "It uses small language models (500M-2B parameters) with RAG over the user's contacts, "
                   "calendar, files, and photos. No data ever leaves the phone.",
        "category": "summarization",
        "expected": "privacy",
        "check": "grounded",
    },
    {
        "prompt": "asdkjf qwerty gibberish",
        "context": "",
        "category": "garbage_input",
        "expected": None,
        "check": "coherent",  # Should respond politely, not crash
    },
]


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ResponseScore:
    """Scores for a single model response."""
    relevance: float       # 0-1: Does it answer the question?
    grounding: float       # 0-1: Does it use provided context (not invent)?
    coherence: float       # 0-1: Is it readable, non-repetitive?
    conciseness: float     # 0-1: Is it appropriately brief?
    overall: float         # Weighted average

    @property
    def verdict_emoji(self) -> str:
        if self.overall >= 0.7:
            return "+"   # positive
        elif self.overall >= 0.4:
            return "~"   # neutral
        else:
            return "-"   # negative


@dataclass
class PromptComparison:
    """Comparison result for one prompt."""
    prompt: str
    context: str
    category: str
    base_response: str
    finetuned_response: str
    base_score: ResponseScore
    finetuned_score: ResponseScore
    verdict: str          # "positive", "negative", "neutral"
    explanation: str


@dataclass
class ModelComparison:
    """Full comparison result for one model."""
    model: str
    base_gguf: str
    finetuned_gguf: str
    timestamp: str
    comparisons: list
    summary: dict
    recommendation: str   # "REPLACE", "KEEP", "REVIEW"
    recommendation_reason: str


# ── Inference ────────────────────────────────────────────────────

def run_inference(gguf_path: Path, prompt: str, context: str = "",
                  ctx_size: int = 2048, max_tokens: int = 200,
                  temp: float = 0.1) -> str:
    """Run a single prompt through a GGUF model via llama-cli."""
    if not LLAMA_CLI.exists():
        print(f"  {RED}[!] llama-cli not found at {LLAMA_CLI}{NC}")
        return "[ERROR: llama-cli not found]"

    # Build ChatML prompt matching Ocula's format
    if context:
        system_msg = "Answer using ONLY the data below. Do not make up information. Be brief."
        user_msg = f"Data:\n{context}\n\nQ: {prompt}"
    else:
        system_msg = "You are Ocula, a helpful phone assistant. Be brief."
        user_msg = prompt

    full_prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    cmd = [
        str(LLAMA_CLI),
        "--model", str(gguf_path),
        "--prompt", full_prompt,
        "--n-predict", str(max_tokens),
        "--ctx-size", str(ctx_size),
        "--threads", str(min(os.cpu_count() or 4, 8)),
        "--no-display-prompt",
        "--log-disable",
        "--temp", str(temp),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            return f"[ERROR: {result.stderr[:100]}]"
        text = result.stdout.strip()
        # Clean ChatML artifacts
        text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        return text
    except subprocess.TimeoutExpired:
        return "[ERROR: timeout]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Scoring ──────────────────────────────────────────────────────

def score_response(response: str, prompt_item: dict) -> ResponseScore:
    """Score a model response on multiple dimensions."""
    check = prompt_item.get("check", "coherent")
    context = prompt_item.get("context", "")
    expected = prompt_item.get("expected")

    is_error = response.startswith("[ERROR")
    is_empty = len(response.split()) < 3

    # ── Coherence: Is it readable? ──
    coherence = 1.0
    if is_error or is_empty:
        coherence = 0.0
    elif _detect_repetition(response):
        coherence = 0.1
    elif _detect_garbage(response):
        coherence = 0.2
    elif _detect_prompt_leak(response):
        coherence = 0.3

    # ── Relevance: Does it address the question? ──
    relevance = 0.5  # default neutral
    if is_error or is_empty:
        relevance = 0.0
    elif expected and expected.lower() in response.lower():
        relevance = 1.0
    elif expected:
        # Partial: check if query keywords appear
        query_words = {w.lower() for w in prompt_item["prompt"].split()
                       if len(w) >= 3}
        match_count = sum(1 for w in query_words if w in response.lower())
        relevance = min(1.0, match_count / max(len(query_words), 1))

    # ── Grounding: Does it stick to provided data? ──
    grounding = 0.5  # default neutral
    if check == "grounded" and context:
        if expected and expected.lower() in response.lower():
            grounding = 1.0
        elif _has_invented_data(response, context):
            grounding = 0.1  # Invented data not in context
        else:
            grounding = 0.5
    elif check == "no_hallucinate":
        # Should NOT invent phone numbers, emails, dates
        if _has_hallucinated_facts(response):
            grounding = 0.0
        else:
            grounding = 1.0
    elif check == "coherent":
        grounding = 0.7 if coherence > 0.5 else 0.3

    # ── Conciseness ──
    word_count = len(response.split())
    if word_count < 3:
        conciseness = 0.0
    elif word_count <= 50:
        conciseness = 1.0
    elif word_count <= 100:
        conciseness = 0.7
    elif word_count <= 200:
        conciseness = 0.4
    else:
        conciseness = 0.2

    # ── Overall (weighted) ──
    overall = (0.30 * relevance +
               0.35 * grounding +
               0.25 * coherence +
               0.10 * conciseness)

    return ResponseScore(
        relevance=round(relevance, 2),
        grounding=round(grounding, 2),
        coherence=round(coherence, 2),
        conciseness=round(conciseness, 2),
        overall=round(overall, 2),
    )


def _detect_repetition(text: str, window: int = 15) -> bool:
    """Check if the model is stuck repeating itself."""
    words = text.split()
    if len(words) < window * 2:
        return False
    tail = " ".join(words[-window:])
    body = " ".join(words[:-window])
    return tail in body


def _detect_garbage(text: str) -> bool:
    """Check for garbage output (random symbols, encoding errors)."""
    if any(c * 5 in text for c in "!@#$%^&*~`"):
        return True
    # High ratio of non-alphanumeric characters
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return alnum < len(text) * 0.5 if text else True


def _detect_prompt_leak(text: str) -> bool:
    """Check if the model leaked the prompt or system instructions."""
    lower = text.lower()
    markers = ["<|im_start|>", "<|im_end|>", "system\n", "you are ocula",
               "answer using only", "do not make up"]
    return any(m in lower for m in markers)


def _has_invented_data(response: str, context: str) -> bool:
    """Check if response contains specific data not in the context."""
    import re
    # Look for phone numbers in response not in context
    phones_in_resp = set(re.findall(r'\+?\d[\d\-\s]{8,}\d', response))
    phones_in_ctx = set(re.findall(r'\+?\d[\d\-\s]{8,}\d', context))
    if phones_in_resp - phones_in_ctx:
        return True
    # Look for email addresses in response not in context
    emails_in_resp = set(re.findall(r'\S+@\S+\.\S+', response))
    emails_in_ctx = set(re.findall(r'\S+@\S+\.\S+', context))
    if emails_in_resp - emails_in_ctx:
        return True
    return False


def _has_hallucinated_facts(response: str) -> bool:
    """Check if response fabricates phone numbers, emails, or dates."""
    import re
    has_phone = bool(re.search(r'\+?\d[\d\-\s]{8,}\d', response))
    has_email = bool(re.search(r'\w+@\w+\.\w+', response))
    # Specific date patterns like "January 15, 2026" or "2026-01-15"
    has_specific_date = bool(re.search(
        r'\d{4}-\d{2}-\d{2}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}',
        response
    ))
    return has_phone or has_email or has_specific_date


# ── Comparison logic ─────────────────────────────────────────────

def compare_responses(prompt_item: dict, base_resp: str, ft_resp: str
                      ) -> PromptComparison:
    """Compare base vs finetuned responses for a single prompt."""
    base_score = score_response(base_resp, prompt_item)
    ft_score = score_response(ft_resp, prompt_item)

    delta = ft_score.overall - base_score.overall
    if delta > 0.15:
        verdict = "positive"
    elif delta < -0.15:
        verdict = "negative"
    else:
        verdict = "neutral"

    # Build explanation
    parts = []
    if ft_score.grounding > base_score.grounding + 0.1:
        parts.append("better grounded in data")
    elif ft_score.grounding < base_score.grounding - 0.1:
        parts.append("less grounded (more hallucination)")
    if ft_score.relevance > base_score.relevance + 0.1:
        parts.append("more relevant answer")
    elif ft_score.relevance < base_score.relevance - 0.1:
        parts.append("less relevant")
    if ft_score.coherence > base_score.coherence + 0.1:
        parts.append("more coherent")
    elif ft_score.coherence < base_score.coherence - 0.1:
        parts.append("less coherent")

    explanation = "; ".join(parts) if parts else "similar quality"

    return PromptComparison(
        prompt=prompt_item["prompt"],
        context=prompt_item.get("context", ""),
        category=prompt_item["category"],
        base_response=base_resp,
        finetuned_response=ft_resp,
        base_score=base_score,
        finetuned_score=ft_score,
        verdict=verdict,
        explanation=explanation,
    )


def find_gguf(pattern: str) -> Optional[Path]:
    """Find the best quantized GGUF (prefer Q4_K_M, then Q8_0)."""
    candidates = sorted(GGUF_DIR.glob(pattern))
    if not candidates:
        return None
    for c in candidates:
        if "Q4_K_M" in c.name:
            return c
    for c in candidates:
        if "Q8_0" in c.name:
            return c
    return candidates[0]


# ── Main comparison ──────────────────────────────────────────────

def compare_model(model_name: str, prompts: list) -> Optional[ModelComparison]:
    """Run full comparison for a model."""
    info = MODELS[model_name]

    base_path = Path(info["base_gguf"]) if info.get("base_gguf") else None
    ft_path = find_gguf(info["finetuned_gguf_pattern"])

    if base_path and not base_path.exists():
        print(f"  {RED}[!] Base GGUF not found: {base_path}{NC}")
        base_path = None
    if not ft_path or not ft_path.exists():
        print(f"  {RED}[!] Fine-tuned GGUF not found for pattern: {info['finetuned_gguf_pattern']}{NC}")
        ft_path = None

    if not base_path and not ft_path:
        print(f"  {RED}[!] No models found for {model_name}, skipping{NC}")
        return None

    ctx = info["ctx"]
    max_tok = info["max_tokens"]
    temp = info["temp"]

    comparisons = []
    verdicts = {"positive": 0, "negative": 0, "neutral": 0}

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        context = item.get("context", "")

        print(f"\n  {DIM}── Prompt {i+1}/{len(prompts)}: {prompt[:60]}...{NC}")

        # Generate from both models
        base_resp = ""
        ft_resp = ""

        if base_path:
            print(f"    {DIM}Base model...{NC}", end="", flush=True)
            base_resp = run_inference(base_path, prompt, context, ctx, max_tok, temp)
            print(f" {len(base_resp.split())} words")

        if ft_path:
            print(f"    {DIM}Fine-tuned...{NC}", end="", flush=True)
            ft_resp = run_inference(ft_path, prompt, context, ctx, max_tok, temp)
            print(f" {len(ft_resp.split())} words")

        comp = compare_responses(item, base_resp, ft_resp)
        comparisons.append(comp)
        verdicts[comp.verdict] += 1

        # Print side-by-side
        _print_comparison(comp)

    # ── Summary & Recommendation ──
    total = len(comparisons)
    avg_base = sum(c.base_score.overall for c in comparisons) / total if total else 0
    avg_ft = sum(c.finetuned_score.overall for c in comparisons) / total if total else 0
    improvement = avg_ft - avg_base

    summary = {
        "total_prompts": total,
        "positive": verdicts["positive"],
        "negative": verdicts["negative"],
        "neutral": verdicts["neutral"],
        "avg_base_score": round(avg_base, 3),
        "avg_finetuned_score": round(avg_ft, 3),
        "improvement": round(improvement, 3),
    }

    # Decision logic
    if verdicts["negative"] > verdicts["positive"]:
        rec = "KEEP"
        reason = (f"Fine-tuned model regressed on {verdicts['negative']}/{total} prompts. "
                  f"Avg score: {avg_base:.2f} → {avg_ft:.2f} ({improvement:+.2f}). "
                  f"Keep the current base model.")
    elif improvement >= 0.1 and verdicts["positive"] >= verdicts["negative"] * 2:
        rec = "REPLACE"
        reason = (f"Fine-tuned model improved on {verdicts['positive']}/{total} prompts. "
                  f"Avg score: {avg_base:.2f} → {avg_ft:.2f} ({improvement:+.2f}). "
                  f"Replace the base model with the fine-tuned version.")
    elif improvement >= 0.05:
        rec = "REPLACE"
        reason = (f"Modest improvement: {avg_base:.2f} → {avg_ft:.2f} ({improvement:+.2f}). "
                  f"{verdicts['positive']} better, {verdicts['negative']} worse, "
                  f"{verdicts['neutral']} same. Replace recommended.")
    else:
        rec = "REVIEW"
        reason = (f"Mixed results: {avg_base:.2f} → {avg_ft:.2f} ({improvement:+.2f}). "
                  f"{verdicts['positive']} better, {verdicts['negative']} worse, "
                  f"{verdicts['neutral']} same. Manual review recommended.")

    return ModelComparison(
        model=model_name,
        base_gguf=str(base_path) if base_path else "N/A",
        finetuned_gguf=str(ft_path) if ft_path else "N/A",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        comparisons=[asdict(c) for c in comparisons],
        summary=summary,
        recommendation=rec,
        recommendation_reason=reason,
    )


# ── Pretty printing ──────────────────────────────────────────────

def _print_comparison(comp: PromptComparison):
    """Print a single prompt comparison with colored verdict."""
    verdict_colors = {
        "positive": GREEN,
        "negative": RED,
        "neutral": YELLOW,
    }
    verdict_symbols = {
        "positive": "+",
        "negative": "-",
        "neutral": "~",
    }
    color = verdict_colors[comp.verdict]
    symbol = verdict_symbols[comp.verdict]

    # Truncate responses for display
    base_preview = comp.base_response[:120].replace("\n", " ")
    ft_preview = comp.finetuned_response[:120].replace("\n", " ")

    print(f"\n    {BOLD}Before:{NC} {DIM}{base_preview}{'...' if len(comp.base_response) > 120 else ''}{NC}")
    print(f"    {BOLD}After: {NC} {DIM}{ft_preview}{'...' if len(comp.finetuned_response) > 120 else ''}{NC}")
    print(f"    {BOLD}Score:{NC}  base={comp.base_score.overall:.2f}  fine-tuned={comp.finetuned_score.overall:.2f}  "
          f"{color}[{symbol}] {comp.verdict.upper()}{NC}  ({comp.explanation})")


def _print_final_report(result: ModelComparison):
    """Print the final recommendation report."""
    s = result.summary
    rec_colors = {"REPLACE": GREEN, "KEEP": RED, "REVIEW": YELLOW}
    rec_color = rec_colors.get(result.recommendation, NC)

    print(f"\n{'='*65}")
    print(f"  {BOLD}COMPARISON REPORT: {result.model}{NC}")
    print(f"{'='*65}")
    print(f"  Base model:       {result.base_gguf}")
    print(f"  Fine-tuned model: {result.finetuned_gguf}")
    print(f"  Prompts tested:   {s['total_prompts']}")
    print(f"")
    print(f"  {GREEN}[+] Improved:  {s['positive']}{NC}")
    print(f"  {RED}[-] Regressed: {s['negative']}{NC}")
    print(f"  {YELLOW}[~] Same:      {s['neutral']}{NC}")
    print(f"")
    print(f"  Average score:    {s['avg_base_score']:.2f} → {s['avg_finetuned_score']:.2f}  "
          f"({s['improvement']:+.3f})")
    print(f"")
    print(f"  {BOLD}Recommendation: {rec_color}{result.recommendation}{NC}")
    print(f"  {DIM}{result.recommendation_reason}{NC}")
    print(f"{'='*65}")


# ── Entry point ──────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned models with side-by-side responses"
    )
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        default="all", help="Which model to compare")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Custom prompts JSON file (list of {prompt, context, category, expected, check})")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to specific JSON file")
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    # Load prompts
    if args.prompts and Path(args.prompts).exists():
        with open(args.prompts) as f:
            prompts = json.load(f)
        print(f"  Loaded {len(prompts)} custom prompts from {args.prompts}")
    else:
        prompts = DEFAULT_PROMPTS

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_name in targets:
        print(f"\n{BOLD}{'='*65}{NC}")
        print(f"  {CYAN}Comparing: {model_name} (base vs fine-tuned){NC}")
        print(f"{BOLD}{'='*65}{NC}")

        result = compare_model(model_name, prompts)
        if result is None:
            continue

        all_results[model_name] = asdict(result)
        _print_final_report(result)

    # Save results
    if all_results:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = args.output or str(LOGS_DIR / f"compare_{args.model}_{ts}.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  {GREEN}Results saved to {output_path}{NC}")

    # Exit code: 0 if all REPLACE, 1 if any KEEP, 2 if REVIEW
    if all_results:
        recs = [r["recommendation"] for r in all_results.values()]
        if all(r == "REPLACE" for r in recs):
            sys.exit(0)
        elif any(r == "KEEP" for r in recs):
            sys.exit(1)
        else:
            sys.exit(2)


if __name__ == "__main__":
    main()
