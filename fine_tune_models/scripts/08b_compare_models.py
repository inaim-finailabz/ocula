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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

# ── Paths ────────────────────────────────────────────────────────
def find_llama_cpp():
    """Find llama.cpp in common locations relative to scripts dir."""
    candidates = [
        Path("../../llama.cpp"),   # Ocula/fine_tune_models/scripts/ → Ocula/llama.cpp
        Path("../llama.cpp"),      # ocula/scripts/ → ocula/llama.cpp (flat layout)
    ]
    for p in candidates:
        if (p / "build" / "bin" / "llama-cli").exists():
            return p
    return candidates[0]

LLAMA_CPP_DIR = find_llama_cpp()
LLAMA_CLI = LLAMA_CPP_DIR / "build" / "bin" / "llama-cli"
LLAMA_SERVER = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"

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
        "base_gguf_pattern": "../models/base/smolvlm2/SmolVLM2-*500M*Q*.gguf",
        "finetuned_gguf_pattern": "SmolVLM2-500M-*finetuned-*.gguf",
        "finetuned_model_pattern": "SmolVLM2-500M-*finetuned-*.gguf",
        "base_api_url": "http://localhost:8080/v1",
        "finetuned_api_url": "http://localhost:8081/v1",
        "ctx": 2048,
        "max_tokens": 150,
        "temp": 0.1,
    },
    "moondream2": {
        "base_gguf": "../../models/moondream2-q4.gguf",
        "finetuned_gguf_pattern": "moondream2-text-model-finetuned-*.gguf",
        "finetuned_model_pattern": "moondream2-text-model-finetuned-*.gguf",
        "ctx": 2048,
        "max_tokens": 200,
        "temp": 0.3,
    },
    "qwen3vl": {
        "base_gguf": "/home/issam-naim/Documents/projects/ocula/models/base/qwen3vl-2b/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        # Base model downloaded under models/Qwen/Qwen3-VL-2B-Instruct-GGUF/
        # This pattern is resolved from GGUF_DIR (../models/gguf), so use ../Qwen/...
        "base_gguf_pattern": "../Qwen/Qwen3-VL-2B-Instruct-GGUF/*.gguf",
        "finetuned_gguf_pattern": "Qwen3-VL-2B-finetuned-*.gguf",
        "finetuned_model_pattern": "Qwen3-VL-2B-finetuned-*.gguf",
        "base_api_url": "http://localhost:8080/v1",
        "finetuned_api_url": "http://localhost:8081/v1",
        "ctx": 2048,
        "max_tokens": 200,
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

def _build_chat_messages(prompt: str, context: str = "") -> list:
    """Build chat messages matching Ocula's format."""
    if context:
        system_msg = "Answer using ONLY the data below. Do not make up information. Be brief."
        user_msg = f"Data:\n{context}\n\nQ: {prompt}"
    else:
        system_msg = "You are Ocula, a helpful phone assistant. Be brief."
        user_msg = prompt

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _build_chat_endpoint(api_url: str) -> str:
    """Normalize API base URL to the chat completions endpoint."""
    if api_url.endswith("/v1/chat/completions") or api_url.endswith("/chat/completions"):
        return api_url
    if api_url.endswith("/v1/"):
        return api_url + "chat/completions"
    if api_url.endswith("/v1"):
        return api_url + "/chat/completions"
    return api_url.rstrip("/") + "/v1/chat/completions"

def _build_models_endpoint(api_url: str) -> str:
    if api_url.endswith("/v1/models"):
        return api_url
    if api_url.endswith("/v1/"):
        return api_url + "models"
    if api_url.endswith("/v1"):
        return api_url + "/models"
    return api_url.rstrip("/") + "/v1/models"


def _is_server_healthy(api_url: str, timeout_s: float = 2.0) -> bool:
    endpoint = _build_models_endpoint(api_url)
    try:
        with urlrequest.urlopen(endpoint, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_llama_server(model_path: Path, api_url: str, ctx_size: int) -> Optional[subprocess.Popen]:
    if not LLAMA_SERVER.exists():
        print(f"  {RED}[!] llama-server not found at {LLAMA_SERVER}{NC}")
        return None

    # Extract host/port from api_url (expects http://host:port[/v1])
    host = "127.0.0.1"
    port = "8080"
    if "://" in api_url:
        try:
            hostport = api_url.split("://", 1)[1].split("/", 1)[0]
            if ":" in hostport:
                host, port = hostport.split(":", 1)
            else:
                host = hostport
        except Exception:
            pass

    log_path = LOGS_DIR / f"llama_server_{model_path.stem}_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")

    cmd = [
        str(LLAMA_SERVER),
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--threads", str(min(os.cpu_count() or 4, 8)),
    ]
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
    return proc


def _derive_api_url_with_port(api_url: str, new_port: int) -> str:
    if "://" not in api_url:
        return f"http://127.0.0.1:{new_port}/v1"
    scheme, rest = api_url.split("://", 1)
    hostport = rest.split("/", 1)[0]
    host = hostport.split(":", 1)[0]
    return f"{scheme}://{host}:{new_port}/v1"


def _extract_port(api_url: str, default_port: int = 8080) -> int:
    try:
        if "://" not in api_url:
            return default_port
        hostport = api_url.split("://", 1)[1].split("/", 1)[0]
        if ":" in hostport:
            return int(hostport.rsplit(":", 1)[1])
    except Exception:
        pass
    return default_port


def _kill_llama_server_on_port(port: int) -> int:
    """Kill llama-server processes listening on a TCP port. Returns kill count."""
    try:
        proc = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return 0

    if proc.returncode != 0 or not proc.stdout.strip():
        return 0

    killed = 0
    pids = {line.strip() for line in proc.stdout.splitlines() if line.strip().isdigit()}
    for pid in pids:
        try:
            cmd = subprocess.run(
                ["ps", "-p", pid, "-o", "command="],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cmdline = cmd.stdout.strip()
            if "llama-server" not in cmdline:
                continue
            os.kill(int(pid), signal.SIGTERM)
            killed += 1
        except Exception:
            continue
    return killed


def _ensure_server(api_url: str, model_path: Optional[Path], ctx_size: int) -> Tuple[bool, Optional[subprocess.Popen]]:
    if _is_server_healthy(api_url):
        return True, None
    if model_path is None:
        print(f"  {RED}[!] API server not reachable at {api_url} and no model path provided{NC}")
        return False, None
    proc = _start_llama_server(model_path, api_url, ctx_size)
    if proc is None:
        return False, None

    # Wait for health
    for _ in range(20):
        if _is_server_healthy(api_url, timeout_s=1.0):
            return True, proc
        time.sleep(0.5)

    print(f"  {RED}[!] API server failed health check at {api_url}{NC}")
    return False, proc


def run_inference_api(api_url: str, model_id: Optional[str],
                      prompt: str, context: str = "",
                      max_tokens: int = 200, temp: float = 0.1,
                      api_key: Optional[str] = None) -> str:
    """Run a single prompt through a model via OpenAI-compatible HTTP API."""
    endpoint = _build_chat_endpoint(api_url)
    body: Dict[str, Any] = {
        "messages": _build_chat_messages(prompt, context),
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if model_id:
        body["model"] = model_id

    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urlrequest.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return f"[ERROR]\nHTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
    except URLError as e:
        return f"[ERROR]\nAPI unreachable: {e}"
    except Exception as e:
        return f"[ERROR]\nAPI error: {e}"

    try:
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"[ERROR]\nUnexpected API response: {json.dumps(payload)[:500]}"


def run_inference_cli(gguf_path: Path, prompt: str, context: str = "",
                      ctx_size: int = 2048, max_tokens: int = 200,
                      temp: float = 0.1) -> str:
    """Run a single prompt through a GGUF model via llama-cli."""
    if not LLAMA_CLI.exists():
        print(f"  {RED}[!] llama-cli not found at {LLAMA_CLI}{NC}")
        return "[ERROR: llama-cli not found]"

    # Build ChatML prompt matching Ocula's format
    messages = _build_chat_messages(prompt, context)
    full_prompt = (
        f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
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
        # FIX: Exit after one generation instead of waiting for more input
        "--single-turn",
    ]

    try:
        # Run without check=True. Prioritize stdout even if stderr has content (like perf stats).
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        output = proc.stdout.strip()

        # If stdout is empty and there was an error, then use stderr.
        if not output and proc.returncode != 0:
            output = f"[ERROR]\n{proc.stderr.strip()}"
            print(f"  [!] Error running {gguf_path.name}:\n{proc.stderr.strip()}")

    except subprocess.TimeoutExpired as e:
        # The process timed out, but we might still have partial output
        output = e.stdout.strip() if e.stdout else "[TIMEOUT]"
        if e.stderr:
            output += f"\n[STDERR]\n{e.stderr.strip()}"
        print(f"  [!] Timeout after 90s for {gguf_path.name}, captured partial output.")

    # Clean ChatML artifacts
    output = output.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return output if output else "[ERROR: empty response]"

def score_response(response: str, prompt_item: dict) -> ResponseScore:
    """Score a model response on multiple dimensions."""
    check = prompt_item.get("check", "coherent")
    context = prompt_item.get("context", "")
    expected = prompt_item.get("expected")

    is_error = response.startswith("[ERROR")
    is_empty = len(response.split()) < 3
    has_exact_expected = bool(expected and expected.lower() in response.lower())

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
    elif has_exact_expected:
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
        if has_exact_expected:
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

    # Short exact matches (e.g., "tax_return_2025.pdf") are valid high-quality answers.
    # Do this after base heuristics so we override the "<3 words" penalty.
    if has_exact_expected and not is_error:
        relevance = 1.0
        if check == "grounded":
            grounding = 1.0
        coherence = max(coherence, 0.8)
        conciseness = 1.0

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


def find_base_gguf(model_name: str, info: dict, finetuned_path: Optional[Path]) -> Optional[Path]:
    """Resolve base GGUF via explicit path, optional pattern, then model-specific fallback."""
    base_gguf = info.get("base_gguf")
    if base_gguf:
        p = Path(base_gguf)
        return p if p.exists() else None

    base_pattern = info.get("base_gguf_pattern")
    if base_pattern:
        cands = sorted(GGUF_DIR.glob(base_pattern))
        cands = [c for c in cands if "finetuned" not in c.name.lower()]
        if cands:
            if model_name == "smolvlm2":
                # For SmolVLM2, prefer Q8_0 for baseline quality parity.
                for c in cands:
                    if "Q8_0" in c.name:
                        return c
            for c in cands:
                if "Q4_K_M" in c.name:
                    return c
            return cands[0]

    # Fallback: look for non-finetuned GGUF with matching family prefix
    family_prefixes = {
        "qwen3vl": "Qwen3-VL-2B*",
        "smolvlm2": "SmolVLM2-*",
        "moondream2": "moondream2-*",
    }
    prefix = family_prefixes.get(model_name)
    if prefix:
        # Search both GGUF_DIR and models/base/ subdirectories
        search_dirs = [GGUF_DIR, GGUF_DIR.parent / "base"]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            cands = sorted(search_dir.rglob(f"{prefix}.gguf"))
            cands = [c for c in cands if "finetuned" not in c.name.lower()]
            if model_name == "smolvlm2":
                cands = [c for c in cands if "500m" in c.name.lower()] or cands
            if finetuned_path:
                cands = [c for c in cands if c.name != finetuned_path.name]
            if cands:
                if model_name == "smolvlm2":
                    for c in cands:
                        if "Q8_0" in c.name:
                            return c
                for c in cands:
                    if "Q4_K_M" in c.name:
                        return c
                return cands[0]

    return None


# ── Main comparison ──────────────────────────────────────────────

def compare_model(model_name: str, prompts: list, backend_default: str,
                  api_url_default: str, api_key: Optional[str]) -> Optional[ModelComparison]:
    """Run full comparison for a model."""
    info = MODELS[model_name]

    backend = info.get("backend", backend_default)
    api_url = info.get("api_url", api_url_default)
    base_api_url = info.get("base_api_url", api_url)
    finetuned_api_url = info.get("finetuned_api_url", api_url)
    base_model_id = info.get("base_model_id")
    finetuned_model_id = info.get("finetuned_model_id")

    finetuned_pattern = info.get("finetuned_gguf_pattern") or info.get("finetuned_model_pattern")
    if not finetuned_pattern:
        print(f"  {RED}[!] Missing finetuned GGUF pattern for {model_name}{NC}")
        return None
    finetuned_gguf_path = find_gguf(finetuned_pattern)
    base_gguf_path = find_base_gguf(model_name, info, finetuned_gguf_path)

    ctx = info["ctx"]
    max_tok = info["max_tokens"]
    temp = info["temp"]

    if backend == "cli":
        if not base_gguf_path:
            print(f"  {RED}[!] Base model not found for {model_name}.{NC}")
        if not finetuned_gguf_path or not finetuned_gguf_path.exists():
            print(f"  {RED}[!] Fine-tuned GGUF not found for pattern: {finetuned_pattern}{NC}")
            finetuned_gguf_path = None
    else:
        # Auto-derive finetuned API URL if both use the same URL and we have both GGUFs.
        if base_api_url == finetuned_api_url and base_gguf_path and finetuned_gguf_path:
            try:
                base_port = int(base_api_url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[1])
                finetuned_api_url = _derive_api_url_with_port(base_api_url, base_port + 1)
                print(f"  {DIM}API URLs: base={base_api_url} finetuned={finetuned_api_url}{NC}")
            except Exception:
                pass

        # API mode: if a model_id is missing, the server's loaded model will be used.
        if base_model_id is None and finetuned_model_id is None and base_api_url == finetuned_api_url:
            print(f"  {YELLOW}[~] API mode with same URL and no model IDs: "
                  f"base/fine-tuned will be identical unless your server routes models.{NC}")

        # Auto-start servers if not healthy (for local llama.cpp usage)
        server_procs: List[subprocess.Popen] = []
        if base_gguf_path is None and base_model_id is None:
            print(f"  {RED}[!] Base model not configured for API comparison. "
                  f"Set base_api_url+base_model_id or base_gguf.{NC}")
            return None
        ok_base, proc_base = _ensure_server(base_api_url, base_gguf_path, ctx)
        if proc_base:
            server_procs.append(proc_base)
        if finetuned_gguf_path is None and finetuned_model_id is None:
            print(f"  {RED}[!] Fine-tuned model not configured for API comparison. "
                  f"Set finetuned_api_url+finetuned_model_id or finetuned_gguf_pattern.{NC}")
            return None
        ok_ft, proc_ft = _ensure_server(finetuned_api_url, finetuned_gguf_path, ctx)
        if proc_ft:
            server_procs.append(proc_ft)
        if not ok_base or not ok_ft:
            return None

    # Comparison is only meaningful when both models are present.
    if not base_gguf_path and base_model_id is None:
        print(f"  {RED}[!] Base model missing for {model_name}.{NC}")
        return None
    if not finetuned_gguf_path and finetuned_model_id is None:
        print(f"  {RED}[!] Fine-tuned model missing for {model_name}.{NC}")
        return None

    comparisons = []
    verdicts = {"positive": 0, "negative": 0, "neutral": 0}

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        context = item.get("context", "")

        print(f"\n  {DIM}── Prompt {i+1}/{len(prompts)}: {prompt[:60]}...{NC}")

        # Generate from both models
        base_resp = ""
        ft_resp = ""

        if backend == "cli" and base_gguf_path:
            print(f"    {DIM}Base model...{NC}", end="", flush=True)
            base_resp = run_inference_cli(base_gguf_path, prompt, context, ctx, max_tok, temp)
            print(f" {len(base_resp.split())} words")
        elif backend == "api":
            print(f"    {DIM}Base model (API)...{NC}", end="", flush=True)
            base_resp = run_inference_api(base_api_url, base_model_id, prompt, context, max_tok, temp, api_key)
            print(f" {len(base_resp.split())} words")

        if backend == "cli" and finetuned_gguf_path:
            print(f"    {DIM}Fine-tuned...{NC}", end="", flush=True)
            ft_resp = run_inference_cli(finetuned_gguf_path, prompt, context, ctx, max_tok, temp)
            print(f" {len(ft_resp.split())} words")
        elif backend == "api":
            print(f"    {DIM}Fine-tuned (API)...{NC}", end="", flush=True)
            ft_resp = run_inference_api(finetuned_api_url, finetuned_model_id, prompt, context, max_tok, temp, api_key)
            print(f" {len(ft_resp.split())} words")

        comp = compare_responses(item, base_resp, ft_resp)
        comparisons.append(comp)
        verdicts[comp.verdict] += 1

        # Print side-by-side
        _print_comparison(comp)

    # Stop any servers we started
    if backend == "api":
        if server_procs:
            print(f"  {DIM}Stopping API server(s)...{NC}")
            try:
                for p in server_procs:
                    p.terminate()
                for p in server_procs:
                    p.wait(timeout=10)
            except Exception as e:
                print(f"  {YELLOW}[~] Could not stop started server process: {e}{NC}")

        # Also stop pre-existing llama-server processes bound to comparison ports.
        killed_total = 0
        for port in sorted({_extract_port(base_api_url), _extract_port(finetuned_api_url)}):
            killed_total += _kill_llama_server_on_port(port)
        if killed_total > 0:
            print(f"  {DIM}Stopped {killed_total} llama-server process(es) on compare ports.{NC}")

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
        base_gguf=str(base_model_id or base_gguf_path or "N/A"),
        finetuned_gguf=str(finetuned_model_id or finetuned_gguf_path or "N/A"),
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
    parser.add_argument("--backend", choices=["cli", "api"],
                        default=os.environ.get("OCULA_COMPARE_BACKEND", "cli"),
                        help="Inference backend (default: cli)")
    parser.add_argument("--api-url", type=str,
                        default=os.environ.get("LLAMA_API_URL", "http://localhost:8080/v1"),
                        help="Base URL for llama-server (default: http://localhost:8080/v1)")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("LLAMA_API_KEY", None),
                        help="Optional API key for Authorization header")
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

        result = compare_model(model_name, prompts, args.backend, args.api_url, args.api_key)
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
    else:
        print(f"\n  {RED}[!] No comparisons were run. Check API server config/health.{NC}")
        sys.exit(2)

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
