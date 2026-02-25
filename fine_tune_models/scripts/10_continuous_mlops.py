#!/usr/bin/env python3
"""
Continuous Ocula MLOps
======================

Iterative pipeline:
1) Train Ocula Text from current champion (or base Qwen)
2) Benchmark candidate vs champion
3) Promote winner as next-cycle base
4) Train Ocula VL from current champion (or base Qwen)
5) Benchmark candidate vs champion
6) Promote winner

Usage:
    python 10_continuous_mlops.py --cycles 3 --backend cuda --compress balanced
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
LOGS_DIR = ROOT_DIR / "logs" / "mlops"
TMP_DIR = LOGS_DIR / "tmp"
STATE_PATH = LOGS_DIR / "state.json"
LLAMA_CLI = ROOT_DIR.parent / "llama.cpp" / "build" / "bin" / "llama-cli"
GGUF_DIR = ROOT_DIR / "models" / "gguf"

TEXT_CONFIG = ROOT_DIR / "configs" / "qwen25_1_5b_text.yaml"
VL_CONFIG = ROOT_DIR / "configs" / "qwen3vl_vision.yaml"


DEFAULT_PROMPTS = [
    "Summarize this note in 3 bullets: Ocula is private, on-device, and fast.",
    "Draft a concise professional email asking for a meeting reschedule.",
    "Explain RAG in simple terms for a mobile app user.",
    "I forgot my context. Ask me one clarifying question before answering.",
    "Translate this to Spanish: The meeting starts at 9 AM tomorrow.",
]


def run(cmd: List[str], cwd: Path = SCRIPT_DIR) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def run_capture(cmd: List[str], cwd: Path = SCRIPT_DIR, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": {"champion_hf": None, "champion_gguf": None, "history": []},
            "vl": {"champion_hf": None, "champion_gguf": None, "history": []},
        }
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def parse_tps(stderr_text: str) -> Optional[float]:
    # Example line usually contains "... tokens per second)"
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second", stderr_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def repetition_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    unique = len(set(words))
    return 1.0 - (unique / len(words))


def benchmark_model(gguf_path: Path, ctx: int = 2048, n_predict: int = 96) -> Dict[str, Any]:
    if not LLAMA_CLI.exists():
        raise RuntimeError(f"llama-cli not found at {LLAMA_CLI}")
    if not gguf_path.exists():
        raise RuntimeError(f"GGUF missing: {gguf_path}")

    success = 0
    tps_vals: List[float] = []
    quality_vals: List[float] = []

    for prompt in DEFAULT_PROMPTS:
        chat_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        cmd = [
            str(LLAMA_CLI),
            "--model", str(gguf_path),
            "--prompt", chat_prompt,
            "--ctx-size", str(ctx),
            "--n-predict", str(n_predict),
            "--temp", "0.2",
            "--threads", str(min(os.cpu_count() or 4, 8)),
            "--no-display-prompt",
        ]
        proc = run_capture(cmd, timeout=240)
        if proc.returncode != 0:
            continue

        text = proc.stdout.strip()
        if len(text.split()) >= 6:
            success += 1

        tps = parse_tps(proc.stderr + "\n" + proc.stdout)
        if tps:
            tps_vals.append(tps)

        rep = repetition_ratio(text)
        # quality heuristic: longer coherent output and low repetition
        q = max(0.0, min(1.0, (len(text.split()) / 80.0))) * (1.0 - rep)
        quality_vals.append(q)

    n = len(DEFAULT_PROMPTS)
    success_rate = success / n if n else 0.0
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0
    avg_quality = sum(quality_vals) / len(quality_vals) if quality_vals else 0.0

    return {
        "success_rate": round(success_rate, 4),
        "avg_tps": round(avg_tps, 4),
        "avg_quality": round(avg_quality, 4),
        "n_prompts": n,
    }


def should_promote(candidate: Dict[str, Any], champion: Optional[Dict[str, Any]]) -> bool:
    if champion is None:
        return True
    # Gate rules: maintain answerability, avoid severe slowdown, slight quality gain.
    if candidate["success_rate"] < champion["success_rate"]:
        return False
    if champion["avg_tps"] > 0 and candidate["avg_tps"] < champion["avg_tps"] * 0.85:
        return False
    if candidate["avg_quality"] + 0.02 < champion["avg_quality"]:
        return False
    return True


def pick_candidate_gguf(prefix: str) -> Path:
    cands = sorted(GGUF_DIR.glob(f"{prefix}-*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise RuntimeError(f"No GGUF found for prefix: {prefix}")
    for preferred in ("Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", "Q3_K_M"):
        for c in cands:
            if preferred in c.name:
                return c
    return cands[0]


def write_temp_config(base_config: Path, out_path: Path, short_name: str, base_hf: str) -> Dict[str, Any]:
    cfg = load_yaml(base_config)
    cfg.setdefault("model", {})
    cfg["model"]["short_name"] = short_name
    cfg["model"]["local_path"] = base_hf
    save_yaml(out_path, cfg)
    return cfg


def train_text_cycle(cycle: int, backend: str, compress: str, base_hf: str) -> Dict[str, str]:
    short = f"ocula-lite-c{cycle}"
    tmp_cfg = TMP_DIR / f"text_cycle_{cycle}.yaml"
    cfg = write_temp_config(TEXT_CONFIG, tmp_cfg, short, base_hf)

    train_backend = backend if backend in ("cuda", "mlx") else "auto"
    merge_backend = "mlx" if backend == "mlx" else "cuda"

    run([sys.executable, "04_train_text_qwen.py", "--config", str(tmp_cfg), "--backend", train_backend])

    method = cfg.get("training", {}).get("method", "qlora")
    adapter_dir = ROOT_DIR / "models" / "lora_adapters" / f"{short}-{method}"
    merged_dir = ROOT_DIR / "models" / "merged" / f"{short}-merged"

    run([
        sys.executable, "06_merge_lora.py",
        "--model", "ocula_lite",
        "--backend", merge_backend,
        "--base-path", base_hf,
        "--adapter-path", str(adapter_dir),
        "--output-path", str(merged_dir),
    ])

    gguf_prefix = f"Ocula-Lite-Cycle{cycle}"
    run([
        sys.executable, "07_quantize_gguf.py",
        "--model", "ocula_lite",
        "--config", str(tmp_cfg),
        "--merged-path", str(merged_dir),
        "--output-name", gguf_prefix,
        "--mobile-preset", compress,
    ])

    cand_gguf = pick_candidate_gguf(gguf_prefix)
    return {"merged_hf": str(merged_dir), "candidate_gguf": str(cand_gguf)}


def train_vl_cycle(cycle: int, backend: str, compress: str, base_hf: str, max_samples: int) -> Dict[str, str]:
    short = f"ocula-vl-c{cycle}"
    tmp_cfg = TMP_DIR / f"vl_cycle_{cycle}.yaml"
    cfg = write_temp_config(VL_CONFIG, tmp_cfg, short, base_hf)

    data_train = cfg.get("data", {}).get("train", "../data/vision_qwen3vl/qwen3vl_vision_train.jsonl")
    data_val = cfg.get("data", {}).get("val", "../data/vision_qwen3vl/qwen3vl_vision_val.jsonl")
    output_dir = ROOT_DIR / "models" / "lora_adapters" / f"{short}-vision"
    merged_dir = ROOT_DIR / "models" / "merged" / f"{short}-vision-merged"

    run([
        sys.executable, "04b_train_vision_qwen.py",
        "--config", str(tmp_cfg),
        "--backend", backend,
        "--train-data", data_train,
        "--val-data", data_val,
        "--output", str(output_dir),
        "--max-samples", str(max_samples),
    ])

    if backend == "mlx":
        run([
            sys.executable, "-m", "mlx_vlm.fuse",
            "--model", base_hf,
            "--adapter-path", str(output_dir),
            "--save-path", str(merged_dir),
        ])

    gguf_prefix = f"Ocula-VL-Cycle{cycle}"
    run([
        sys.executable, "07_quantize_gguf.py",
        "--model", "ocula_plus",
        "--config", str(tmp_cfg),
        "--merged-path", str(merged_dir),
        "--output-name", gguf_prefix,
        "--mobile-preset", compress,
    ])

    cand_gguf = pick_candidate_gguf(gguf_prefix)
    return {"merged_hf": str(merged_dir), "candidate_gguf": str(cand_gguf)}


def cycle_stage(
    state: Dict[str, Any],
    stage: str,
    cycle: int,
    backend: str,
    compress: str,
    max_samples: int,
) -> None:
    assert stage in ("text", "vl")
    node = state[stage]

    if stage == "text":
        base_hf = node["champion_hf"] or load_yaml(TEXT_CONFIG)["model"]["local_path"]
        out = train_text_cycle(cycle, backend, compress, base_hf)
    else:
        base_hf = node["champion_hf"] or load_yaml(VL_CONFIG)["model"]["local_path"]
        out = train_vl_cycle(cycle, backend, compress, base_hf, max_samples)

    candidate_hf = out["merged_hf"]
    candidate_gguf = out["candidate_gguf"]
    cand_metrics = benchmark_model(Path(candidate_gguf), ctx=4096 if stage == "vl" else 2048)

    champ_metrics = None
    if node.get("champion_gguf"):
        champ_metrics = benchmark_model(Path(node["champion_gguf"]), ctx=4096 if stage == "vl" else 2048)

    promote = should_promote(cand_metrics, champ_metrics)
    record = {
        "cycle": cycle,
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_hf": base_hf,
        "candidate_hf": candidate_hf,
        "candidate_gguf": candidate_gguf,
        "candidate_metrics": cand_metrics,
        "champion_before_hf": node.get("champion_hf"),
        "champion_before_gguf": node.get("champion_gguf"),
        "champion_before_metrics": champ_metrics,
        "promoted": promote,
    }
    node["history"].append(record)

    if promote:
        node["champion_hf"] = candidate_hf
        node["champion_gguf"] = candidate_gguf
        print(f"[PROMOTE] {stage} cycle {cycle} candidate promoted.")
    else:
        print(f"[KEEP] {stage} cycle {cycle} candidate rejected; champion retained.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Ocula MLOps with benchmark gates")
    parser.add_argument("--cycles", type=int, default=2, help="Number of iterative cycles")
    parser.add_argument("--backend", choices=["auto", "cuda", "mlx", "mps", "cpu"], default="auto")
    parser.add_argument("--compress", choices=["balanced", "aggressive", "extreme"], default="balanced")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--start-cycle", type=int, default=1)
    parser.add_argument("--stages", nargs="+", choices=["text", "vl"], default=["text", "vl"])
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()

    for cycle in range(args.start_cycle, args.start_cycle + args.cycles):
        print(f"\n{'=' * 70}")
        print(f"Continuous MLOps Cycle {cycle}")
        print(f"{'=' * 70}")
        for stage in args.stages:
            cycle_stage(
                state=state,
                stage=stage,
                cycle=cycle,
                backend=args.backend,
                compress=args.compress,
                max_samples=args.max_samples,
            )
            save_state(state)

    save_state(state)
    print(f"\n[OK] Completed {args.cycles} cycle(s). State: {STATE_PATH}")


if __name__ == "__main__":
    main()
