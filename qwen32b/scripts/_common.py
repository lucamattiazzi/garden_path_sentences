"""Shared scaffolding for the qwen32b/ triage scripts.

Mirrors `qwen4b/scripts/_common.py` but loads the model via `mlx_lm`
instead of `transformers`. Identical tee / save_json / run_step helpers
so the per-step files at the top of `scripts/` look the same shape.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from _surprisal import load_model  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-32B-MLX-8bit"

QWEN32B_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = QWEN32B_DIR / "logs"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def tee_to(step_name: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{step_name}.log"
    # Handle has to outlive this function — it is wired into sys.stdout.
    f = open(log_path, "w")  # noqa: SIM115
    sys.stdout = _Tee(sys.__stdout__, f)
    return log_path


def save_json(step_name: str, payload: dict) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"{step_name}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out}")
    return out


def run_step(
    step_name: str,
    body: Callable[[object, object], dict],
) -> None:
    """Standard scaffolding every step uses."""
    tee_to(step_name)
    print(f"=== {step_name} ===")
    print(f"model = {MODEL_NAME} (MLX-LM, 8-bit quantised)")
    print()

    model, tokenizer = load_model(MODEL_NAME)
    t0 = time.time()
    try:
        results = body(model, tokenizer)
        save_json(step_name, {"step": step_name, "model": MODEL_NAME, "results": results})
    finally:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
