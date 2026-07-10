"""Shared helpers for the qwen4b per-step scripts.

Defines:
  - MODEL_SPEC       : the qwen3-4B-Base spec used in every step
  - LOGS_DIR         : where every step writes its `.log` and `.json`
  - tee_to(name)     : duplicate stdout to a file under LOGS_DIR
  - save_json(...)   : structured-result sink next to the log
  - run_step(...)    : load model, tee the log, invoke step body, clean up
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

# Load HF_TOKEN, PYTORCH_ENABLE_MPS_FALLBACK, etc. BEFORE torch / transformers
# import-paths get triggered.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from model_finder.models import ModelSpec  # noqa: E402
from model_finder.surprisal import load_model  # noqa: E402
from model_finder.utils import cleanup, get_device  # noqa: E402

# Target model — fixed for this entire folder. See README for the model_finder
# selection that motivated this choice.
MODEL_SPEC = ModelSpec(hf_name="Qwen/Qwen3-4B-Base", approx_params="4B")

QWEN4B_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = QWEN4B_DIR / "logs"


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
    """Redirect stdout so every print also lands in LOGS_DIR/<step_name>.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{step_name}.log"
    # The handle has to outlive this function — it is wired into sys.stdout.
    f = open(log_path, "w")  # noqa: SIM115
    sys.stdout = _Tee(sys.__stdout__, f)
    return log_path


def save_json(step_name: str, payload: dict) -> Path:
    """Persist structured results next to the log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"{step_name}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out}")
    return out


def run_step(
    step_name: str,
    body: Callable[[object, object, str], dict],
) -> None:
    """Standard scaffolding every step uses.

    `body(model, tokenizer, device)` returns a dict; we write it as JSON next
    to the .log so re-running a step is fully reproducible.
    """
    tee_to(step_name)
    print(f"=== {step_name} ===")
    print(f"device = {get_device()}")
    print(f"model  = {MODEL_SPEC.hf_name} (dtype={MODEL_SPEC.dtype})")
    print()

    device = get_device()
    model, tokenizer = load_model(device, MODEL_SPEC)
    t0 = time.time()
    try:
        results = body(model, tokenizer, device)
        save_json(step_name, {"step": step_name, "results": results})
    finally:
        del model, tokenizer
        cleanup(device)
        print(f"\nElapsed: {time.time() - t0:.1f}s")
