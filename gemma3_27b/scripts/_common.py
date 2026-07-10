"""Shared helpers for the gemma3_27b per-step scripts.

Mirror of qwen4b/scripts/_common.py, targeting `google/gemma-3-27b-pt` — a
dense, base (pretrained, "-pt") Gemma-3 checkpoint. Gemma is trained in bf16
and produces NaN logits in fp16, so dtype is pinned to bfloat16.

Gemma-3 is standard softmax attention with a per-head `o_proj`, so the full
ablation + attention pipeline applies (unlike the Qwen3.5 Gated-DeltaNet
hybrid). The 27B (~54 GB bf16) is intended for the cloud/GPU runner, not the Mac.

The ablation/diagnostic libraries (`_ablation.py`, `_extra.py`) are reused from
qwen4b/scripts, appended to sys.path below (appended, not prepended: this
folder's `_common` must keep winning name resolution). Model-structure access
inside those libs goes through `model_finder.arch`, which handles Gemma's
nested layout transparently.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Load HF_TOKEN, PYTORCH_ENABLE_MPS_FALLBACK, etc. BEFORE torch / transformers
# import-paths get triggered.
load_dotenv(_REPO_ROOT / ".env")

sys.path.append(str(_REPO_ROOT / "qwen4b" / "scripts"))

from model_finder.models import ModelSpec  # noqa: E402
from model_finder.surprisal import load_model  # noqa: E402
from model_finder.utils import cleanup, get_device  # noqa: E402

MODEL_SPEC = ModelSpec(
    hf_name="google/gemma-3-27b-pt", approx_params="27B", dtype="bfloat16"
)

MODEL_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = MODEL_DIR / "logs"


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
    """Standard scaffolding every step uses."""
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
