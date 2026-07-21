"""Shared scaffolding for the inverse_gps per-step scripts.

Mirror of qwen4b/scripts/_common.py, except the model is not fixed: every
step accepts `--model HF_NAME` (default Qwen/Qwen3-4B-Base, the smallest
model with a full main-pipeline run to compare against). Gemma models are
loaded in bfloat16 automatically, matching model_finder/models.py.

Log/JSON outputs land in inverse_gps/logs/<step>__<model-slug>.{log,json}
so sweeps over several models don't clobber each other.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Load HF_TOKEN etc. BEFORE torch / transformers import-paths get triggered.
load_dotenv(_REPO_ROOT / ".env")

sys.path.append(str(_REPO_ROOT))

from model_finder.models import ModelSpec  # noqa: E402
from model_finder.surprisal import load_model  # noqa: E402
from model_finder.utils import cleanup, get_device  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"

LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"


def parse_model_arg() -> ModelSpec:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    args, _ = parser.parse_known_args()
    dtype = "bfloat16" if "gemma" in args.model.lower() else "float16"
    return ModelSpec(hf_name=args.model, approx_params="?", dtype=dtype)


def _slug(hf_name: str) -> str:
    return hf_name.split("/")[-1].replace(".", "_")


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


def run_step(
    step_name: str,
    body: Callable[[object, object, str], dict],
    spec: ModelSpec | None = None,
) -> None:
    """Standard scaffolding every step uses (tee + json, per-model files)."""
    if spec is None:
        spec = parse_model_arg()
    stem = f"{step_name}__{_slug(spec.hf_name)}"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LOGS_DIR / f"{stem}.log", "w")  # noqa: SIM115
    sys.stdout = _Tee(sys.__stdout__, f)

    print(f"=== {step_name} ===")
    print(f"device = {get_device()}")
    print(f"model  = {spec.hf_name} (dtype={spec.dtype})")
    print()

    device = get_device()
    model, tokenizer = load_model(device, spec)
    t0 = time.time()
    try:
        results = body(model, tokenizer, device)
        out = LOGS_DIR / f"{stem}.json"
        with open(out, "w") as fh:
            json.dump(
                {"step": step_name, "model": spec.hf_name, "results": results},
                fh,
                indent=2,
            )
        print(f"\nSaved → {out}")
    finally:
        del model, tokenizer
        cleanup(device)
        print(f"\nElapsed: {time.time() - t0:.1f}s")
