"""Pinned-checkpoint orchestration for the garden-path follow-up."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

MODEL = "google/gemma-3-1b-pt"
REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
SOURCE_FILES = (
    "__init__.py",
    "materials.py",
    "experiment.py",
    "followup_run.py",
    "followup_modal.py",
    "garden_followup.py",
    "GARDEN_PROTOCOL.md",
)


def source_bundle(root: Path, names: tuple[str, ...]) -> tuple[dict, dict]:
    snapshot = {name: (root / name).read_text() for name in names}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    return snapshot, hashes


def start_run(output: Path, track: str) -> None:
    if track != "garden":
        raise ValueError("Only the garden track is in scope")
    output.mkdir(parents=True, exist_ok=False)


def run_followup(output: Path, track: str) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from v2.experiment import Measurements

    start_run(output, track)
    start = time.time()
    root = Path(__file__).parent
    snapshot, hashes = source_bundle(root, SOURCE_FILES)
    metadata = {
        "track": track,
        "model": MODEL,
        "revision": REVISION,
        "seed": 0,
        "dtype": "bfloat16",
        "attention": "eager",
        "layer_index": "zero-based, post-block",
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": hashes,
        "complete": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "huggingface_hub", "tokenizers")
        },
    }
    (output / "source.json").write_text(json.dumps(snapshot, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .to("cuda")
            .eval()
        )
        measure = Measurements(model, tokenizer)
        from v2.garden_followup import run_garden_followup

        result = run_garden_followup(measure, output)
        (output / f"{track}-followup.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        metadata["complete"] = True
    except Exception as error:
        metadata["error_type"] = type(error).__name__
        raise
    finally:
        metadata["elapsed_seconds"] = time.time() - start
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
