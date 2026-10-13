"""Pinned, auditable and bounded behavioral bridge run."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

MODEL = "google/gemma-3-1b-pt"
REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
SOURCES = (
    "v3/__init__.py",
    "v3/bridge/__init__.py",
    "v3/bridge/materials.py",
    "v3/bridge/measure.py",
    "v3/bridge/experiment.py",
    "v3/bridge/run.py",
    "v3/bridge/modal_app.py",
    "v3/bridge/PROTOCOL.md",
    "v2/__init__.py",
    "v2/experiment.py",
    "v2/materials.py",
    "v2/garden_followup.py",
)


def run(output: Path) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from v3.bridge.experiment import run_experiment
    from v3.bridge.materials import items
    from v3.bridge.measure import QuestionMeasurements

    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    root = Path(__file__).resolve().parents[2]
    sources = {name: (root / name).read_text() for name in SOURCES}
    meta = dict(
        model=MODEL,
        revision=REVISION,
        dtype="bfloat16",
        attention="eager",
        seed=0,
        fixed_layer=17,
        complete=False,
        python=platform.python_version(),
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        gpu=torch.cuda.get_device_name(),
        versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "numpy", "tokenizers", "huggingface_hub"]
        },
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES
        },
    )
    (output / "source.json").write_text(json.dumps(sources, indent=2))
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    (output / "materials.json").write_text(json.dumps(items(), indent=2))
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
        result = run_experiment(QuestionMeasurements(model, tokenizer), output)
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        meta["complete"] = True
    except Exception as error:
        meta["error_type"] = type(error).__name__
        meta["error"] = str(error)
        raise
    finally:
        meta["elapsed_seconds"] = time.time() - started
        (output / "metadata.json").write_text(json.dumps(meta, indent=2))
