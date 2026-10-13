"""Pinned development-only direct-answer run with source snapshots."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from v3.scale.run import MODELS, validate_loading

SOURCES = (
    "v3/__init__.py",
    "v3/bridge/__init__.py",
    "v3/bridge/materials.py",
    "v3/direct/__init__.py",
    "v3/direct/experiment.py",
    "v3/direct/measure.py",
    "v3/scale/__init__.py",
    "v3/scale/run.py",
    "v3/bridge12/__init__.py",
    "v3/bridge12/run.py",
    "v3/bridge12/modal_app.py",
    "v3/bridge12/PROTOCOL.md",
    "v3/bridge12/measure.py",
    "v3/bridge12/experiment.py",
    "v2/__init__.py",
    "v2/experiment.py",
    "v2/materials.py",
    "v2/garden_followup.py",
)


def run(output: Path) -> None:
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    from v3.bridge.materials import items
    from v3.bridge12.experiment import run_experiment
    from v3.bridge12.measure import BridgeMeasurements

    model_id, revision = MODELS["12b"]
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = {name: (root / name).read_text() for name in SOURCES}
    rows = items()
    (output / "source.json").write_text(json.dumps(source, indent=2))
    (output / "materials.json").write_text(json.dumps(rows, indent=2))
    started = time.time()
    meta = dict(
        complete=False,
        model=model_id,
        revision=revision,
        dtype="float32",
        tf32=False,
        attention="eager",
        seed=0,
        gpu=torch.cuda.get_device_name(),
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "tokenizers", "numpy"]
        },
        source_sha256={name: hashlib.sha256(s.encode()).hexdigest() for name, s in source.items()},
    )
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model, info = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        meta["loading_info"] = validate_loading(info)
        model = model.to("cuda").eval()
        meta["layers"] = len(model.model.language_model.layers)
        meta["parameter_count"] = sum(p.numel() for p in model.parameters())
        if meta["layers"] != 48:
            raise ValueError("Expected 48 decoder blocks")
        meta["fixed_layer"] = 32
        result = run_experiment(BridgeMeasurements(model, tokenizer), output)
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        meta["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        meta["complete"] = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        meta["error"] = str(error)
        raise
    finally:
        meta["elapsed_seconds"] = time.time() - started
        (output / "metadata.json").write_text(json.dumps(meta, indent=2))
