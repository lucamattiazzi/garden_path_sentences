"""Pinned development-only direct-answer run with source snapshots."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

MODELS = {
    "4b": ("google/gemma-3-4b-pt", "cc012e0a6d0787b4adcc0fa2c4da74402494554d"),
    "12b": ("google/gemma-3-12b-pt", "295efb63d01a7017928f273a94ebb86105c9526f"),
}
SOURCES = (
    "v3/__init__.py",
    "v3/bridge/__init__.py",
    "v3/bridge/materials.py",
    "v3/direct/__init__.py",
    "v3/direct/experiment.py",
    "v3/direct/measure.py",
    "v3/scale/__init__.py",
    "v3/scale/run.py",
    "v3/scale/measure.py",
    "v3/scale/modal_app.py",
    "v3/scale/PROTOCOL.md",
    "v2/__init__.py",
    "v2/experiment.py",
    "v2/materials.py",
)


def next_size(outcomes: list[bool]) -> str | None:
    if any(outcomes) or len(outcomes) >= 2:
        return None
    return ["4b", "12b"][len(outcomes)]


def validate_loading(info: dict) -> dict:
    if any(
        info.get(k) for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    ):
        raise ValueError(f"Checkpoint did not load exactly: {info}")
    return {key: list(value) for key, value in info.items()}


def run(output: Path, size: str) -> None:
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    from v3.direct.experiment import decisions, materials
    from v3.scale.measure import ScaleMeasurements

    model_id, revision = MODELS[size]
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = {name: (root / name).read_text() for name in SOURCES}
    rows = materials()
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
        measure = ScaleMeasurements(model, tokenizer)
        measured = []
        with (output / "observations.jsonl").open("x") as raw:
            for row in rows:
                score = measure.score(row["prompt"], row["candidates"])
                values = [c["logp"] for c in score["candidates"]]
                gold = row["gold_index"]
                row = dict(row, scores=score, margin=values[gold] - values[1 - gold])
                raw.write(json.dumps(row, allow_nan=False) + "\n")
                raw.flush()
                measured.append(row)
        decision = decisions(measured)
        result = dict(
            status="instrument_gate_passed" if decision["passed"] else "instrument_gate_failed",
            gate=decision,
            baselines=len(measured),
            interventions=0,
            test_inference=False,
        )
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
