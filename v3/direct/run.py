"""Pinned development-only direct-answer run with source snapshots."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

MODEL = "google/gemma-3-1b-pt"
REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
SOURCES = (
    "v3/__init__.py",
    "v3/bridge/__init__.py",
    "v3/bridge/materials.py",
    "v3/direct/__init__.py",
    "v3/direct/experiment.py",
    "v3/direct/measure.py",
    "v3/direct/run.py",
    "v3/direct/modal_app.py",
    "v3/direct/PROTOCOL.md",
    "v2/__init__.py",
    "v2/experiment.py",
    "v2/materials.py",
)


def run(output: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from v3.direct.experiment import decisions, materials
    from v3.direct.measure import DirectMeasurements

    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = {name: (root / name).read_text() for name in SOURCES}
    rows = materials()
    (output / "source.json").write_text(json.dumps(source, indent=2))
    (output / "materials.json").write_text(json.dumps(rows, indent=2))
    started = time.time()
    meta = dict(
        complete=False,
        model=MODEL,
        revision=REVISION,
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
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
            )
            .to("cuda")
            .eval()
        )
        measure = DirectMeasurements(model, tokenizer)
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
        meta["complete"] = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        meta["error"] = str(error)
        raise
    finally:
        meta["elapsed_seconds"] = time.time() - started
        (output / "metadata.json").write_text(json.dumps(meta, indent=2))
