"""Post-hoc numerical diagnostic on development only, without causal patches."""

import hashlib
import json
import time
from pathlib import Path

from v3.bridge.run import MODEL, REVISION, SOURCES

EXTRA_SOURCES = ("v3/bridge/precision.py", "v3/bridge/precision_app.py", "v3/bridge/PRECISION.md")


def select_rows(rows):
    return [
        r
        for r in rows
        if r.get("record") == "baseline"
        and r["split"] == "development"
        and r["condition"] in ["simple", "no_context"]
    ]


def run(parent: Path, output: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from v3.bridge.experiment import behavioral_summary, gate
    from v3.bridge.measure import QuestionMeasurements

    output.mkdir(exist_ok=False, parents=True)
    root = Path(__file__).resolve().parents[2]
    snapshot = {name: (root / name).read_text() for name in (*SOURCES, *EXTRA_SOURCES)}
    (output / "source.json").write_text(json.dumps(snapshot, indent=2))
    parent_bytes = (parent / "observations.jsonl").read_bytes()
    selected = select_rows([json.loads(line) for line in parent_bytes.splitlines()])
    if len(selected) != 192:
        raise ValueError("Expected 192 development-only baseline rows")
    started = time.time()
    meta = dict(
        complete=False,
        purpose="post-hoc development-only numerical diagnostic",
        dtype="float32",
        model=MODEL,
        revision=REVISION,
        parent_run=parent.name,
        parent_observations_sha256=hashlib.sha256(parent_bytes).hexdigest(),
        source_sha256={
            name: hashlib.sha256(s.encode()).hexdigest() for name, s in snapshot.items()
        },
        gpu=torch.cuda.get_device_name(),
        tf32=False,
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
        measure = QuestionMeasurements(model, tokenizer)
        rows = []
        with (output / "observations.jsonl").open("x") as raw:
            for original in selected:
                score, _, _ = measure.evaluate(original["prompt"], None, None)
                if score["input_ids"] != original["scores"]["input_ids"]:
                    raise ValueError("Prompt tokenization changed")
                gold = original["gold_index"]
                row = dict(
                    original,
                    scores=score,
                    answer_mass=score["answer_mass"],
                    correct_margin=score["answer_logp"][gold] - score["answer_logp"][1 - gold],
                )
                raw.write(json.dumps(row, allow_nan=False) + "\n")
                raw.flush()
                rows.append(row)
        result = dict(
            gate=gate(rows),
            behavioral=behavioral_summary(rows),
            baseline_count=len(rows),
            interventions=0,
            test_inference=False,
            note="Diagnostic only; does not reopen the failed original experiment.",
        )
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        meta["complete"] = True
    finally:
        meta["elapsed_seconds"] = time.time() - started
        (output / "metadata.json").write_text(json.dumps(meta, indent=2))
