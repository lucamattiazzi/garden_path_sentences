"""Development-gated causal bridge at fixed relative layer 32 in Gemma 3 12B."""

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.garden_followup import matched_state
from v3.bridge.materials import items, sentence
from v3.direct.experiment import DEMONSTRATIONS

LAYER = 32
CONDITIONS = ("simple", "native", "gap", "clean")


def make_prompt(item, reversal, role, condition, lexical=False):
    text, site, gap = sentence(item, reversal, "passive", condition, lexical)
    first, second = item["participants"][:: 1 if reversal == 0 else -1]
    correct = second if role == "agent" else first
    before = DEMONSTRATIONS + "Sentence: "
    question = f"Who {item['event']} someone?" if role == "agent" else f"Who was {item['event']}?"
    return dict(
        text=before + text + f"\nQuestion: {question}\nAnswer: the",
        candidates=[" " + name + "." for name in item["participants"]],
        gold_index=item["participants"].index(correct),
        site_end=None if site is None else len(before) + site,
        gap_start=len(before) + gap if condition == "gap" else None,
        sentence_start=len(before),
    )


def competence(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if (
            row["record"] == "baseline"
            and row["split"] == "development"
            and row["condition"] in ["simple", "clean"]
        ):
            pairs[row["condition"], row["role"], row["family"]][row["reversal"]] = row["margin"]
    cells = defaultdict(list)
    for (condition, role, _), pair in pairs.items():
        if set(pair) != {0, 1}:
            raise ValueError("Incomplete gate reversal pair")
        cells[condition, role].append(min(pair.values()) > 0)
    rates = {f"{c}/{r}": float(np.mean(v)) for (c, r), v in sorted(cells.items())}
    return dict(passed=len(rates) == 4 and min(rates.values()) >= 0.75, both_reversal_rates=rates)


def run_experiment(measure, output: Path):
    if len(measure.layers) <= LAYER + 1:
        raise ValueError("Non-final intervention layer required")
    rows = []
    cache = {}
    dtype = next(measure.model.parameters()).dtype
    with (output / "observations.jsonl").open("x") as raw:

        def record(row):
            rows.append(row)
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()

        record(dict(record="frozen_protocol", layer=LAYER, gate_before_test=True, voice="passive"))
        for split in ["development", "test"]:
            selected = [item for item in items() if item["split"] == split]
            for item in selected:
                for reversal, role, condition in itertools.product(
                    [0, 1], ["agent", "patient"], CONDITIONS
                ):
                    p = make_prompt(item, reversal, role, condition)
                    score, state = measure.evaluate(
                        p, layer=LAYER, capture=condition in ["gap", "clean"]
                    )
                    gold = p["gold_index"]
                    row = dict(
                        record="baseline",
                        split=split,
                        family=item["family"],
                        id=item["id"],
                        reversal=reversal,
                        role=role,
                        condition=condition,
                        prompt=p,
                        margin=score["answer_logp"][gold] - score["answer_logp"][1 - gold],
                        scores=score,
                    )
                    record(row)
                    cache[item["id"], reversal, role, condition] = (row, state)
                print(f"Baseline {split}: {item['id']}", flush=True)
            if split == "development":
                gate = competence(rows)
                record(dict(record="development_gate", **gate))
                if not gate["passed"]:
                    break
            for item in selected:
                for reversal, role in itertools.product([0, 1], ["agent", "patient"]):
                    for recipient, donor, direction in [
                        ("gap", "clean", "clean_to_gap"),
                        ("clean", "gap", "gap_to_clean"),
                    ]:
                        base, target = cache[item["id"], reversal, role, recipient]
                        other, source = cache[item["id"], reversal, role, donor]
                        clean, gap = (other, base) if recipient == "gap" else (base, other)
                        comma = clean["prompt"]["text"].index(
                            ",", clean["prompt"]["sentence_start"]
                        )
                        offsets = measure.encode(clean["prompt"]["text"])["offsets"]
                        index = next(i for i, (a, b) in enumerate(offsets) if a <= comma < b)
                        for key in ["input_ids", "position_ids"]:
                            values = clean["scores"][key]
                            if values[:index] + values[index + 1 :] != gap["scores"][key]:
                                raise ValueError("Common tokens/positions do not align")
                        lexical = make_prompt(item, reversal, role, recipient, lexical=True)
                        _, lexstate = measure.evaluate(lexical, layer=LAYER, capture=True)
                        norm = float(torch.linalg.vector_norm(source - target))
                        vector, info = matched_state(target, lexstate - target, norm, dtype)
                        patches = [
                            ("cross", None, source, {"actual_l2": norm}),
                            ("identity", None, target, {"actual_l2": 0.0}),
                            ("lexical", None, vector, info),
                        ]
                        for seed in range(3):
                            key = f"bridge12:{item['id']}:{reversal}:{direction}:{seed}"
                            rng = np.random.default_rng(
                                int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                            )
                            vector = torch.from_numpy(
                                rng.normal(size=target.shape).astype("float32")
                            )
                            vector, info = matched_state(target, vector, norm, dtype)
                            patches.append(("random", seed, vector, info))
                        for kind, seed, vector, info in patches:
                            score, _ = measure.evaluate(base["prompt"], layer=LAYER, donor=vector)
                            gold = base["prompt"]["gold_index"]
                            margin = score["answer_logp"][gold] - score["answer_logp"][1 - gold]
                            qa = margin - base["margin"]
                            nxt = score["next_margin"] - base["scores"]["next_margin"]
                            if kind == "identity" and max(abs(qa), abs(nxt)) > 1e-4:
                                raise ValueError("Identity patch failed")
                            record(
                                dict(
                                    record="intervention",
                                    split=split,
                                    family=item["family"],
                                    id=item["id"],
                                    reversal=reversal,
                                    role=role,
                                    condition=recipient,
                                    direction=direction,
                                    kind=kind,
                                    seed=seed,
                                    margin=margin,
                                    qa_effect=qa,
                                    next_effect=nxt,
                                    scores=score,
                                    **info,
                                )
                            )
                print(f"Patches {split}: {item['id']}", flush=True)
    return dict(
        status="completed_bridge" if gate["passed"] else "instrument_gate_failed",
        gate=gate,
        baselines=sum(r["record"] == "baseline" for r in rows),
        interventions=sum(r["record"] == "intervention" for r in rows),
        maximum_identity_error=max(
            (
                max(abs(r["qa_effect"]), abs(r["next_effect"]))
                for r in rows
                if r.get("kind") == "identity"
            ),
            default=None,
        ),
    )
