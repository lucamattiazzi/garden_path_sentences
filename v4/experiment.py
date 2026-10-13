"""Fresh confirmation only after every development construction/role passes."""

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.garden_followup import matched_state
from v4.materials import CONSTRUCTIONS, items, make_prompt


def components(agent_first_effect: float, patient_first_effect: float) -> dict:
    return dict(
        common_first=(agent_first_effect + patient_first_effect) / 2,
        question_difference=(agent_first_effect - patient_first_effect) / 2,
    )


def competence(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        key = (row["construction"], row["role"], row["family"])
        if row["reversal"] in pairs[key]:
            raise ValueError("Duplicate reversal")
        pairs[key][row["reversal"]] = row["margin"]
    cells = defaultdict(list)
    for (construction, role, _), pair in pairs.items():
        if set(pair) != {0, 1}:
            raise ValueError("Incomplete reversal pair")
        cells[construction, role].append(min(pair.values()) > 0)
    rates = {c + "/" + r: float(np.mean(v)) for (c, r), v in sorted(cells.items())}
    return dict(passed=len(rates) == 8 and min(rates.values()) >= 0.75, both_reversal_rates=rates)


def run_experiment(measure, output: Path, layer: int):
    rows = []
    cache = {}
    dtype = next(measure.model.parameters()).dtype
    with (output / "observations.jsonl").open("x") as raw:

        def record(row):
            rows.append(row)
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()

        record(dict(record="frozen_protocol", layer=layer, gate_before_confirmation=True))
        for split in ["development", "confirmation"]:
            selected = [r for r in items() if r["split"] == split]
            for item in selected:
                for construction, reversal, role, condition in itertools.product(
                    CONSTRUCTIONS,
                    [0, 1],
                    ["agent", "patient"],
                    ["clean"] if split == "development" else ["clean", "gap"],
                ):
                    p = make_prompt(item, reversal, construction, role, condition)
                    score, state = measure.evaluate(p, layer=layer, capture=split == "confirmation")
                    gold = p["gold_index"]
                    row = dict(
                        record="baseline",
                        split=split,
                        id=item["id"],
                        family=item["family"],
                        construction=construction,
                        reversal=reversal,
                        role=role,
                        condition=condition,
                        prompt=p,
                        margin=score["answer_logp"][gold] - score["answer_logp"][1 - gold],
                        scores=score,
                    )
                    record(row)
                    if split == "confirmation":
                        cache[item["id"], construction, reversal, role, condition] = (row, state)
                print(f"Baseline {split}: {item['id']}", flush=True)
            if split == "development":
                gate = competence([r for r in rows if r["record"] == "baseline"])
                record(dict(record="development_gate", **gate))
                if not gate["passed"]:
                    break
                continue
            for item in selected:
                for construction, reversal, role in itertools.product(
                    CONSTRUCTIONS, [0, 1], ["agent", "patient"]
                ):
                    for recipient, donor, direction in [
                        ("gap", "clean", "clean_to_gap"),
                        ("clean", "gap", "gap_to_clean"),
                    ]:
                        base, target = cache[item["id"], construction, reversal, role, recipient]
                        other, source = cache[item["id"], construction, reversal, role, donor]
                        clean, gap = (other, base) if recipient == "gap" else (base, other)
                        comma = clean["prompt"]["text"].index(
                            ",", clean["prompt"]["sentence_start"]
                        )
                        offsets = measure.encode(clean["prompt"]["text"])["offsets"]
                        index = next(i for i, (a, b) in enumerate(offsets) if a <= comma < b)
                        for field in ["input_ids", "position_ids"]:
                            values = clean["scores"][field]
                            if values[:index] + values[index + 1 :] != gap["scores"][field]:
                                raise ValueError("Positional alignment failed")
                        lexical = make_prompt(
                            item, reversal, construction, role, recipient, lexical=True
                        )
                        _, lexstate = measure.evaluate(lexical, layer=layer, capture=True)
                        norm = float(torch.linalg.vector_norm(source - target))
                        vector, info = matched_state(target, lexstate - target, norm, dtype)
                        patches = [
                            ("cross", source, dict(actual_l2=norm)),
                            ("identity", target, dict(actual_l2=0.0)),
                            ("lexical", vector, info),
                        ]
                        seed = int.from_bytes(
                            hashlib.sha256(
                                f"v4:{item['id']}:{reversal}:{direction}".encode()
                            ).digest()[:8],
                            "big",
                        )
                        random = torch.from_numpy(
                            np.random.default_rng(seed).normal(size=target.shape).astype("float32")
                        )
                        vector, info = matched_state(target, random, norm, dtype)
                        patches.append(("random", vector, dict(info, seed=seed)))
                        for kind, vector, info in patches:
                            score, _ = measure.evaluate(base["prompt"], layer=layer, donor=vector)
                            gold = base["prompt"]["gold_index"]
                            first = base["prompt"]["first_index"]
                            margin = score["answer_logp"][gold] - score["answer_logp"][1 - gold]
                            qa = margin - base["margin"]
                            first_effect = (
                                score["answer_logp"][first] - score["answer_logp"][1 - first]
                            ) - (
                                base["scores"]["answer_logp"][first]
                                - base["scores"]["answer_logp"][1 - first]
                            )
                            nxt = score["next_margin"] - base["scores"]["next_margin"]
                            if (
                                kind == "identity"
                                and max(abs(qa), abs(first_effect), abs(nxt)) > 1e-4
                            ):
                                raise ValueError("Identity patch failed")
                            record(
                                dict(
                                    record="intervention",
                                    split=split,
                                    id=item["id"],
                                    family=item["family"],
                                    construction=construction,
                                    reversal=reversal,
                                    role=role,
                                    condition=recipient,
                                    direction=direction,
                                    kind=kind,
                                    margin=margin,
                                    qa_effect=qa,
                                    first_effect=first_effect,
                                    next_effect=nxt,
                                    scores=score,
                                    **info,
                                )
                            )
                print(f"Patches confirmation: {item['id']}", flush=True)
    return dict(
        status="completed_confirmation" if gate["passed"] else "instrument_gate_failed",
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
