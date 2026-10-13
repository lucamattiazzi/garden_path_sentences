"""Gate the QA instrument on development before inspecting test or patching."""

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.experiment import family_interval
from v2.garden_followup import matched_state
from v3.bridge.materials import items, prompt

LAYER = 17
CONDITIONS = ("simple", "no_context", "native", "gap", "clean")


def behavioral_summary(rows: list[dict]) -> list[dict]:
    pairs = defaultdict(dict)
    for r in rows:
        key = tuple(
            r[k] for k in ["split", "condition", "voice", "family", "id", "reversal", "role"]
        )
        if r["order"] in pairs[key]:
            raise ValueError("Duplicate answer order")
        pairs[key][r["order"]] = r["correct_margin"]
    grouped = defaultdict(list)
    for (split, condition, voice, family, _item, _reversal, _role), orders in pairs.items():
        if set(orders) != {0, 1}:
            raise ValueError("Both answer orders required")
        margins = list(orders.values())
        measures = {
            "margin": float(np.mean(margins)),
            "accuracy": float(np.mean([v > 0 for v in margins])),
            "both_orders": float(all(v > 0 for v in margins)),
        }
        for metric, value in measures.items():
            for domain in [voice, "all"]:
                grouped[split, condition, domain, metric].append((family, value))
    return [
        dict(split=k[0], condition=k[1], voice=k[2], metric=k[3], **family_interval(sorted(v)))
        for k, v in sorted(grouped.items())
    ]


def gate(rows: list[dict]) -> dict:
    summary = behavioral_summary(rows)
    rates = {
        voice: next(
            r["mean"]
            for r in summary
            if (r["split"], r["condition"], r["voice"], r["metric"])
            == ("development", "simple", voice, "both_orders")
        )
        for voice in ["active", "passive"]
    }
    mass = float(
        np.median(
            [
                r["answer_mass"]
                for r in rows
                if r["split"] == "development" and r["condition"] == "simple"
            ]
        )
    )
    return dict(
        passed=min(rates.values()) >= 0.75 and mass >= 0.10,
        both_orders_correct_by_voice=rates,
        median_answer_mass=mass,
        rule=">=75% both-order correct per voice on simple development; median A+B probability >=10%",
    )


def patch_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if r["record"] != "intervention":
            continue
        for metric in ["qa_effect", "next_effect"]:
            sign = 1 if r["direction"] == "clean_to_gap" else -1
            key = (
                r["split"],
                r["family"],
                r["id"],
                r["reversal"],
                r["role"],
                r["order"],
                r["direction"],
                r["kind"],
                metric,
            )
            groups[key].append(sign * r[metric])
    combined = defaultdict(list)
    for key, values in groups.items():
        split, family, _item, _reversal, _role, _order, direction, kind, metric = key
        expected = 3 if kind == "random" else 1
        if len(values) != expected:
            raise ValueError("Unexpected seed count")
        value = float(np.mean(values))
        combined[split, direction, kind, metric].append((family, value))
        combined[split, "average", kind, metric].append((family, value))
        if kind == "cross":
            for control in ["lexical", "random"]:
                control_key = (*key[:-2], control, metric)
                if control_key not in groups:
                    raise ValueError("Missing paired control")
                delta = value - float(np.mean(groups[control_key]))
                combined[split, direction, "cross_minus_" + control, metric].append((family, delta))
                combined[split, "average", "cross_minus_" + control, metric].append((family, delta))
    return [
        dict(split=k[0], direction=k[1], kind=k[2], metric=k[3], **family_interval(sorted(v)))
        for k, v in sorted(combined.items())
    ]


def run_experiment(measure, output: Path) -> dict:
    material = items()
    rows = []
    cache = {}
    if len(measure.layers) <= LAYER + 1:
        raise ValueError("Fixed layer must be non-final")
    dtype = next(measure.model.parameters()).dtype
    with (output / "observations.jsonl").open("x") as raw:

        def record(row):
            rows.append(row)
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()

        record(
            dict(
                record="frozen_protocol", layer=LAYER, patch_voice="passive", gate_before_test=True
            )
        )
        decision = None
        for split in ["development", "test"]:
            for item in [r for r in material if r["split"] == split]:
                for reversal, voice, role, order in itertools.product(
                    [0, 1], ["active", "passive"], ["agent", "patient"], [0, 1]
                ):
                    for condition in CONDITIONS:
                        p = prompt(item, reversal, voice, condition, role, order)
                        capture = voice == "passive" and condition in ["gap", "clean"]
                        score, state, site = measure.evaluate(
                            p["text"], p["site_end"], p["gap_start"], layer=LAYER, capture=capture
                        )
                        gold = p["gold_index"]
                        margin = score["answer_logp"][gold] - score["answer_logp"][1 - gold]
                        row = dict(
                            record="baseline",
                            id=item["id"],
                            family=item["family"],
                            split=split,
                            reversal=reversal,
                            voice=voice,
                            condition=condition,
                            role=role,
                            order=order,
                            gold_index=gold,
                            correct=p["correct"],
                            correct_margin=margin,
                            answer_mass=score["answer_mass"],
                            site=site,
                            site_end=p["site_end"],
                            gap_start=p["gap_start"],
                            prompt=p["text"],
                            scores=score,
                        )
                        record(row)
                        if capture:
                            cache[item["id"], reversal, role, order, condition] = (row, state)
                print(f"Bridge baseline {split}: {item['id']}", flush=True)
            if split == "development":
                decision = gate([r for r in rows if r["record"] == "baseline"])
                record(dict(record="development_gate", **decision))
                if not decision["passed"]:
                    print(
                        "Development QA instrument failed: test and causal patches remain unrun.",
                        flush=True,
                    )
                    break
            for item in [r for r in material if r["split"] == split]:
                for reversal, role, order in itertools.product(
                    [0, 1], ["agent", "patient"], [0, 1]
                ):
                    for recipient, donor, direction in [
                        ("gap", "clean", "clean_to_gap"),
                        ("clean", "gap", "gap_to_clean"),
                    ]:
                        base, target = cache[item["id"], reversal, role, order, recipient]
                        donor_row, source = cache[item["id"], reversal, role, order, donor]
                        # Common tokens and their RoPE positions must align once the comma is removed.
                        small, large = (
                            (base, donor_row) if recipient == "gap" else (donor_row, base)
                        )
                        comma_char = large["prompt"].find(
                            ",", large["prompt"].index("Sentence: ", large["prompt"].rfind("\n\n"))
                        )
                        encoded = measure.encode(large["prompt"])
                        comma_index = next(
                            i for i, (a, b) in enumerate(encoded["offsets"]) if a <= comma_char < b
                        )
                        for field in ["input_ids", "position_ids"]:
                            values = large["scores"][field]
                            if (
                                values[:comma_index] + values[comma_index + 1 :]
                                != small["scores"][field]
                            ):
                                raise ValueError("Question prompt positional alignment failed")
                        lexical = prompt(
                            item, reversal, "passive", recipient, role, order, lexical=True
                        )
                        _, lexical_state, _ = measure.evaluate(
                            lexical["text"],
                            lexical["site_end"],
                            lexical["gap_start"],
                            layer=LAYER,
                            capture=True,
                        )
                        norm = float(torch.linalg.vector_norm(source - target))
                        vector, info = matched_state(target, lexical_state - target, norm, dtype)
                        patches = [
                            ("cross", None, source, {}),
                            ("identity", None, target, {}),
                            ("lexical", None, vector, info),
                        ]
                        for seed in range(3):
                            # Same random vector across question roles and answer orders.
                            key = f"bridge:{item['id']}:{reversal}:{direction}:{seed}"
                            rng = np.random.default_rng(
                                int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                            )
                            vector = torch.from_numpy(
                                rng.normal(size=target.shape).astype("float32")
                            )
                            vector, info = matched_state(target, vector, norm, dtype)
                            patches.append(("random", seed, vector, info))
                        for kind, seed, vector, info in patches:
                            score, _, _ = measure.evaluate(
                                base["prompt"],
                                base["site_end"],
                                base["gap_start"],
                                layer=LAYER,
                                patch=vector,
                            )
                            gold = base["gold_index"]
                            margin = score["answer_logp"][gold] - score["answer_logp"][1 - gold]
                            qa = margin - base["correct_margin"]
                            next_effect = score["next_margin"] - base["scores"]["next_margin"]
                            if kind == "identity" and max(abs(qa), abs(next_effect)) > 1e-4:
                                raise ValueError("Identity patch failed")
                            row = dict(
                                record="intervention",
                                id=item["id"],
                                family=item["family"],
                                split=split,
                                reversal=reversal,
                                voice="passive",
                                condition=recipient,
                                role=role,
                                order=order,
                                direction=direction,
                                kind=kind,
                                seed=seed,
                                qa_effect=qa,
                                next_effect=next_effect,
                                correct_margin=margin,
                                scores=score,
                                **info,
                            )
                            row["actual_l2"] = float(
                                torch.linalg.vector_norm(
                                    vector.to(dtype).float() - target.to(dtype).float()
                                )
                            )
                            record(row)
                print(f"Bridge patch {split}: {item['id']}", flush=True)
    baseline = [r for r in rows if r["record"] == "baseline"]
    patches = [r for r in rows if r["record"] == "intervention"]
    return dict(
        status="completed_bridge" if decision["passed"] else "instrument_gate_failed",
        gate=decision,
        materials=material,
        baseline_count=len(baseline),
        intervention_count=len(patches),
        behavioral=behavioral_summary(baseline),
        patches=patch_summary(patches),
        maximum_identity_error=max(
            (
                max(abs(r["qa_effect"]), abs(r["next_effect"]))
                for r in patches
                if r["kind"] == "identity"
            ),
            default=None,
        ),
    )
