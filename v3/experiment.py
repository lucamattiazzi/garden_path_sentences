"""Fixed-layer NP/Z assay; repeated items and random seeds remain within family."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.experiment import family_interval
from v2.garden_followup import matched_state

LAYER = 17
CONTRASTS = {
    "comma_native": ("native", "comma"),
    "comma_aligned": ("gap", "comma"),
    "position_only": ("native", "gap"),
}


def conditions(item: dict, verb_class: str, lexical: bool = False) -> dict:
    prefix = item["prefixes"][verb_class]
    if lexical:
        prefix = prefix.replace("the " + item["actor"] + " ", "the " + item["other_actor"] + " ", 1)
    head, noun = prefix.rsplit(" the ", 1)
    return dict(native=prefix, gap=prefix, comma=head + ", the " + noun)


def metrics(scores: dict, stage: str) -> dict:
    if stage == "next":
        return {key: scores[key] for key in ["margin", "was_logp", "comma_logp"]}
    return {
        **{f"word_{i}": value for i, value in enumerate(scores["word_logp"])},
        "three_words": sum(scores["token_logp"]),
    }


def summarize_effects(rows: list[dict]) -> list[dict]:
    cells = defaultdict(dict)
    for r in rows:
        for metric, effect in r["effects"].items():
            key = (
                r["split"],
                r["family"],
                r["id"],
                r["contrast"],
                r["stage"],
                r["kind"],
                r["seed"],
                metric,
            )
            cell = (r["verb_class"], r["direction"])
            if cell in cells[key]:
                raise ValueError("Duplicate intervention cell")
            cells[key][cell] = effect
    item_groups = defaultdict(list)
    for key, values in cells.items():
        if set(values) != {
            (v, d) for v in ["optional", "intransitive"] for d in ["forward", "reverse"]
        }:
            raise ValueError("Incomplete intervention cells")
        split, family, item, contrast, stage, kind, seed, metric = key
        forward = values["optional", "forward"] - values["intransitive", "forward"]
        reverse = -(values["optional", "reverse"] - values["intransitive", "reverse"])
        for direction, value in [
            ("forward", forward),
            ("reverse", reverse),
            ("average", (forward + reverse) / 2),
        ]:
            item_groups[split, family, item, contrast, stage, kind, metric, direction].append(value)
    grouped = defaultdict(list)
    for (
        split,
        family,
        _item,
        contrast,
        stage,
        kind,
        metric,
        direction,
    ), values in item_groups.items():
        grouped[split, contrast, stage, kind, metric, direction].append(
            (family, float(np.mean(values)))
        )
    # Paired control differences use the same source item and endpoint.
    for key, values in list(item_groups.items()):
        split, family, item, contrast, stage, kind, metric, direction = key
        if kind != "cross":
            continue
        for control in ["lexical", "random"]:
            comparator = item_groups[
                split, family, item, contrast, stage, control, metric, direction
            ]
            if not comparator:
                raise ValueError("Missing paired control")
            grouped[split, contrast, stage, f"cross_minus_{control}", metric, direction].append(
                (family, float(np.mean(values) - np.mean(comparator)))
            )
    return [
        dict(
            split=key[0],
            contrast=key[1],
            stage=key[2],
            kind=key[3],
            metric=key[4],
            direction=key[5],
            **family_interval(sorted(values)),
        )
        for key, values in sorted(grouped.items())
    ]


def run_experiment(measure, output: Path, items: list[dict]) -> dict:
    if len(measure.layers) <= LAYER + 1:
        raise ValueError("Fixed layer must not be the final block")
    baselines, interventions = [], []
    dtype = next(measure.model.parameters()).dtype
    with (output / "observations.jsonl").open("x") as raw:

        def record(row):
            raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()

        record(dict(record="frozen_selection", layer=LAYER, basis="V2; no V3 layer selection"))
        for split in ["development", "test"]:
            for item in [r for r in items if r["split"] == split]:
                cache = {}
                for verb in ["optional", "intransitive"]:
                    prefixes = conditions(item, verb)
                    measure.validate_alignment(prefixes["native"], prefixes["comma"])
                    lexical_prefixes = conditions(item, verb, lexical=True)
                    for stage, suffix in [("next", ""), ("trajectory", item["continuation"])]:
                        for condition, prefix in prefixes.items():
                            score, states, pos = measure.evaluate(
                                prefix, suffix, gap=condition == "gap", capture=True
                            )
                            _, lexical, _ = measure.evaluate(
                                lexical_prefixes[condition],
                                suffix,
                                gap=condition == "gap",
                                capture=True,
                            )
                            row = dict(
                                record="baseline",
                                id=item["id"],
                                family=item["family"],
                                split=split,
                                verb_class=verb,
                                condition=condition,
                                stage=stage,
                                prefix=prefix,
                                continuation=suffix,
                                scores=score,
                            )
                            baselines.append(row)
                            record(row)
                            cache[verb, stage, condition] = (
                                score,
                                states[LAYER],
                                pos,
                                lexical[LAYER],
                            )
                        for contrast, (low, high) in CONTRASTS.items():
                            for direction, recipient, donor in [
                                ("forward", low, high),
                                ("reverse", high, low),
                            ]:
                                base, target, pos, lexical = cache[verb, stage, recipient]
                                source = cache[verb, stage, donor][1]
                                norm = float(torch.linalg.vector_norm(source - target))
                                patches = [
                                    ("cross", None, source, {}),
                                    ("identity", None, target, {}),
                                ]
                                state, info = matched_state(target, lexical - target, norm, dtype)
                                patches.append(("lexical", None, state, info))
                                for seed in range(3):
                                    key = f"{item['id']}:{verb}:{contrast}:{direction}:{seed}"
                                    rng = np.random.default_rng(
                                        int.from_bytes(
                                            hashlib.sha256(key.encode()).digest()[:8], "big"
                                        )
                                    )
                                    vector = torch.from_numpy(
                                        rng.normal(size=target.shape).astype("float32")
                                    )
                                    state, info = matched_state(target, vector, norm, dtype)
                                    patches.append(("random", seed, state, info))
                                for kind, seed, state, info in patches:
                                    changed, _, _ = measure.evaluate(
                                        prefixes[recipient],
                                        suffix,
                                        gap=recipient == "gap",
                                        patch=(LAYER, pos, state),
                                    )
                                    effects = {
                                        k: value - metrics(base, stage)[k]
                                        for k, value in metrics(changed, stage).items()
                                    }
                                    if (
                                        kind == "identity"
                                        and max(abs(v) for v in effects.values()) > 1e-4
                                    ):
                                        raise ValueError("Identity patch failed")
                                    row = dict(
                                        record="intervention",
                                        id=item["id"],
                                        family=item["family"],
                                        split=split,
                                        verb_class=verb,
                                        contrast=contrast,
                                        stage=stage,
                                        direction=direction,
                                        recipient=recipient,
                                        donor=donor if kind == "cross" else None,
                                        kind=kind,
                                        seed=seed,
                                        layer=LAYER,
                                        position=pos,
                                        effects=effects,
                                        scores=changed,
                                        **info,
                                    )
                                    row["actual_l2"] = float(
                                        torch.linalg.vector_norm(
                                            state.to(dtype).float() - target.to(dtype).float()
                                        )
                                    )
                                    interventions.append(row)
                                    record(row)
                print(f"V3 {split}: {item['id']} / {item['family']}", flush=True)
    baseline_summaries = []
    for split in ["development", "test"]:
        for stage in ["next", "trajectory"]:
            selected = [r for r in baselines if r["split"] == split and r["stage"] == stage]
            names = list(metrics(selected[0]["scores"], stage))
            for contrast, (low, high) in CONTRASTS.items():
                for metric in names:
                    values = []
                    for item in [r for r in items if r["split"] == split]:
                        lookup = {
                            (r["verb_class"], r["condition"]): metrics(r["scores"], stage)[metric]
                            for r in selected
                            if r["id"] == item["id"]
                        }
                        value = (
                            lookup["optional", high]
                            - lookup["optional", low]
                            - lookup["intransitive", high]
                            + lookup["intransitive", low]
                        )
                        values.append((item["family"], value))
                    baseline_summaries.append(
                        dict(
                            split=split,
                            stage=stage,
                            contrast=contrast,
                            metric=metric,
                            **family_interval(sorted(values)),
                        )
                    )
    errors = [
        abs(r["actual_l2"] / r["requested_l2"] - 1)
        for r in interventions
        if r.get("requested_l2", 0) > 0
    ]
    return dict(
        materials=items,
        fixed_layer=LAYER,
        baseline_summaries=baseline_summaries,
        summaries=summarize_effects(interventions),
        baselines=baselines,
        numerical_checks=dict(
            maximum_identity_error=max(
                abs(v)
                for r in interventions
                if r["kind"] == "identity"
                for v in r["effects"].values()
            ),
            maximum_norm_mismatch=max(errors),
            median_norm_mismatch=float(np.median(errors)),
        ),
        interventions=len(interventions),
        status="exploratory published-material adaptation; no human validation of our exact assay",
    )
