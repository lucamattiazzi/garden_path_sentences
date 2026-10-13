"""Exploratory NP/Z factorial assay with development-only layer selection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.experiment import family_interval


def garden_materials() -> list[dict]:
    """Synthetic sentences, frozen before inference; no human validation."""
    families = [
        ("painted", "sneezed", "artist", "painter", "colorful canvas"),
        ("knitted", "snored", "tailor", "weaver", "woolen scarf"),
        ("embroidered", "sighed", "designer", "dressmaker", "linen napkin"),
        ("baked", "dozed", "baker", "cook", "birthday cake"),
        ("studied", "fainted", "student", "scholar", "thick textbook"),
        ("sewed", "blushed", "seamstress", "costumer", "silk dress"),
        ("cleaned", "yawned", "caretaker", "janitor", "kitchen window"),
        ("wrote", "hesitated", "poet", "novelist", "short note"),
        ("drew", "stumbled", "illustrator", "cartoonist", "paper portrait"),
        ("cooked", "slept", "chef", "caterer", "vegetable stew"),
        ("ironed", "shivered", "butler", "housekeeper", "cotton shirt"),
        ("ate", "dined", "traveler", "tourist", "buttered toast"),
    ]
    rows = []
    for index, (verb, control, actor, other, target) in enumerate(families):
        prefixes, lexical = {}, {}
        for verb_class, word in (("optional_transitive", verb), ("intransitive", control)):
            prefixes[verb_class], lexical[verb_class] = {}, {}
            for condition, punctuation in (("comma", ","), ("no_comma", "")):
                for output, person in ((prefixes, actor), (lexical, other)):
                    output[verb_class][condition] = (
                        f"While the {person} {word}{punctuation} the {target}"
                    )
        rows.append(
            {
                "id": f"npz-followup-{index:02d}",
                "family": verb,
                "split": "development" if index < 6 else "test",
                "actor": actor,
                "other_actor": other,
                "control_verb": control,
                "target": target,
                "prefixes": prefixes,
                "lexical_prefixes": lexical,
                "continuations": {
                    "main": "was beside the table.",
                    "object": "beside the table, the assistant waited.",
                },
            }
        )
    return rows


def factorial_interaction(values: dict) -> float:
    """Comma effect in optional-transitive minus intransitive contexts."""
    return float(
        values["optional_transitive"]["comma"]
        - values["optional_transitive"]["no_comma"]
        - values["intransitive"]["comma"]
        + values["intransitive"]["no_comma"]
    )


def matched_state(
    target: torch.Tensor, direction: torch.Tensor, requested_l2: float, dtype: torch.dtype
) -> tuple[torch.Tensor, dict]:
    """Match in FP32, then report the displacement actually injected after casting."""
    norm = float(torch.linalg.vector_norm(direction.float()))
    if not np.isfinite(requested_l2) or requested_l2 < 0 or not np.isfinite(norm):
        raise ValueError("Nonfinite or negative patch norm")
    if norm == 0 and requested_l2 > 0:
        raise ValueError("Cannot norm-match a zero control direction")
    scale = requested_l2 / norm if norm else 0.0
    rounded = (target.float() + direction.float() * scale).to(dtype).float()
    return rounded, {
        "requested_l2": requested_l2,
        "actual_l2": float(torch.linalg.vector_norm(rounded - target.to(dtype).float())),
        "direction_l2": norm,
        "scale": scale,
    }


def select_layer(rows: list[dict], num_layers: int) -> int:
    """Choose the largest signed mean interaction; deterministic lower-layer tie break."""
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "development" or not 0 <= row["layer"] < num_layers - 1:
            raise ValueError("Selection accepts development observations at non-final blocks only")
        if not np.isfinite(row["value"]):
            raise ValueError("Nonfinite selection observation")
        grouped[row["layer"]].append(row["value"])
    if not grouped:
        raise ValueError("No development observations")
    return max(sorted(grouped), key=lambda layer: float(np.mean(grouped[layer])))


def _intervention_interactions(rows: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for row in rows:
        key = (row["split"], row["family"], row["layer"], row["kind"], row["random_seed"])
        groups[key][(row["verb_class"], row["recipient"], row["candidate"])] = row["effect"]
    result = []
    for (split, family, layer, kind, seed), values in groups.items():
        directions = {}
        for recipient, sign in (("no_comma", 1), ("comma", -1)):
            effects = {}
            for verb_class in ("optional_transitive", "intransitive"):
                effects[verb_class] = sign * (
                    values[(verb_class, recipient, "main")]
                    - values[(verb_class, recipient, "object")]
                )
            directions[recipient] = effects["optional_transitive"] - effects["intransitive"]
        result.append(
            {
                "split": split,
                "family": family,
                "layer": layer,
                "kind": kind,
                "random_seed": seed,
                "to_no_comma": directions["no_comma"],
                "to_comma": directions["comma"],
                "value": float(np.mean(list(directions.values()))),
            }
        )
    return result


def run_garden_followup(measure, out: Path) -> dict:
    """Run a bounded assay; retain raw scores, fixed-test selection and numerical checks."""
    items = garden_materials()
    num_layers = len(measure.layers)
    if num_layers < 2:
        raise ValueError("At least two blocks are required to exclude the readout block")
    dtype = next(measure.model.parameters()).dtype
    baselines, interventions, prepared = [], [], {}
    out.mkdir(parents=True, exist_ok=True)
    raw = (out / "garden-followup-observations.jsonl").open("x")

    def record(row):
        raw.write(json.dumps(row, allow_nan=False) + "\n")
        raw.flush()

    def prepare(item):
        cache = {}
        for verb_class, conditions in item["prefixes"].items():
            for condition, prefix in conditions.items():
                for candidate, completion in item["continuations"].items():
                    states, pos = measure.capture(prefix, continuation=completion)
                    lexical_prefix = item["lexical_prefixes"][verb_class][condition]
                    lexical, _ = measure.capture(lexical_prefix, continuation=completion)
                    score = measure.score(prefix, completion)
                    critical_score = measure.score(prefix, "was") if candidate == "main" else None
                    count = (
                        measure.encode(prefix + " " + completion)["inputs"]["input_ids"].shape[1]
                        - measure.encode(prefix)["inputs"]["input_ids"].shape[1]
                    )
                    row = {
                        "record": "baseline",
                        "family": item["family"],
                        "split": item["split"],
                        "verb_class": verb_class,
                        "condition": condition,
                        "candidate": candidate,
                        "prefix": prefix,
                        "completion": completion,
                        "completion_tokens": count,
                        "log_probability": score,
                        "critical_word_log_probability": critical_score,
                    }
                    baselines.append(row)
                    record(row)
                    cache[(verb_class, condition, candidate)] = (states, pos, lexical, score)
        prepared[item["id"]] = cache

    def intervene(item, layers, controls, cross):
        cache = prepared[item["id"]]
        for verb_class, conditions in item["prefixes"].items():
            for recipient, donor in (("no_comma", "comma"), ("comma", "no_comma")):
                for candidate, completion in item["continuations"].items():
                    states, pos, lexical, baseline = cache[(verb_class, recipient, candidate)]
                    donor_states, donor_pos, _, _ = cache[(verb_class, donor, candidate)]
                    for layer in layers:
                        target, source = states[layer], donor_states[layer]
                        norm = float(torch.linalg.vector_norm(source - target))
                        patches = []
                        if cross:
                            patches.append(("cross_condition", None, source, {}))
                        if controls:
                            patches.append(("identity", None, target, {}))
                            value, info = matched_state(
                                target, lexical[layer] - target, norm, dtype
                            )
                            patches.append(("lexical", None, value, info))
                            for seed in range(3):
                                label = f"{item['id']}:{verb_class}:{recipient}:{layer}:{seed}"
                                stable = int.from_bytes(
                                    hashlib.sha256(label.encode()).digest()[:8], "big"
                                )
                                rng = np.random.default_rng(stable)
                                direction = torch.from_numpy(
                                    rng.normal(size=target.shape).astype("float32")
                                )
                                value, info = matched_state(target, direction, norm, dtype)
                                patches.append(("random", seed, value, info))
                        for kind, seed, value, info in patches:
                            score = measure.score(
                                conditions[recipient], completion, (layer, pos, value)
                            )
                            row = {
                                "record": "intervention",
                                "family": item["family"],
                                "split": item["split"],
                                "verb_class": verb_class,
                                "recipient": recipient,
                                "donor": donor if kind == "cross_condition" else None,
                                "layer": layer,
                                "position": pos,
                                "donor_position": donor_pos if kind == "cross_condition" else None,
                                "candidate": candidate,
                                "kind": kind,
                                "random_seed": seed,
                                "prefix": conditions[recipient],
                                "completion": completion,
                                "lexical_control_prefix": item["lexical_prefixes"][verb_class][
                                    recipient
                                ],
                                "baseline": baseline,
                                "patched": score,
                                "effect": score - baseline,
                                "actual_l2": float(
                                    torch.linalg.vector_norm(
                                        value.to(dtype).float() - target.to(dtype).float()
                                    )
                                ),
                                **info,
                            }
                            interventions.append(row)
                            record(row)
                            if kind == "identity" and abs(row["effect"]) > 1e-4:
                                raise ValueError(f"Identity check failed: {item['id']} L{layer}")

    try:
        for item in (r for r in items if r["split"] == "development"):
            prepare(item)
            intervene(item, range(num_layers - 1), controls=False, cross=True)
            print(f"Garden development: {item['family']}", flush=True)
        development = _intervention_interactions(interventions)
        selected = select_layer(development, num_layers)
        selection = {
            "layer": selected,
            "criterion": "maximum development mean signed bidirectional factorial interaction",
            "excluded_layers": [num_layers - 1],
            "test_layers_computed": [selected],
            "development_family_interactions": development,
        }
        record({"record": "selection_frozen_before_test", **selection})
        for item in items:
            test = item["split"] == "test"
            if test:
                prepare(item)
            intervene(item, [selected], controls=True, cross=test)
            print(f"Garden selected block {selected}: {item['split']} {item['family']}", flush=True)
    except Exception as error:
        record({"record": "aborted", "error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        raw.close()

    baseline_interactions = []
    for item in items:
        values = {verb_class: {} for verb_class in ("optional_transitive", "intransitive")}
        for verb_class in values:
            for condition in ("comma", "no_comma"):
                cache = prepared[item["id"]]
                values[verb_class][condition] = (
                    cache[(verb_class, condition, "main")][3]
                    - cache[(verb_class, condition, "object")][3]
                )
        baseline_interactions.append(
            {
                "family": item["family"],
                "split": item["split"],
                "margins": values,
                "value": factorial_interaction(values),
                "critical_word_interaction": factorial_interaction(
                    {
                        verb_class: {
                            condition: next(
                                r["critical_word_log_probability"]
                                for r in baselines
                                if r["family"] == item["family"]
                                and r["verb_class"] == verb_class
                                and r["condition"] == condition
                                and r["candidate"] == "main"
                            )
                            for condition in ("comma", "no_comma")
                        }
                        for verb_class in ("optional_transitive", "intransitive")
                    }
                ),
            }
        )
    interactions = _intervention_interactions(interventions)
    summaries = []
    for split in ("development", "test"):
        subset = [r for r in baseline_interactions if r["split"] == split]
        summaries.append(
            {
                "split": split,
                "kind": "behavioral_interaction",
                **family_interval([(r["family"], r["value"]) for r in subset]),
            }
        )
        summaries.append(
            {
                "split": split,
                "kind": "critical_word_interaction",
                **family_interval([(r["family"], r["critical_word_interaction"]) for r in subset]),
            }
        )
        for kind in ("cross_condition", "identity", "lexical", "random"):
            subset = [
                r
                for r in interactions
                if r["split"] == split and r["layer"] == selected and r["kind"] == kind
            ]
            for direction in ("value", "to_no_comma", "to_comma"):
                summaries.append(
                    {
                        "split": split,
                        "kind": kind,
                        "direction": direction,
                        "layer": selected,
                        **family_interval([(r["family"], r[direction]) for r in subset]),
                    }
                )
        selected_rows = [r for r in interactions if r["split"] == split and r["layer"] == selected]
        for control in ("lexical", "random"):
            for direction in ("value", "to_no_comma", "to_comma"):
                paired = []
                for family in {r["family"] for r in selected_rows}:
                    observed = next(
                        r[direction]
                        for r in selected_rows
                        if r["family"] == family and r["kind"] == "cross_condition"
                    )
                    controls = [
                        r[direction]
                        for r in selected_rows
                        if r["family"] == family and r["kind"] == control
                    ]
                    paired.append((family, float(observed - np.mean(controls))))
                summaries.append(
                    {
                        "split": split,
                        "kind": f"cross_minus_{control}",
                        "direction": direction,
                        "layer": selected,
                        **family_interval(sorted(paired)),
                    }
                )
    norm_errors = [
        abs(r["actual_l2"] / r["requested_l2"] - 1)
        for r in interventions
        if r.get("requested_l2", 0) > 0
    ]
    return {
        "materials": items,
        "selection": selection,
        "baselines": baselines,
        "baseline_interactions": baseline_interactions,
        "interventions": interventions,
        "family_interactions": interactions,
        "summaries": summaries,
        "exclusions": [],
        "numerical_checks": {
            "maximum_identity_error": max(
                abs(r["effect"]) for r in interventions if r["kind"] == "identity"
            ),
            "median_control_relative_norm_error_after_cast": float(np.median(norm_errors)),
            "maximum_control_relative_norm_error_after_cast": max(norm_errors),
        },
        "readout": "summed log probability of complete main versus object-compatible continuations",
        "status": "synthetic exploratory assay; no human linguistic validation",
    }
