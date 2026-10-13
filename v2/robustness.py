"""Post-hoc diagnostics of frozen results; no inference, fitting or test-layer search.

Run from repository root:
uv run --no-project --with numpy --with matplotlib --with markdown python v2/robustness.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "v2/results/robustness-20261008"


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected finite, nonempty observations")
    draws = np.random.default_rng(0).choice(a, (10000, len(a))).mean(axis=1)
    return dict(
        mean=float(a.mean()),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        n=len(a),
        positive=int((a > 0).sum()),
    )


def candidate_components(rows: list[dict], layer: int) -> list[dict]:
    groups = defaultdict(dict)
    for r in rows:
        if r["split"] == "test" and r["layer"] == layer and r["kind"] == "cross_condition":
            key = (r["recipient"], r["verb_class"], r["candidate"])
            if key in groups[r["family"]]:
                raise ValueError("Duplicate factorial cell")
            groups[r["family"]][key] = r["effect"]
    result = []
    for family, cells in groups.items():
        if len(cells) != 8:
            raise ValueError("Incomplete factorial cells")
        values = {}
        for candidate, candidate_sign in [("main", 1), ("object", -1)]:
            values[candidate] = (
                candidate_sign
                * sum(
                    direction_sign
                    * (
                        cells[(recipient, "optional_transitive", candidate)]
                        - cells[(recipient, "intransitive", candidate)]
                    )
                    for recipient, direction_sign in [("no_comma", 1), ("comma", -1)]
                )
                / 2
            )
        result.append(dict(family=family, **values, total=sum(values.values())))
    return result


def layer_stability(rows: list[dict]) -> dict:
    if not rows or any(r["split"] != "development" for r in rows):
        raise ValueError("Use development families only")
    families = sorted({r["family"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    lookup = {(r["family"], r["layer"]): r["value"] for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(families) * len(layers):
        raise ValueError("Expected a complete, unique family-by-layer grid")
    a = np.array([[lookup[f, layer] for layer in layers] for f in families])
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite layer values")
    samples = np.random.default_rng(0).integers(0, len(families), (10000, len(families)))
    choices = [layers[i] for i in a[samples].mean(axis=1).argmax(axis=1)]
    means = a.mean(axis=0)
    return dict(
        layers=layers,
        means=means.tolist(),
        bootstrap_selection_counts={str(k): v for k, v in sorted(Counter(choices).items())},
        within_95_percent_peak=[
            layer for layer, v in zip(layers, means, strict=True) if v >= 0.95 * means.max()
        ],
        leave_one_family_out={
            f: layers[np.delete(a, i, axis=0).mean(axis=0).argmax()] for i, f in enumerate(families)
        },
    )


def analyze() -> dict:
    paths = [ROOT / "v2/results/garden-followup-20261008-02/garden-followup.json"]
    paths += [
        ROOT / f"v1/gemma3_{size}/logs/11_patching.json" for size in ["1b", "4b", "12b", "27b"]
    ]
    garden, *legacy = [json.loads(p.read_text()) for p in paths]
    meta = json.loads((paths[0].parent / "metadata.json").read_text())
    if not meta["complete"]:
        raise ValueError("Incomplete source run")
    components = candidate_components(garden["interventions"], garden["selection"]["layer"])
    saved = {
        r["family"]: r["value"]
        for r in garden["family_interactions"]
        if r["split"] == "test" and r["kind"] == "cross_condition"
    }
    for row in components:
        if not np.isclose(row["total"], saved[row["family"]], atol=1e-9, rtol=0):
            raise ValueError("Component sum disagrees with saved interaction")
    baseline = defaultdict(dict)
    for r in garden["baselines"]:
        if r["split"] == "test":
            baseline[r["family"], r["verb_class"], r["condition"]][r["candidate"]] = r
    margins = []
    for (family, verb, condition), pair in sorted(baseline.items()):
        main, obj = pair["main"], pair["object"]
        margins.append(
            dict(
                family=family,
                verb_class=verb,
                condition=condition,
                main_tokens=main["completion_tokens"],
                object_tokens=obj["completion_tokens"],
                raw=main["log_probability"] - obj["log_probability"],
                per_token=main["log_probability"] / main["completion_tokens"]
                - obj["log_probability"] / obj["completion_tokens"],
            )
        )
    curves = []
    for size, source in zip(["1b", "4b", "12b", "27b"], legacy, strict=True):
        rows = source["results"]["per_layer"]
        peak = max(rows, key=lambda r: r["recovery_v_amb"])
        later = max((r for r in rows if r["layer"] >= 4), key=lambda r: r["recovery_v_amb"])
        curves.append(
            dict(
                model=size,
                layers=[r["layer"] for r in rows],
                recovery=[r["recovery_v_amb"] for r in rows],
                peak_layer=peak["layer"],
                peak=peak["recovery_v_amb"],
                later_layer=later["layer"],
                later_peak=later["recovery_v_amb"],
            )
        )
    return dict(
        status="post-hoc descriptive diagnostics; no new confirmation data",
        seed=0,
        resamples=10000,
        provenance={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        garden_components=components,
        garden_component_summary={
            k: summarize([r[k] for r in components]) for k in ["main", "object", "total"]
        },
        garden_layer_stability=layer_stability(
            garden["selection"]["development_family_interactions"]
        ),
        baseline_margin_sensitivity=margins,
        legacy_curves=curves,
    )


if __name__ == "__main__":
    result = analyze()
    OUTPUT.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Recomputed GPS-only robustness analysis")
