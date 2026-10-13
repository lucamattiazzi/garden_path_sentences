"""Audit saved logs without loading models or rewriting historical results.

Run from the repository root: python -m research.audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen4b", "qwen14b", "qwen32b", "gemma3_1b", "gemma3_4b", "gemma3_12b", "gemma3_27b")


def summarize_items(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Fixed-seed item bootstrap and leave-one-out sensitivity; no silent exclusions."""
    values = np.array([r[key] for r in rows], dtype=float)
    if not len(rows) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty, finite paired observations")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate item IDs")
    means = np.random.default_rng(0).choice(values, (10_000, len(values))).mean(axis=1)
    return {
        "n": len(rows),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": np.percentile(means, [2.5, 97.5]).tolist(),
        "n_positive": int((values > 0).sum()),
        "leave_one_out": [
            {"omitted_id": r["id"], "mean": float(np.delete(values, i).mean())}
            for i, r in enumerate(rows)
            if len(rows) > 1
        ],
    }


def paired_change(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    """Second minus first on identical item IDs, regardless of input ordering."""
    left, right = ({r["id"]: r["vc_delta"] for r in rows} for rows in (first, second))
    if left.keys() != right.keys() or len(left) != len(first) or len(right) != len(second):
        raise ValueError("Paired comparison requires identical, unique item IDs")
    return summarize_items(
        [{"id": i, "change": right[i] - left[i]} for i in sorted(left)], "change"
    )


def nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite_paths(v, f"{prefix}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in nonfinite_paths(v, f"{prefix}[{i}]")]
    return [prefix] if isinstance(value, float) and not math.isfinite(value) else []


def read_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())["results"]


def audit_model(logs: Path) -> dict[str, Any]:
    steps = {p.stem: read_results(p) for p in sorted(logs.glob("*.json"))}
    result: dict[str, Any] = {
        "nonfinite": [
            {"file": f"{step}.json", "path": p}
            for step, data in steps.items()
            for p in nonfinite_paths(data)
        ],
        "baseline_changes": [],
        "baseline_comparison_available": "01_baseline" in steps and "02_layer_sweep" in steps,
    }
    baseline = {r["id"]: r for r in steps.get("01_baseline", {}).get("pair_results", [])}
    sweep = {r["id"]: r for r in steps.get("02_layer_sweep", {}).get("baseline_pair_results", [])}
    for item_id in sorted(baseline.keys() | sweep.keys()):
        if not result["baseline_comparison_available"]:
            break
        if item_id not in baseline or item_id not in sweep:
            result["baseline_changes"].append(
                {"id": item_id, "reason": "missing from one baseline"}
            )
            continue
        fields = []
        for key in ("surprisal_gps", "surprisal_normal", "delta"):
            a, b = baseline[item_id][key], sweep[item_id][key]
            if not (math.isfinite(a) and math.isfinite(b)) or not math.isclose(a, b, abs_tol=1e-4):
                fields.append(key)
        if fields:
            result["baseline_changes"].append({"id": item_id, "fields": fields})

    if "09_novel_predict" in steps:
        result["continuation"] = {
            condition: summarize_items(
                steps["09_novel_predict"][condition]["pair_results"], "vc_delta"
            )
            for condition in ("canonical", "novel")
        }
    if "13_null_distribution" in steps:
        samples = steps["13_null_distribution"]["samples"]
        lo = min(r["nv_effect_drop"] for r in samples)
        hi = max(r["nv_effect_drop"] for r in samples)
        result["nv_random_reference"] = {
            "n": len(samples),
            "min": lo,
            "max": hi,
            "heads_outside_sampled_range": [
                {
                    "layer": data["layer"],
                    "head": h["head"],
                    "nv_change": h["per_template_drop"]["NV"],
                    "control_inflation": h["control_inflation"],
                }
                for step, data in steps.items()
                if step.startswith("03_head_sweep_")
                for h in data["per_head"]
                if not lo <= h["per_template_drop"]["NV"] <= hi
            ],
        }
    if "14_null_distribution_vc" in steps:
        heads = [
            h
            for step, data in steps.items()
            if step.startswith("06_predict_head_sweep_")
            for h in data["per_head"]
        ]
        reference = steps["14_null_distribution_vc"]
        threshold = reference["max_of_k"].get(str(len(heads)))
        if heads and threshold:
            best = max(h["vc_gps_inflation"] for h in heads)
            result["continuation_random_reference"] = {
                "k": len(heads),
                "best_vc_gps_inflation": best,
                "max_of_k_p95": threshold["p95"],
                "exceeds_reference": best > threshold["p95"],
            }
    if "11_patching" in steps:
        patch = steps["11_patching"]
        result["patching"] = {
            "gap": patch["mean_s_gps"] - patch["mean_s_norm"],
            "last_layer_critical_recovery": patch["per_layer"][-1]["recovery_critical"],
            "peak_vamb_recovery": max(r["recovery_v_amb"] for r in patch["per_layer"]),
        }
    return result


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Historical results audit",
        "",
        "Recomputed from saved JSON files; no checkpoint inference was rerun.",
        "Item-bootstrap intervals use 10,000 draws, seed 0. Random-head comparisons are descriptive,",
        "not tests of absence of a circuit or corrections for the complete selection procedure.",
        "",
        "## Baseline integrity",
        "",
    ]
    for name, model in audit["models"].items():
        lines.append(
            f"- **{name}**: {len(model['nonfinite'])} non-finite fields; changed/missing baseline items: "
            f"{([r['id'] for r in model['baseline_changes']] or 'none') if model['baseline_comparison_available'] else 'comparison unavailable'}."
        )
    lines += [
        "",
        "## Continuation proxy and sensitivity",
        "",
        "Historical ‘canonical’ and ‘novel’ labels do not certify corpus novelty or linguistic validity.",
        "",
        "| Model | Set | n | Mean [95% CI] | Median | Leave-one-out mean range | Most influential omitted item |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for name, model in audit["models"].items():
        for condition, s in model.get("continuation", {}).items():
            loo = s["leave_one_out"]
            influential = max(loo, key=lambda r: abs(r["mean"] - s["mean"]))
            lines.append(
                f"| {name} | {condition} | {s['n']} | {s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] "
                f"| {s['median']:.3f} | {min(r['mean'] for r in loo):.3f}–{max(r['mean'] for r in loo):.3f} "
                f"| #{influential['omitted_id']} → {influential['mean']:.3f} |"
            )
    lines += [
        "",
        "## Paired scale comparisons",
        "",
        "Differences are larger minus smaller model, matched by item ID.",
        "",
    ]
    for label, s in audit["paired_scale"].items():
        lines.append(f"- {label}: {s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}].")
    lines += ["", "## Random-head reference checks", ""]
    for name, model in audit["models"].items():
        ref = model.get("nv_random_reference")
        if ref:
            lines.append(
                f"- **{name}**, NV Δ: sampled range [{ref['min']:.3f}, {ref['max']:.3f}], n={ref['n']}."
            )
            for h in ref["heads_outside_sampled_range"]:
                lines.append(
                    f"  - L{h['layer']}.H{h['head']}: {h['nv_change']:+.3f}; controls {h['control_inflation']:+.3f}. "
                    "Outside the sampled range; no confirmatory significance claim."
                )
        vc = model.get("continuation_random_reference")
        if vc:
            lines.append(
                f"- **{name}**, continuation: best {vc['best_vc_gps_inflation']:.3f}, "
                f"max-of-{vc['k']} p95 {vc['max_of_k_p95']:.3f}; exceeds reference: {vc['exceeds_reference']}."
            )
    lines += ["", "## Patching checks", ""]
    for name, model in audit["models"].items():
        if p := model.get("patching"):
            lines.append(
                f"- **{name}**: gap {p['gap']:.3f}; final critical-state recovery "
                f"{p['last_layer_critical_recovery']:.3f}; peak ambiguous-word recovery {p['peak_vamb_recovery']:.3f}."
            )
    lines += [
        "",
        "Final-state recovery for a single-token target copies the source readout by construction.",
        "Intermediate patching effects do not isolate syntactic information.",
        "",
        "## Provenance",
        "",
        f"{len(audit['inputs_sha256'])} input files are fingerprinted in `legacy-audit.json`.",
        "Historical logs do not record enough checkpoint/runtime provenance to certify rerun equivalence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/results")
    args = parser.parse_args()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "bootstrap_seed": 0,
        "bootstrap_draws": 10_000,
        "models": {name: audit_model(ROOT / name / "logs") for name in MODELS},
        "paired_scale": {},
        "inputs_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in MODELS
            for p in sorted((ROOT / name / "logs").glob("*.json"))
        },
    }
    for smaller, larger in (("gemma3_1b", "gemma3_27b"), ("qwen4b", "qwen14b")):
        for condition in ("canonical", "novel"):
            first, second = [
                read_results(ROOT / m / "logs/09_novel_predict.json")[condition]["pair_results"]
                for m in (smaller, larger)
            ]
            audit["paired_scale"][f"{larger} − {smaller}, {condition}"] = paired_change(
                first, second
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "legacy-audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "legacy-audit.md").write_text(render_markdown(audit))
    print(args.output / "legacy-audit.md")


if __name__ == "__main__":
    main()
