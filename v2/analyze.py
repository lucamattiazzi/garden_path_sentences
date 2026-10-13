"""Recompute a concise descriptive report from saved raw pilot outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.experiment import family_interval


def describe(run: Path) -> str:
    metadata = json.loads((run / "metadata.json").read_text())
    gps = json.loads((run / "garden-paths.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("Run is incomplete")
    last = max(row["layer"] for row in gps["rows"])
    layers = sorted({0, last // 4, last // 2, 3 * last // 4, last - 1, last})
    lines = [
        f"# Descriptive results: {run.name}",
        "",
        f"Model: `{metadata['model']}`; revision `{metadata['revision']}`.",
        f"GPU: {metadata['gpu']}; inference/analysis wall time: {metadata['elapsed_seconds']:.1f} s.",
        "",
        "All results are exploratory. Layer indices are zero-based.",
        "Intervals are family-bootstrap, pointwise, with no layer-selection correction.",
        "",
        "## Incremental prediction",
        "",
        "Positive cost means the word was less expected without the comma.",
        "",
        "| Word position | Mean cost, nats | 95% interval | Families |",
        "|---|---:|---|---:|",
    ]
    for site in ("pre_disambiguation", "post_disambiguation"):
        subset = [
            r
            for r in gps["rows"]
            if r["site"] == site and r["recipient"] == "ambiguous" and r["layer"] == 0
        ]
        stat = family_interval([(r["family"], r["donor_baseline"] - r["baseline"]) for r in subset])
        lo, hi = stat["ci95"]
        lines.append(f"| {site} | {stat['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {stat['families']} |")
    lines += [
        "",
        "## Bidirectional patching",
        "",
        "Mean change in recipient log P(word), nats. Lexical donor changes the first actor only.",
        "The final block is a readout-copy control; exclude it from localization claims.",
        "",
        "| Layer | Site | Recipient | Syntax/punctuation donor | Lexical donor |",
        "|---:|---|---|---:|---:|",
    ]
    for row in gps["summaries"]:
        if row["layer"] in layers:
            lines.append(
                f"| {row['layer']} | {row['site']} | {row['recipient']} | {row['effect']['mean']:.3f} | {row['lexical_effect']['mean']:.3f} |"
            )
    error = max(abs(r["self_patch"] - r["baseline"]) for r in gps["rows"])
    lines += [
        "",
        f"Maximum identity-patch discrepancy: {error:.8g} nats.",
        "",
        "Full curves, individual scores, patch norms and intervals are in the JSON artifacts.",
        "These measurements do not establish a syntax-specific circuit, online role revision,",
        "or a model-size trend. See the protocol before drawing those conclusions.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    (args.run / "summary.md").write_text(describe(args.run))
