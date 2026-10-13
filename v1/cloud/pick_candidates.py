#!/usr/bin/env python3
"""Pick ablation candidates from the sweep JSONs, so run_all.sh can chain
03/04/12 automatically instead of a human eyeballing the sweeps.

Mirrors the manual selection in summary.md:
  * surprise layers/heads  -> most negative `effect_drop` (biggest reduction of
    the GPS surprise), dropping degenerate rows where the model is broken
    globally (large |control_inflation|, e.g. layer 0);
  * comprehension layers   -> largest positive `vc_gps_inflation` (ablating the
    layer hurts the post-disambiguator prediction most), same control filter;
  * comprehension heads    -> largest positive `vc_gps_inflation` from the 06
    sweeps, additionally requiring `vc_delta_inflation` > 0 (the damage must be
    GPS-specific, not shared with the normal sentence), same control filter.

Prints a plain space-separated list to stdout ("23 25 35" or "25:7 35:0"), so
the shell can splice it straight into the next command. Stdlib only.

Usage:
  pick_candidates.py layers   02_layer_sweep.json          [--top N] [--max-control C]
  pick_candidates.py heads    03_head_sweep_L*.json ...     [--top K] [--max-control C]
  pick_candidates.py vclayers 05_predict_layer_sweep.json   [--top N] [--max-control C]
  pick_candidates.py vcheads  06_predict_head_sweep_L*.json [--top K] [--max-control C]
"""

from __future__ import annotations

import argparse
import json
import sys


def _results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)["results"]


def pick_layers(paths: list[str], top: int, max_control: float) -> list[str]:
    rows = _results(paths[0])["per_layer"]
    cand = [
        r
        for r in rows
        if r["effect_drop"] < 0 and abs(r["control_inflation"]) <= max_control
    ]
    cand.sort(key=lambda r: r["effect_drop"])  # most negative first
    return [str(r["layer"]) for r in cand[:top]]


def pick_heads(paths: list[str], top: int, max_control: float) -> list[str]:
    cand: list[tuple[int, int, float]] = []
    for p in paths:
        try:
            res = _results(p)
        except (FileNotFoundError, KeyError):
            continue  # a sweep that wasn't produced just contributes nothing
        layer = res["layer"]
        for h in res["per_head"]:
            if h["effect_drop"] < 0 and abs(h["control_inflation"]) <= max_control:
                cand.append((layer, h["head"], h["effect_drop"]))
    cand.sort(key=lambda t: t[2])  # most negative effect_drop first
    return [f"{li}:{hd}" for li, hd, _ in cand[:top]]


def pick_vclayers(paths: list[str], top: int, max_control: float) -> list[str]:
    rows = _results(paths[0])["per_layer"]
    cand = [
        r
        for r in rows
        if r["vc_gps_inflation"] > 0 and abs(r["control_inflation"]) <= max_control
    ]
    cand.sort(key=lambda r: r["vc_gps_inflation"], reverse=True)  # biggest hurt first
    return [str(r["layer"]) for r in cand[:top]]


def pick_vcheads(paths: list[str], top: int, max_control: float) -> list[str]:
    cand: list[tuple[int, int, float]] = []
    for p in paths:
        try:
            res = _results(p)
        except (FileNotFoundError, KeyError):
            continue  # a sweep that wasn't produced just contributes nothing
        layer = res["layer"]
        for h in res["per_head"]:
            if (
                h["vc_gps_inflation"] > 0
                and h["vc_delta_inflation"] > 0
                and abs(h["control_inflation"]) <= max_control
            ):
                cand.append((layer, h["head"], h["vc_gps_inflation"]))
    cand.sort(key=lambda t: t[2], reverse=True)  # biggest hurt first
    return [f"{li}:{hd}" for li, hd, _ in cand[:top]]


_PICKERS = {
    "layers": pick_layers,
    "heads": pick_heads,
    "vclayers": pick_vclayers,
    "vcheads": pick_vcheads,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=_PICKERS)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-control", type=float, default=0.5)
    args = ap.parse_args()
    try:
        out = _PICKERS[args.kind](args.paths, args.top, args.max_control)
    except (FileNotFoundError, KeyError) as e:
        print(f"pick_candidates: {args.kind}: {e}", file=sys.stderr)
        out = []
    print(" ".join(out))


if __name__ == "__main__":
    main()
