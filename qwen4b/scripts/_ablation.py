"""Zero-ablation library for the qwen4b investigation.

Each transformer layer's `self_attn.o_proj` is a Linear that maps the
concatenated per-head outputs (shape `(B, T, n_heads * head_dim)`) into the
residual stream. Zeroing a contiguous slice of columns `[h*d : (h+1)*d]` of
`o_proj.weight` is mathematically identical to zeroing the output of head
`h` before it is mixed back into the residual — but cheaper and side-effect
free vs PyTorch hooks.

`HeadsAblated` is the context manager that saves the original columns,
zeroes them for the duration of the with-block, and restores them on exit.

Three families of probes live here:
    measure_effect           — Δ-surprisal at the disambiguator     (steps 02-03)
    measure_verb_continuation — comprehension after disambiguator    (step  04-06)
    measure_lexical_baseline  — parsing vs n-gram diagnostic         (step  07)

Each has a matching `sweep_*` driver that iterates the probe under per-layer
or per-head ablations.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from model_finder.arch import decoder_layers, text_config
from model_finder.sentences import PAIRS, TEMPLATES, Pair
from model_finder.surprisal import surprisal

# Short, structurally varied English sentences with NO garden-path
# ambiguity. We measure the surprisal of the last word in each under every
# ablation: if it explodes, the model has been broken globally, not just
# selectively damaged on garden-path processing.
CONTROL_PROBES: list[tuple[str, str]] = [
    ("The capital of France is", "Paris"),
    ("She opened the door and walked", "inside"),
    ("Water boils at one hundred", "degrees"),
    ("The sun rises in the", "east"),
    ("He bought a loaf of", "bread"),
    ("The cat sat on the", "mat"),
    ("Children love to play in the", "park"),
    ("The doctor wrote a", "prescription"),
]


def get_architecture(model) -> tuple[int, int, int]:
    """Return (n_layers, n_heads, head_dim) from the model config.

    Reads through `arch.text_config` so the nested Gemma-3 multimodal configs
    (params under `config.text_config`) resolve the same as flat ones.
    """
    cfg = text_config(model)
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    return n_layers, n_heads, head_dim


class HeadsAblated:
    """Context manager that zeroes selected (layer, head) pairs.

    Targets are a list of `(layer_idx, head_idx)` tuples. While the context
    is active, the corresponding columns of each layer's `o_proj.weight`
    are zero. On exit the original values are restored verbatim.
    """

    def __init__(
        self,
        model,
        head_dim: int,
        targets: Iterable[tuple[int, int]],
    ):
        self.model = model
        self.head_dim = head_dim
        self.targets = list(targets)
        self._saved: list[tuple[torch.nn.Linear, int, int, torch.Tensor]] = []

    def __enter__(self) -> HeadsAblated:
        for layer_idx, head_idx in self.targets:
            o_proj = decoder_layers(self.model)[layer_idx].self_attn.o_proj
            start = head_idx * self.head_dim
            end = start + self.head_dim
            original = o_proj.weight.data[:, start:end].clone()
            self._saved.append((o_proj, start, end, original))
            o_proj.weight.data[:, start:end].zero_()
        return self

    def __exit__(self, *_exc) -> None:
        for o_proj, start, end, original in self._saved:
            o_proj.weight.data[:, start:end] = original
        self._saved.clear()


# ──────────────────────────────────────────────────────────────────────────
# Metric 1 — Δ-surprisal at the disambiguator (steps 02 & 03)
# ──────────────────────────────────────────────────────────────────────────


def measure_effect(
    model,
    tokenizer,
    pairs: list[Pair],
    device: str,
) -> tuple[dict, list[dict]]:
    """Compute Δ-surprisal AND absolute surprisals across pairs + controls.

    The absolute values are what tell us whether an ablation is hurting GPS
    specifically (s_normal stays low, s_gps stays high → wider gap) or just
    breaking the model overall (both balloon together).
    """
    pair_results: list[dict] = []
    for p in pairs:
        s_gps = surprisal(model, tokenizer, p.gps_prefix, p.critical_word, device)
        s_norm = surprisal(model, tokenizer, p.normal_prefix, p.critical_word, device)
        pair_results.append(
            {
                "id": p.id,
                "template": p.template,
                "critical_word": p.critical_word,
                "surprisal_gps": s_gps,
                "surprisal_normal": s_norm,
                "delta": s_gps - s_norm,
            }
        )

    deltas = np.array([r["delta"] for r in pair_results if not np.isnan(r["delta"])])
    s_gps_arr = np.array(
        [r["surprisal_gps"] for r in pair_results if not np.isnan(r["surprisal_gps"])]
    )
    s_norm_arr = np.array(
        [r["surprisal_normal"] for r in pair_results if not np.isnan(r["surprisal_normal"])]
    )
    s_controls = np.array(
        [surprisal(model, tokenizer, prefix, word, device) for prefix, word in CONTROL_PROBES]
    )
    s_controls = s_controls[~np.isnan(s_controls)]

    summary = {
        "mean_delta": float(deltas.mean()) if deltas.size else float("nan"),
        "mean_s_gps": float(s_gps_arr.mean()) if s_gps_arr.size else float("nan"),
        "mean_s_normal": float(s_norm_arr.mean()) if s_norm_arr.size else float("nan"),
        "mean_s_control": float(s_controls.mean()) if s_controls.size else float("nan"),
    }
    return summary, pair_results


def per_template_mean_delta(pair_results: list[dict]) -> dict[str, float]:
    """Mean (s_gps - s_normal) per template, NaN-safe."""
    out: dict[str, float] = {}
    for t in TEMPLATES:
        deltas = [
            r["delta"]
            for r in pair_results
            if r["template"] == t and not np.isnan(r["delta"])
        ]
        out[t] = float(np.mean(deltas)) if deltas else float("nan")
    return out


def _print_template_summary(
    entries: list[dict],
    baseline_per_template: dict[str, float],
    key: str,
    top_n: int = 5,
) -> None:
    """Print a per-template breakdown for the top-N entries by |effect_drop|."""
    ranked = sorted(entries, key=lambda e: abs(e["effect_drop"]), reverse=True)[:top_n]
    print()
    print(f"  Top {top_n} {key} by |effect_drop|, per-template drop:")
    header = (
        f"    {key:>5}  {'overall':>8}  "
        + "  ".join(f"{t:>7}" for t in TEMPLATES)
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    print(
        f"    {'base':>5}  {0.0:>+8.3f}  "
        + "  ".join(f"{0.0:>+7.3f}" for _ in TEMPLATES)
        + "      (baseline per-template: "
        + ", ".join(f"{t}={baseline_per_template[t]:+.2f}" for t in TEMPLATES)
        + ")"
    )
    for e in ranked:
        ptd = e["per_template_drop"]
        print(
            f"    {e[key]:>5}  {e['effect_drop']:>+8.3f}  "
            + "  ".join(f"{ptd[t]:>+7.3f}" for t in TEMPLATES)
        )


def _print_header() -> None:
    print(
        f"  {'layer':>5}  {'Δ':>7}  {'drop':>7}  "
        f"{'s_gps':>7}  {'s_norm':>7}  {'s_ctrl':>7}  bar"
    )


def _print_row(label: str, summary: dict, baseline: dict | None) -> None:
    delta = summary["mean_delta"]
    drop = delta - baseline["mean_delta"] if baseline else 0.0
    bar_len = max(0, min(40, int(-drop * 8)))
    bar = "█" * bar_len
    print(
        f"  {label:>5}  "
        f"{delta:+7.3f}  {drop:+7.3f}  "
        f"{summary['mean_s_gps']:7.3f}  {summary['mean_s_normal']:7.3f}  "
        f"{summary['mean_s_control']:7.3f}  {bar}",
        flush=True,
    )


def sweep_layers(
    model,
    tokenizer,
    device: str,
    pairs: list[Pair] = PAIRS,
) -> dict:
    """Ablate ALL heads in each layer, one layer at a time."""
    n_layers, n_heads, head_dim = get_architecture(model)

    print(f"  architecture: {n_layers} layers, {n_heads} heads, head_dim={head_dim}")
    print("  baseline (no ablation) ...", flush=True)
    baseline_summary, baseline_pair_results = measure_effect(model, tokenizer, pairs, device)
    baseline_per_template = per_template_mean_delta(baseline_pair_results)
    _print_header()
    _print_row("base", baseline_summary, None)

    per_layer = []
    for layer_idx in range(n_layers):
        targets = [(layer_idx, h) for h in range(n_heads)]
        with HeadsAblated(model, head_dim, targets):
            summary, pair_results = measure_effect(model, tokenizer, pairs, device)
        abl_per_template = per_template_mean_delta(pair_results)
        per_layer.append(
            {
                "layer": layer_idx,
                "summary_under_ablation": summary,
                "effect_drop": summary["mean_delta"] - baseline_summary["mean_delta"],
                "control_inflation": summary["mean_s_control"]
                - baseline_summary["mean_s_control"],
                "per_template_drop": {
                    t: abl_per_template[t] - baseline_per_template[t] for t in TEMPLATES
                },
            }
        )
        _print_row(str(layer_idx), summary, baseline_summary)

    _print_template_summary(per_layer, baseline_per_template, key="layer", top_n=5)

    return {
        "model": getattr(model.config, "_name_or_path", "unknown"),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "baseline_summary": baseline_summary,
        "baseline_per_template": baseline_per_template,
        "baseline_pair_results": baseline_pair_results,
        "per_layer": per_layer,
    }


def sweep_heads_in_layer(
    model,
    tokenizer,
    device: str,
    layer_idx: int,
    pairs: list[Pair] = PAIRS,
) -> dict:
    """Ablate each head individually within a single layer."""
    n_layers, n_heads, head_dim = get_architecture(model)
    if not 0 <= layer_idx < n_layers:
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers})")

    print(f"  layer {layer_idx}: ablating each of {n_heads} heads individually")
    print("  baseline (no ablation) ...", flush=True)
    baseline_summary, baseline_pair_results = measure_effect(model, tokenizer, pairs, device)
    baseline_per_template = per_template_mean_delta(baseline_pair_results)
    _print_header()
    _print_row("base", baseline_summary, None)

    per_head = []
    for head_idx in range(n_heads):
        with HeadsAblated(model, head_dim, [(layer_idx, head_idx)]):
            summary, pair_results = measure_effect(model, tokenizer, pairs, device)
        abl_per_template = per_template_mean_delta(pair_results)
        per_head.append(
            {
                "head": head_idx,
                "summary_under_ablation": summary,
                "effect_drop": summary["mean_delta"] - baseline_summary["mean_delta"],
                "control_inflation": summary["mean_s_control"]
                - baseline_summary["mean_s_control"],
                "per_template_drop": {
                    t: abl_per_template[t] - baseline_per_template[t] for t in TEMPLATES
                },
            }
        )
        _print_row(str(head_idx), summary, baseline_summary)

    _print_template_summary(per_head, baseline_per_template, key="head", top_n=5)

    return {
        "layer": layer_idx,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "baseline_summary": baseline_summary,
        "baseline_per_template": baseline_per_template,
        "per_head": per_head,
    }


# ──────────────────────────────────────────────────────────────────────────
# Metric 2 — Verb-continuation surprisal (steps 04, 05, 06)
# ──────────────────────────────────────────────────────────────────────────


def measure_verb_continuation(
    model,
    tokenizer,
    pairs: list[Pair],
    device: str,
) -> tuple[dict, list[dict]]:
    """Behavioural comprehension test.

    For every NV pair, compute -log P(verb_continuation | full_prefix) where
    `full_prefix = gps_prefix + " " + critical_word` (the disambiguator has
    already been emitted). A low surprisal means the model has committed to
    the verb-parse and now expects a plausible direct object. We do the same
    on the NORMAL prefix so we can compare:

        vc_delta = s_vc_gps - s_vc_normal

    A positive `vc_delta` is the residual cost of garden-path interpretation
    AFTER disambiguation. The behavioural goal of ablation is to *widen* this
    gap (the model loses the ability to reanalyse), while keeping controls
    intact.
    """
    pair_results: list[dict] = []
    for p in pairs:
        if p.template != "NV" or not p.verb_continuation:
            continue
        full_gps = p.gps_prefix + " " + p.critical_word
        full_norm = p.normal_prefix + " " + p.critical_word
        s_gps = surprisal(model, tokenizer, full_gps, p.verb_continuation, device)
        s_norm = surprisal(model, tokenizer, full_norm, p.verb_continuation, device)
        pair_results.append(
            {
                "id": p.id,
                "verb_continuation": p.verb_continuation,
                "s_vc_gps": s_gps,
                "s_vc_normal": s_norm,
                "vc_delta": s_gps - s_norm,
            }
        )

    deltas = np.array([r["vc_delta"] for r in pair_results if not np.isnan(r["vc_delta"])])
    s_gps_arr = np.array(
        [r["s_vc_gps"] for r in pair_results if not np.isnan(r["s_vc_gps"])]
    )
    s_norm_arr = np.array(
        [r["s_vc_normal"] for r in pair_results if not np.isnan(r["s_vc_normal"])]
    )
    s_controls = np.array(
        [surprisal(model, tokenizer, prefix, word, device) for prefix, word in CONTROL_PROBES]
    )
    s_controls = s_controls[~np.isnan(s_controls)]

    summary = {
        "mean_s_vc_gps": float(s_gps_arr.mean()) if s_gps_arr.size else float("nan"),
        "mean_s_vc_normal": float(s_norm_arr.mean()) if s_norm_arr.size else float("nan"),
        "mean_vc_delta": float(deltas.mean()) if deltas.size else float("nan"),
        "mean_s_control": float(s_controls.mean()) if s_controls.size else float("nan"),
    }
    return summary, pair_results


def _print_predict_row(label: str, summary: dict, baseline: dict | None) -> None:
    def diff(key: str) -> str:
        if baseline is None:
            return ""
        return f"({summary[key] - baseline[key]:+.3f})"

    print(
        f"  {label:<14}  "
        f"vc_gps={summary['mean_s_vc_gps']:6.3f} {diff('mean_s_vc_gps'):>9}  "
        f"vc_norm={summary['mean_s_vc_normal']:6.3f} {diff('mean_s_vc_normal'):>9}  "
        f"Δ={summary['mean_vc_delta']:+6.3f} {diff('mean_vc_delta'):>9}  "
        f"ctrl={summary['mean_s_control']:6.3f} {diff('mean_s_control'):>9}",
        flush=True,
    )


def sweep_predict(
    model,
    tokenizer,
    device: str,
    target_sets: list[list[tuple[int, int]]],
    pairs: list[Pair] = PAIRS,
) -> dict:
    """Comprehension test under multiple ablation conditions.

    Baseline is run first; each subsequent target set is reported with
    deltas relative to baseline.
    """
    _, _, head_dim = get_architecture(model)

    print("  computing baseline ...", flush=True)
    baseline_summary, baseline_pair_results = measure_verb_continuation(
        model, tokenizer, pairs, device
    )
    _print_predict_row("baseline", baseline_summary, None)

    conditions: list[dict] = [
        {
            "label": "baseline",
            "targets": [],
            "summary": baseline_summary,
            "pair_results": baseline_pair_results,
        }
    ]

    for targets in target_sets:
        label = "+".join(f"L{li}.H{h}" for li, h in targets) if targets else "baseline"
        with HeadsAblated(model, head_dim, targets):
            summary, pair_results = measure_verb_continuation(model, tokenizer, pairs, device)
        conditions.append(
            {
                "label": label,
                "targets": [list(t) for t in targets],
                "summary": summary,
                "pair_results": pair_results,
            }
        )
        _print_predict_row(label, summary, baseline_summary)

    print()
    print("  Per-pair s_vc_gps (higher = model less able to predict verb-object):")
    header = f"    {'id':>3}  {'vc':<10}  " + "  ".join(
        f"{c['label']:>16}" for c in conditions
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for i, r0 in enumerate(baseline_pair_results):
        row = f"    {r0['id']:>3}  {r0['verb_continuation']:<10}  "
        row += "  ".join(
            f"{c['pair_results'][i]['s_vc_gps']:>16.3f}" for c in conditions
        )
        print(row)

    return {"conditions": conditions}


def _print_predict_sweep_header() -> None:
    print(
        f"  {'lyr':>4}  {'vc_gps':>7}  {'+gps':>7}  {'vc_norm':>7}  "
        f"{'+norm':>7}  {'Δ_vc':>7}  {'+Δ':>7}  {'ctrl':>7}  {'+ctrl':>7}  bar",
        flush=True,
    )


def _print_predict_sweep_row(label: str, summary: dict, baseline: dict | None) -> None:
    if baseline is None:
        d_gps = d_norm = d_delta = d_ctrl = 0.0
    else:
        d_gps = summary["mean_s_vc_gps"] - baseline["mean_s_vc_gps"]
        d_norm = summary["mean_s_vc_normal"] - baseline["mean_s_vc_normal"]
        d_delta = summary["mean_vc_delta"] - baseline["mean_vc_delta"]
        d_ctrl = summary["mean_s_control"] - baseline["mean_s_control"]
    bar_len = max(0, min(40, int(d_gps * 16)))
    bar = "█" * bar_len
    print(
        f"  {label:>4}  "
        f"{summary['mean_s_vc_gps']:7.3f}  {d_gps:+7.3f}  "
        f"{summary['mean_s_vc_normal']:7.3f}  {d_norm:+7.3f}  "
        f"{summary['mean_vc_delta']:+7.3f}  {d_delta:+7.3f}  "
        f"{summary['mean_s_control']:7.3f}  {d_ctrl:+7.3f}  {bar}",
        flush=True,
    )


def _print_predict_sweep_ranking(entries: list[dict], key: str, top_n: int = 10) -> None:
    print()
    print(f"  Top {top_n} {key} by vc_gps_inflation (selectivity = vc_gps - ctrl):")
    print(
        f"    {key:>5}  {'+vc_gps':>8}  {'+vc_delta':>10}  {'+ctrl':>8}  {'selectivity':>12}"
    )
    print("    " + "-" * 52)
    ranked = sorted(entries, key=lambda e: e["vc_gps_inflation"], reverse=True)[:top_n]
    for e in ranked:
        sel = e["vc_gps_inflation"] - e["control_inflation"]
        print(
            f"    {e[key]:>5}  {e['vc_gps_inflation']:>+8.3f}  "
            f"{e['vc_delta_inflation']:>+10.3f}  {e['control_inflation']:>+8.3f}  "
            f"{sel:>+12.3f}"
        )


def sweep_predict_layers(
    model,
    tokenizer,
    device: str,
    pairs: list[Pair] = PAIRS,
) -> dict:
    """Per-layer ablation sweep with comprehension as the metric.

    For each layer ablate ALL heads and measure `mean_s_vc_gps`. A layer
    whose ablation INCREASES vc_gps significantly without proportionally
    increasing controls contributes to the verb-parse reanalysis.

    Selectivity = vc_gps_inflation − control_inflation.
    """
    n_layers, n_heads, head_dim = get_architecture(model)

    print(f"  architecture: {n_layers} layers, {n_heads} heads, head_dim={head_dim}")
    print("  baseline (no ablation) ...", flush=True)
    baseline_summary, _ = measure_verb_continuation(model, tokenizer, pairs, device)
    _print_predict_sweep_header()
    _print_predict_sweep_row("base", baseline_summary, None)

    per_layer: list[dict] = []
    for layer_idx in range(n_layers):
        targets = [(layer_idx, h) for h in range(n_heads)]
        with HeadsAblated(model, head_dim, targets):
            summary, _ = measure_verb_continuation(model, tokenizer, pairs, device)
        per_layer.append(
            {
                "layer": layer_idx,
                "summary_under_ablation": summary,
                "vc_gps_inflation": summary["mean_s_vc_gps"]
                - baseline_summary["mean_s_vc_gps"],
                "vc_normal_inflation": summary["mean_s_vc_normal"]
                - baseline_summary["mean_s_vc_normal"],
                "vc_delta_inflation": summary["mean_vc_delta"]
                - baseline_summary["mean_vc_delta"],
                "control_inflation": summary["mean_s_control"]
                - baseline_summary["mean_s_control"],
            }
        )
        _print_predict_sweep_row(str(layer_idx), summary, baseline_summary)

    _print_predict_sweep_ranking(per_layer, key="layer", top_n=10)

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "baseline_summary": baseline_summary,
        "per_layer": per_layer,
    }


def sweep_predict_heads_in_layer(
    model,
    tokenizer,
    device: str,
    layer_idx: int,
    pairs: list[Pair] = PAIRS,
) -> dict:
    """Per-head sweep within a single layer, comprehension metric."""
    n_layers, n_heads, head_dim = get_architecture(model)
    if not 0 <= layer_idx < n_layers:
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers})")

    print(f"  layer {layer_idx}: ablating each of {n_heads} heads individually")
    print("  baseline (no ablation) ...", flush=True)
    baseline_summary, _ = measure_verb_continuation(model, tokenizer, pairs, device)
    _print_predict_sweep_header()
    _print_predict_sweep_row("base", baseline_summary, None)

    per_head: list[dict] = []
    for head_idx in range(n_heads):
        with HeadsAblated(model, head_dim, [(layer_idx, head_idx)]):
            summary, _ = measure_verb_continuation(model, tokenizer, pairs, device)
        per_head.append(
            {
                "head": head_idx,
                "summary_under_ablation": summary,
                "vc_gps_inflation": summary["mean_s_vc_gps"]
                - baseline_summary["mean_s_vc_gps"],
                "vc_normal_inflation": summary["mean_s_vc_normal"]
                - baseline_summary["mean_s_vc_normal"],
                "vc_delta_inflation": summary["mean_vc_delta"]
                - baseline_summary["mean_vc_delta"],
                "control_inflation": summary["mean_s_control"]
                - baseline_summary["mean_s_control"],
            }
        )
        _print_predict_sweep_row(str(head_idx), summary, baseline_summary)

    _print_predict_sweep_ranking(per_head, key="head", top_n=10)

    return {
        "layer": layer_idx,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "baseline_summary": baseline_summary,
        "per_head": per_head,
    }


# ──────────────────────────────────────────────────────────────────────────
# Metric 3 — Lexical-floor diagnostic (step 07)
# ──────────────────────────────────────────────────────────────────────────


# Neutral 4-word prefixes ending in each critical_word used by NV pairs.
# Used as a "lexical floor" by `measure_lexical_baseline`.
_LEXICAL_ANCHORS: dict[str, str] = {
    "the": "He looked carefully at the",
    "their": "They wrote down all their",
    "for": "She had been searching for",
}


def _v_amb_word(p: Pair) -> str:
    """The noun/verb-ambiguous word — the last word of `gps_prefix`."""
    return p.gps_prefix.split()[-1]


def measure_lexical_baseline(
    model,
    tokenizer,
    pairs: list[Pair],
    device: str,
) -> tuple[dict, list[dict]]:
    """Diagnostic probe: is verb_continuation prediction parsing or n-gram?

    For every NV pair, compute -log P(verb_continuation | prefix) under
    three prefixes that share the SAME critical word at the end:

        full      = gps_prefix + " " + critical_word
        stripped  = v_amb_word + " " + critical_word  (drops NP setup)
        anchor    = neutral 4-word prefix ending in critical_word

    Interpretation:
        s_full ≈ s_stripped  → local lexical association dominates
        s_full ≈ s_anchor    → no specific computation at all (rare)
        s_full << s_stripped → global context matters; computation present
        s_full >> s_anchor   → GPS prefix actively suppresses verb-parse
    """
    pair_results: list[dict] = []
    for p in pairs:
        if p.template != "NV" or not p.verb_continuation:
            continue
        anchor = _LEXICAL_ANCHORS.get(p.critical_word)
        if anchor is None:
            continue
        full = p.gps_prefix + " " + p.critical_word
        v_amb = _v_amb_word(p)
        stripped = v_amb + " " + p.critical_word
        s_full = surprisal(model, tokenizer, full, p.verb_continuation, device)
        s_stripped = surprisal(model, tokenizer, stripped, p.verb_continuation, device)
        s_anchor = surprisal(model, tokenizer, anchor, p.verb_continuation, device)
        pair_results.append(
            {
                "id": p.id,
                "v_amb_word": v_amb,
                "verb_continuation": p.verb_continuation,
                "full_prefix": full,
                "stripped_prefix": stripped,
                "anchor_prefix": anchor,
                "s_full": s_full,
                "s_stripped": s_stripped,
                "s_anchor": s_anchor,
            }
        )

    arrs = {
        k: np.array([r[k] for r in pair_results if not np.isnan(r[k])])
        for k in ("s_full", "s_stripped", "s_anchor")
    }
    summary = {
        f"mean_{k}": (float(a.mean()) if a.size else float("nan")) for k, a in arrs.items()
    }
    return summary, pair_results
