"""Probes for the qwen32b triage. **No ablation**.

Only the two probes needed for steps 01 and 07 are reimplemented here, so
this folder stays independent from `qwen4b/`. Identical content to the
relevant fragments of `qwen4b/scripts/_ablation.py`, modulo the absence of
the `device` argument (MLX picks Apple Silicon automatically).
"""

from __future__ import annotations

import numpy as np
from _surprisal import surprisal

from model_finder.sentences import TEMPLATES, Pair

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

_LEXICAL_ANCHORS: dict[str, str] = {
    "the": "He looked carefully at the",
    "their": "They wrote down all their",
    "for": "She had been searching for",
}


def _v_amb_word(p: Pair) -> str:
    return p.gps_prefix.split()[-1]


def measure_effect(model, tokenizer, pairs: list[Pair]) -> tuple[dict, list[dict]]:
    """Δ-surprisal at the disambiguator + absolute s_gps / s_normal / s_control."""
    pair_results: list[dict] = []
    for p in pairs:
        s_gps = surprisal(model, tokenizer, p.gps_prefix, p.critical_word)
        s_norm = surprisal(model, tokenizer, p.normal_prefix, p.critical_word)
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
        [surprisal(model, tokenizer, prefix, word) for prefix, word in CONTROL_PROBES]
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
    out: dict[str, float] = {}
    for t in TEMPLATES:
        deltas = [
            r["delta"]
            for r in pair_results
            if r["template"] == t and not np.isnan(r["delta"])
        ]
        out[t] = float(np.mean(deltas)) if deltas else float("nan")
    return out


def measure_lexical_baseline(
    model, tokenizer, pairs: list[Pair]
) -> tuple[dict, list[dict]]:
    """Compare P(verb_continuation) under full / stripped / anchor prefixes."""
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
        s_full = surprisal(model, tokenizer, full, p.verb_continuation)
        s_stripped = surprisal(model, tokenizer, stripped, p.verb_continuation)
        s_anchor = surprisal(model, tokenizer, anchor, p.verb_continuation)
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
        f"mean_{k}": (float(a.mean()) if a.size else float("nan"))
        for k, a in arrs.items()
    }
    return summary, pair_results
