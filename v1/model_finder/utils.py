import gc
from dataclasses import asdict

import numpy as np
import torch
from scipy import stats

from model_finder.models import ModelSpec
from model_finder.sentences import TEMPLATES


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cleanup(device: str):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def aggregate(spec: ModelSpec, pair_results: list[dict]) -> dict:
    deltas_all = np.array(
        [r["delta"] for r in pair_results if not np.isnan(r["delta"])]
    )

    per_template = {}
    for tmpl in TEMPLATES:
        d = np.array(
            [
                r["delta"]
                for r in pair_results
                if r["template"] == tmpl and not np.isnan(r["delta"])
            ]
        )
        per_template[tmpl] = {
            "n": int(d.size),
            "mean_delta": float(d.mean()) if d.size else float("nan"),
            "median_delta": float(np.median(d)) if d.size else float("nan"),
            "accuracy": float((d > 0).mean()) if d.size else float("nan"),
        }

    if deltas_all.size == 0:
        return {
            "model": asdict(spec),
            "n_pairs": 0,
            "error": "all pairs failed to tokenise",
            "pair_results": pair_results,
        }

    try:
        w_stat, w_p = stats.wilcoxon(deltas_all, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    rng = np.random.default_rng(42)
    boots = rng.choice(deltas_all, size=(2000, deltas_all.size), replace=True).mean(
        axis=1
    )
    ci_low, ci_high = (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )

    return {
        "model": asdict(spec),
        "n_pairs": int(deltas_all.size),
        "mean_delta": float(deltas_all.mean()),
        "median_delta": float(np.median(deltas_all)),
        "ci95": [ci_low, ci_high],
        "accuracy": float((deltas_all > 0).mean()),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "per_template": per_template,
        "pair_results": pair_results,
    }
