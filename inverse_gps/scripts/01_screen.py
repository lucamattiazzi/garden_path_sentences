"""Step 01 — surprisal + capture screen of the IDIOM and QUOTE traps.

For every item: D1 Δ-surprisal at the forced continuation and D2 capture
scores (attractor-vs-target preference, trap vs control prefix).

The inverse-GPS signature to look for:
    delta > 0                   the trap prefix inflates surprisal at the
                                syntactically forced continuation
    capture_trap > 0            the model outright prefers the wrong
                                continuation in the trap
    capture_specificity > 0     the pull disappears in the control

    uv run python inverse_gps/scripts/01_screen.py --model Qwen/Qwen3-4B-Base
"""

import numpy as np
from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from inverse_gps.detectors import capture_scores, surprisal_delta
    from inverse_gps.sentences import SURPRISAL_TEMPLATES, by_template

    item_results = []
    for tmpl in SURPRISAL_TEMPLATES:
        for it in by_template(tmpl):
            r = {"id": it.id, "template": it.template}
            r.update(surprisal_delta(model, tokenizer, it, device))
            r.update(capture_scores(model, tokenizer, it, device))
            item_results.append(r)
            print(
                f"[{it.id:2d}] {it.template:5}  Δ={r['delta']:+7.3f}  "
                f"cap_trap={r['capture_trap']:+7.3f}  "
                f"cap_ctrl={r['capture_control']:+7.3f}  "
                f"spec={r['capture_specificity']:+7.3f}"
            )

    per_template = {}
    for tmpl in SURPRISAL_TEMPLATES:
        rs = [r for r in item_results if r["template"] == tmpl]
        deltas = np.array([r["delta"] for r in rs])
        caps = np.array([r["capture_trap"] for r in rs])
        specs = np.array([r["capture_specificity"] for r in rs])
        per_template[tmpl] = {
            "n": len(rs),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "frac_delta_pos": float((deltas > 0).mean()),
            "mean_capture_trap": float(caps.mean()),
            "frac_captured": float((caps > 0).mean()),
            "mean_capture_specificity": float(specs.mean()),
        }

    print("\nPer-template summary:")
    for tmpl, s in per_template.items():
        print(
            f"  {tmpl:6}  Δ̄={s['mean_delta']:+.3f}  "
            f"P(Δ>0)={s['frac_delta_pos']:.2f}  "
            f"captured={s['frac_captured']:.2f}  "
            f"spec̄={s['mean_capture_specificity']:+.3f}"
        )

    return {"per_template": per_template, "item_results": item_results}


if __name__ == "__main__":
    run_step("01_screen", body)
