"""Step 03 — forced-choice comprehension probe (IDIOM and ROLE traps).

LM version of the Christianson et al. (2001) lingering-misinterpretation
paradigm (D4): score the correct vs trap answer after a cloze QA prompt,
on the trap sentence and on its matched control.

The inverse-GPS signature: accuracy(control) high, accuracy(trap) low —
the model answers from plausibility/idiom priors instead of the parse,
on sentences whose surface syntax is unambiguous for a human reader.

Note: base models answer cloze QA imperfectly across the board; the
measure of interest is the trap-vs-control GAP, not absolute accuracy.
Instruct models can be screened with the same items via their chat
template (future step).

    uv run python inverse_gps/scripts/03_qa_probe.py --model Qwen/Qwen3-4B-Base
"""

import numpy as np
from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from inverse_gps.detectors import qa_probe
    from inverse_gps.sentences import QA_TEMPLATES, by_template

    item_results = []
    for tmpl in QA_TEMPLATES:
        for it in by_template(tmpl):
            r = {"id": it.id, "template": it.template}
            r.update(qa_probe(model, tokenizer, it, device))
            item_results.append(r)
            print(
                f"[{it.id:2d}] {it.template:5}  "
                f"trap {'✓' if r['trap_correct'] else '✗'} "
                f"(margin {r['trap_margin']:+6.3f})   "
                f"ctrl {'✓' if r['control_correct'] else '✗'} "
                f"(margin {r['control_margin']:+6.3f})"
            )

    per_template = {}
    for tmpl in QA_TEMPLATES:
        rs = [r for r in item_results if r["template"] == tmpl]
        acc_t = np.mean([r["trap_correct"] for r in rs])
        acc_c = np.mean([r["control_correct"] for r in rs])
        per_template[tmpl] = {
            "n": len(rs),
            "accuracy_trap": float(acc_t),
            "accuracy_control": float(acc_c),
            "gap": float(acc_c - acc_t),
            "mean_trap_margin": float(np.mean([r["trap_margin"] for r in rs])),
            "mean_control_margin": float(
                np.mean([r["control_margin"] for r in rs])
            ),
        }

    print("\nPer-template accuracy (trap / control / gap):")
    for tmpl, s in per_template.items():
        print(
            f"  {tmpl:6}  {s['accuracy_trap']:.2f} / "
            f"{s['accuracy_control']:.2f} / {s['gap']:+.2f}"
        )

    return {"per_template": per_template, "item_results": item_results}


if __name__ == "__main__":
    run_step("03_qa_probe", body)
