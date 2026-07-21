"""Step 02 — free-continuation behaviour on the IDIOM and QUOTE traps.

Greedy-decodes 10 tokens after the trap and control prefixes and flags
whether the attractor surfaces (D3). Behavioural analogue of the main
pipeline's predict steps: does the pull measured in step 01 actually steer
generation off the grammatical rails?

    uv run python inverse_gps/scripts/02_continuation.py --model Qwen/Qwen3-4B-Base
"""

from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from inverse_gps.detectors import continuation_captured, greedy_continuation
    from inverse_gps.sentences import SURPRISAL_TEMPLATES, by_template

    item_results = []
    for tmpl in SURPRISAL_TEMPLATES:
        for it in by_template(tmpl):
            cont_trap = greedy_continuation(
                model, tokenizer, it.trap_prefix, device
            )
            cont_ctrl = greedy_continuation(
                model, tokenizer, it.control_prefix, device
            )
            captured_trap = continuation_captured(cont_trap, it.attractor)
            captured_ctrl = continuation_captured(cont_ctrl, it.attractor)
            item_results.append({
                "id": it.id,
                "template": it.template,
                "trap_continuation": cont_trap,
                "control_continuation": cont_ctrl,
                "captured_trap": captured_trap,
                "captured_control": captured_ctrl,
            })
            mark = "✗ CAPTURED" if captured_trap else "✓"
            print(f"[{it.id:2d}] {it.template:5} {mark}")
            print(f"     trap: ...{it.trap_prefix.split(',')[-1]} → {cont_trap!r}")
            print(f"     ctrl: → {cont_ctrl!r}")

    per_template = {}
    for tmpl in SURPRISAL_TEMPLATES:
        rs = [r for r in item_results if r["template"] == tmpl]
        per_template[tmpl] = {
            "n": len(rs),
            "capture_rate_trap": sum(r["captured_trap"] for r in rs) / len(rs),
            "capture_rate_control": sum(r["captured_control"] for r in rs) / len(rs),
        }

    print("\nPer-template capture rates (trap / control):")
    for tmpl, s in per_template.items():
        print(
            f"  {tmpl:6}  {s['capture_rate_trap']:.2f} / "
            f"{s['capture_rate_control']:.2f}"
        )

    return {"per_template": per_template, "item_results": item_results}


if __name__ == "__main__":
    run_step("02_continuation", body)
