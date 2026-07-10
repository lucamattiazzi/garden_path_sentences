"""Step 01 — baseline GPS-surprisal effect on Qwen3-32B-MLX-8bit.

MLX-native counterpart of `qwen4b/scripts/01_baseline.py`. Confirms that
the 32B model perceives the garden-path effect on the same 50-pair set
and reports the per-template Δ for direct comparison to the 4B numbers.
"""

from _common import run_step  # noqa: E402
from _lib import measure_effect, per_template_mean_delta  # noqa: E402

from model_finder.sentences import PAIRS, TEMPLATES  # noqa: E402


def body(model, tokenizer) -> dict:
    summary, pair_results = measure_effect(model, tokenizer, PAIRS)
    per_t = per_template_mean_delta(pair_results)

    print(f"Baseline (no ablation) over {len(pair_results)} pairs:")
    print(f"  mean Δ          = {summary['mean_delta']:+.4f}")
    print(f"  mean s_gps      = {summary['mean_s_gps']:.4f}")
    print(f"  mean s_normal   = {summary['mean_s_normal']:.4f}")
    print(f"  mean s_control  = {summary['mean_s_control']:.4f}")
    print()
    print("  Per-template mean Δ:")
    for t in TEMPLATES:
        print(f"    {t:6}  Δ̄ = {per_t[t]:+.3f}")

    return {
        "summary": summary,
        "per_template": per_t,
        "pair_results": pair_results,
    }


if __name__ == "__main__":
    run_step("01_baseline", body)
