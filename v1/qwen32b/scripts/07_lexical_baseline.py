"""Step 07 — lexical-floor diagnostic on Qwen3-32B-MLX-8bit.

The decisive triage test. On the 4B we found:
    s_full ≈ s_stripped (Δ = +0.099)  →  global NP setup does NOT help
                                          beyond local V-amb-word bigram
plus heavy per-pair variance dominated by memorisation vs n-gram cheating.

If the 32B closes the per-pair variance (consistent negative full-strip
across all 10 NV pairs), scale brings a real parsing capability and the
full ablation pipeline becomes worth running at this size. If the same
mosaic shows up, the conclusion that "GPS comprehension = lexical hacks"
generalises to the 30B-class.
"""

from _common import run_step  # noqa: E402
from _lib import measure_lexical_baseline  # noqa: E402

from model_finder.sentences import PAIRS  # noqa: E402


def body(model, tokenizer) -> dict:
    print("Lexical baseline probe — is verb_continuation prediction driven by")
    print("the full GPS context or by local lexical association?\n")
    summary, pair_results = measure_lexical_baseline(model, tokenizer, PAIRS)

    print(f"  Means across {len(pair_results)} NV pairs:")
    print(f"    s_full     = {summary['mean_s_full']:.3f}    (full GPS prefix)")
    print(f"    s_stripped = {summary['mean_s_stripped']:.3f}    (V-amb-word + critical_word only)")
    print(f"    s_anchor   = {summary['mean_s_anchor']:.3f}    (unrelated NP ending in critical_word)")
    print()
    print(f"    full vs stripped  Δ = {summary['mean_s_full'] - summary['mean_s_stripped']:+.3f}")
    print(f"    full vs anchor    Δ = {summary['mean_s_full'] - summary['mean_s_anchor']:+.3f}")
    print()
    print("  Per-pair breakdown:")
    header = (
        f"    {'id':>3}  {'v_amb':<10}  {'vc':<10}  "
        f"{'s_full':>7}  {'s_strip':>7}  {'s_anch':>7}  "
        f"{'full-strip':>10}  {'full-anch':>10}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in pair_results:
        print(
            f"    {r['id']:>3}  {r['v_amb_word']:<10}  {r['verb_continuation']:<10}  "
            f"{r['s_full']:>7.3f}  {r['s_stripped']:>7.3f}  {r['s_anchor']:>7.3f}  "
            f"{r['s_full'] - r['s_stripped']:>+10.3f}  "
            f"{r['s_full'] - r['s_anchor']:>+10.3f}"
        )

    return {"summary": summary, "pair_results": pair_results}


if __name__ == "__main__":
    run_step("07_lexical_baseline", body)
