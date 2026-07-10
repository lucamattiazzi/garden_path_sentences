"""Step 08 — memorisation-vs-parsing diagnostic on Qwen3-32B-MLX-8bit.

Same probe as step 07 but on the 10 NOVEL NV pairs
(`model_finder/novel_pairs.py`) whose GPS sentences do not appear as
textbook examples (see that module's docstring for what "novel" does and
does not mean).

Read against step 07:

    step 07 (10 canonical NV pairs)   full − stripped = -0.559
    step 08 (10 novel NV pairs)       full − stripped =   ?

- Δ ≈ 0 → the step-07 signal was mostly training-set recall; no genuine
  parsing capability at 32B scale
- Δ ≈ step-07 or more negative → post-disambiguator comprehension really
  does improve with scale, independent of memorisation
- intermediate → partial emergence
"""

from _common import run_step  # noqa: E402
from _lib import measure_lexical_baseline  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402


def body(model, tokenizer) -> dict:
    print("Novel-pair lexical baseline probe — separating memorisation from parsing.")
    print("Compare aggregate `full - stripped` here to step 07 on canonical pairs.\n")

    summary, pair_results = measure_lexical_baseline(model, tokenizer, NOVEL_NV_PAIRS)

    print(f"  Means across {len(pair_results)} NOVEL NV pairs:")
    print(f"    s_full     = {summary['mean_s_full']:.3f}    (full GPS prefix)")
    print(f"    s_stripped = {summary['mean_s_stripped']:.3f}    (V-amb-word + critical_word only)")
    print(f"    s_anchor   = {summary['mean_s_anchor']:.3f}    (unrelated NP ending in critical_word)")
    print()
    print(f"    full vs stripped  Δ = {summary['mean_s_full'] - summary['mean_s_stripped']:+.3f}")
    print(f"    full vs anchor    Δ = {summary['mean_s_full'] - summary['mean_s_anchor']:+.3f}")
    print()
    print("  Per-pair breakdown:")
    header = (
        f"    {'id':>3}  {'v_amb':<10}  {'vc':<12}  "
        f"{'s_full':>7}  {'s_strip':>7}  {'s_anch':>7}  "
        f"{'full-strip':>10}  {'full-anch':>10}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in pair_results:
        print(
            f"    {r['id']:>3}  {r['v_amb_word']:<10}  {r['verb_continuation']:<12}  "
            f"{r['s_full']:>7.3f}  {r['s_stripped']:>7.3f}  {r['s_anchor']:>7.3f}  "
            f"{r['s_full'] - r['s_stripped']:>+10.3f}  "
            f"{r['s_full'] - r['s_anchor']:>+10.3f}"
        )

    return {"summary": summary, "pair_results": pair_results}


if __name__ == "__main__":
    run_step("08_novel_lexical_baseline", body)
