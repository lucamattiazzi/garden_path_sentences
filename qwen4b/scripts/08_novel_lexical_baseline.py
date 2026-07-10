"""Step 08 — memorisation-vs-parsing diagnostic on the NOVEL NV pairs.

Same probe as step 07 but on the 10 novel pairs in
`model_finder/novel_pairs.py`. This is the 4B cell of the 2×2 design
(model scale × canonical/novel) — without it the cross-scale comparison
of `full − stripped` has no same-model reference point.
"""

from _ablation import measure_lexical_baseline  # noqa: E402
from _common import run_step  # noqa: E402
from _extra import report_lexical_baseline  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import PAIRS  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    print("Novel-pair lexical baseline probe — separating memorisation from parsing.\n")
    novel = report_lexical_baseline(model, tokenizer, NOVEL_NV_PAIRS, device, "NOVEL NV pairs")
    print("  Canonical reference (same probe, step-07 pairs):")
    canon_summary, _ = measure_lexical_baseline(model, tokenizer, PAIRS, device)
    print(
        f"    canonical full − stripped = "
        f"{canon_summary['mean_s_full'] - canon_summary['mean_s_stripped']:+.3f}"
    )
    return {"novel": novel, "canonical_summary": canon_summary}


if __name__ == "__main__":
    run_step("08_novel_lexical_baseline", body)
