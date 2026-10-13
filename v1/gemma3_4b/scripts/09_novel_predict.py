"""Step 09 — verb-continuation baseline (with CIs), canonical vs novel (4B).

vc_delta is the residual comprehension cost after the disambiguator — the
metric that distinguishes successful reanalysis from commitment to the
wrong parse. Compared against the same numbers on the 4B, this is the
clean scale test.
"""

from _common import run_step  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import PAIRS  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from _extra import report_verb_continuation

    print("Verb-continuation baseline: residual comprehension cost, canonical vs novel.\n")
    canonical = report_verb_continuation(model, tokenizer, PAIRS, device, "CANONICAL NV pairs")
    novel = report_verb_continuation(model, tokenizer, NOVEL_NV_PAIRS, device, "NOVEL NV pairs")
    return {"canonical": canonical, "novel": novel}


if __name__ == "__main__":
    run_step("09_novel_predict", body)
