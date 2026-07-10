"""Step 09 — verb-continuation baseline (with CIs), canonical vs novel.

`vc_delta = s_vc_gps − s_vc_normal` is the residual comprehension cost
after the disambiguator — the metric that actually distinguishes
"successful reanalysis" (vc_delta → 0) from "commitment to the wrong
parse" (vc_delta large). Step 07/08's `full − stripped` cannot make that
distinction; this can, and it was never measured on the novel pairs.
"""

from _common import run_step  # noqa: E402
from _extra import report_verb_continuation  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import PAIRS  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    print("Verb-continuation baseline: residual comprehension cost, canonical vs novel.\n")
    canonical = report_verb_continuation(model, tokenizer, PAIRS, device, "CANONICAL NV pairs")
    novel = report_verb_continuation(model, tokenizer, NOVEL_NV_PAIRS, device, "NOVEL NV pairs")
    return {"canonical": canonical, "novel": novel}


if __name__ == "__main__":
    run_step("09_novel_predict", body)
