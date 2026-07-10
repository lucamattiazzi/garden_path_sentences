"""Step 08 — memorisation-vs-parsing diagnostic on the NOVEL NV pairs (12B)."""

from _common import run_step  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from _extra import report_lexical_baseline

    print("Novel-pair lexical baseline probe — separating memorisation from parsing.\n")
    return report_lexical_baseline(model, tokenizer, NOVEL_NV_PAIRS, device, "NOVEL NV pairs")


if __name__ == "__main__":
    run_step("08_novel_lexical_baseline", body)
