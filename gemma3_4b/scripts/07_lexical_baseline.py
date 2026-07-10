"""Step 07 — lexical-floor diagnostic on the canonical NV pairs (4B)."""

from _common import run_step  # noqa: E402

from model_finder.sentences import PAIRS  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from _extra import report_lexical_baseline

    print("Lexical baseline probe on the CANONICAL NV pairs.\n")
    return report_lexical_baseline(model, tokenizer, PAIRS, device, "CANONICAL NV pairs")


if __name__ == "__main__":
    run_step("07_lexical_baseline", body)
