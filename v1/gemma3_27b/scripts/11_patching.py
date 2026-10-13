"""Step 11 — activation patching NORMAL → GPS (27B).

Denoising interchange intervention: patch the unambiguous run's hidden
state into the GPS run at the shared positions, per layer, and measure
recovery of the comprehension gap. Localises the computation even if it
is distributed or redundant.
"""

from _common import run_step  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import by_template  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from _extra import run_patching_sweep

    pairs = by_template("NV") + NOVEL_NV_PAIRS
    print(f"Activation patching NORMAL → GPS over {len(pairs)} NV pairs "
          "(canonical + novel):\n")
    return run_patching_sweep(model, tokenizer, pairs, device)


if __name__ == "__main__":
    run_step("11_patching", body)
