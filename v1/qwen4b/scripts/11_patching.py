"""Step 11 — activation patching NORMAL → GPS.

Denoising interchange intervention on the residual stream: patch the
hidden state of the unambiguous (NORMAL) run into the GPS run at the
shared-surface positions (V-amb word, critical word), one layer at a
time, and measure how much of the comprehension gap
(s_vc_gps − s_vc_normal) is recovered.

Unlike the zero-ablation sweeps (steps 02-05), this can localise a
computation even if it is distributed or redundant — the standard tool
this project was missing before making "no circuit" claims.
"""

from _common import run_step  # noqa: E402
from _extra import run_patching_sweep  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import by_template  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    pairs = by_template("NV") + NOVEL_NV_PAIRS
    print(f"Activation patching NORMAL → GPS over {len(pairs)} NV pairs "
          "(canonical + novel):\n")
    return run_patching_sweep(model, tokenizer, pairs, device)


if __name__ == "__main__":
    run_step("11_patching", body)
