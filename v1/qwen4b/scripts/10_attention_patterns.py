"""Step 10 — attention patterns at the disambiguator.

The project's founding hypothesis (§2.3 of the summary) is a
"retrospective read": at the disambiguator, some attention head queries the
V-ambiguous word's representation. This step measures that attention
directly — per (layer, head), GPS vs NORMAL, canonical + novel pairs —
and correlates it per-pair with the residual comprehension cost.

Needs eager attention (sdpa does not return attention weights), so it
loads the model itself instead of going through run_step.
"""

import time

from _common import MODEL_SPEC, save_json, tee_to  # noqa: E402
from _extra import load_model_eager, run_attention_analysis  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import by_template  # noqa: E402
from model_finder.utils import cleanup, get_device  # noqa: E402

STEP = "10_attention_patterns"


def main() -> None:
    tee_to(STEP)
    print(f"=== {STEP} ===")
    device = get_device()
    print(f"device = {device}")
    print(f"model  = {MODEL_SPEC.hf_name} (dtype={MODEL_SPEC.dtype}, attn=eager)\n")

    model, tokenizer = load_model_eager(device, MODEL_SPEC)
    t0 = time.time()
    try:
        pairs = by_template("NV") + NOVEL_NV_PAIRS
        print(f"Attention disambiguator → V-amb word over {len(pairs)} NV pairs "
              "(canonical + novel):\n")
        results = run_attention_analysis(model, tokenizer, pairs, device)
        save_json(STEP, {"step": STEP, "results": results})
    finally:
        del model, tokenizer
        cleanup(device)
        print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
