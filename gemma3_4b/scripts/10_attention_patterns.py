"""Step 10 — attention patterns at the disambiguator (4B).

Direct test of the "retrospective read" hypothesis: per (layer, head)
attention from the disambiguator back to the V-amb word, GPS vs NORMAL,
canonical + novel pairs. Loads the model itself (eager attention needed).
"""

import time

from _common import MODEL_SPEC, save_json, tee_to  # noqa: E402

from model_finder.novel_pairs import NOVEL_NV_PAIRS  # noqa: E402
from model_finder.sentences import by_template  # noqa: E402
from model_finder.utils import cleanup, get_device  # noqa: E402

STEP = "10_attention_patterns"


def main() -> None:
    from _extra import load_model_eager, run_attention_analysis

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
