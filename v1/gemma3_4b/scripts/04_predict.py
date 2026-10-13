"""Step 04 — comprehension test of the top Δ-surprisal candidates.

The candidate heads come from this model's own steps 02-03, so they are
passed on the command line instead of being hard-coded:

    uv run python gemma3_4b/scripts/04_predict.py 25:7 35:0

Each L:H argument becomes a single-head condition; the union of all of
them is added as a combined condition.
"""

import sys

from _common import run_step  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: 04_predict.py L:H [L:H ...]")
    singles = []
    for arg in sys.argv[1:]:
        li, h = arg.split(":")
        singles.append((int(li), int(h)))
    target_sets = [[t] for t in singles]
    if len(singles) > 1:
        target_sets.append(singles)

    def body(model, tokenizer, device: str) -> dict:
        from _ablation import sweep_predict

        return sweep_predict(model, tokenizer, device, target_sets)

    run_step("04_predict", body)


if __name__ == "__main__":
    main()
