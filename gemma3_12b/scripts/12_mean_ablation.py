"""Step 12 — mean-ablation replication of the key zero-ablation results.

Zero-ablation pushes activations off-distribution; mean ablation replaces
each target head's output with its dataset mean. If a zero-ablation
finding does not survive mean ablation, it was an artifact of the
off-distribution intervention, not a property of the head.

Targets come from this model's own steps 03/05:

    uv run python gemma3_12b/scripts/12_mean_ablation.py 25:7 35:0 -- 8 35

Arguments before `--` are L:H head targets (each also combined), after it
are full layers.
"""

import sys

from _common import run_step  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    if "--" in args:
        split = args.index("--")
        head_args, layer_args = args[:split], args[split + 1:]
    else:
        head_args, layer_args = args, []
    if not head_args and not layer_args:
        raise SystemExit("usage: 12_mean_ablation.py L:H [L:H ...] [-- LAYER ...]")

    singles = []
    for arg in head_args:
        li, h = arg.split(":")
        singles.append((int(li), int(h)))
    head_sets = [[t] for t in singles]
    if len(singles) > 1:
        head_sets.append(singles)
    layers = [int(a) for a in layer_args]

    def body(model, tokenizer, device: str) -> dict:
        from _extra import run_mean_ablation

        return run_mean_ablation(model, tokenizer, device, head_sets, layers)

    run_step("12_mean_ablation", body)


if __name__ == "__main__":
    main()
