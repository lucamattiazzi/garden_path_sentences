"""Step 03 — per-head Δ-surprisal sweep within a single layer.

Run AFTER step 02 on the layers with the largest GPS-specific drop.
Usage:
    uv run python gemma3_27b/scripts/03_head_sweep.py <LAYER>
"""

import sys

from _common import run_step  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 03_head_sweep.py <LAYER>")
    layer = int(sys.argv[1])

    def body(model, tokenizer, device: str) -> dict:
        from _ablation import sweep_heads_in_layer

        return sweep_heads_in_layer(model, tokenizer, device, layer)

    run_step(f"03_head_sweep_L{layer}", body)


if __name__ == "__main__":
    main()
