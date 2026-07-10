"""Step 06 — per-head sweep within a layer, comprehension metric.

The direct hunt for a "reanalysis head": ablate each head of a layer chosen
by step 05 (top vc_gps_inflation) and measure the verb-continuation
surprisal. A head whose ablation inflates vc_gps — but not vc_normal or the
controls — is a candidate reanalysis head, selected on the comprehension
metric itself rather than via the Δ-surprisal track (steps 02-03), which is
biased toward prior-encoding heads.

Usage:
    uv run python gemma3_27b/scripts/06_predict_head_sweep.py <LAYER>
"""

import sys

from _common import run_step  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 06_predict_head_sweep.py <LAYER>")
    layer = int(sys.argv[1])

    def body(model, tokenizer, device: str) -> dict:
        from _ablation import sweep_predict_heads_in_layer

        return sweep_predict_heads_in_layer(model, tokenizer, device, layer)

    run_step(f"06_predict_head_sweep_L{layer}", body)


if __name__ == "__main__":
    main()
