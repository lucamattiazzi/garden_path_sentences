"""Step 06 — per-head sweep within a layer, comprehension metric.

NOT EXECUTED for Qwen3-4B-Base: step 05 produced no layer with both
significant comprehension impact AND clean selectivity, so drilling into a
single layer was not warranted. Script is kept in place because the same
investigation will be re-run as soon as a larger model surfaces a viable
target.

Usage:
    uv run python qwen4b/scripts/06_predict_head_sweep.py <LAYER>
"""

import sys

from _ablation import sweep_predict_heads_in_layer  # noqa: E402
from _common import run_step  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 06_predict_head_sweep.py <LAYER>")
    layer = int(sys.argv[1])

    def body(model, tokenizer, device: str) -> dict:
        return sweep_predict_heads_in_layer(model, tokenizer, device, layer)

    run_step(f"06_predict_head_sweep_L{layer}", body)


if __name__ == "__main__":
    main()
