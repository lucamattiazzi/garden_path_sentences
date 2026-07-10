"""Step 02 — per-layer Δ-surprisal ablation sweep.

For each of the 36 transformer layers, ablate ALL heads and measure the
change in Δ at the disambiguator vs baseline. Layer 0 is expected to be a
false-positive (breaks the model globally — high control inflation). The
output ranking points at the GPS-specific layers.
"""

from _ablation import sweep_layers  # noqa: E402
from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    return sweep_layers(model, tokenizer, device)


if __name__ == "__main__":
    run_step("02_layer_sweep", body)
