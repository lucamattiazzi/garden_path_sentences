"""Step 02 — per-layer Δ-surprisal ablation sweep on gemma-3-1b-pt."""

from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from _ablation import sweep_layers

    return sweep_layers(model, tokenizer, device)


if __name__ == "__main__":
    run_step("02_layer_sweep", body)
