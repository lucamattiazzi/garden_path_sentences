"""Step 05 — per-layer ablation sweep, comprehension metric, gemma-3-27b-pt."""

from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    from _ablation import sweep_predict_layers

    return sweep_predict_layers(model, tokenizer, device)


if __name__ == "__main__":
    run_step("05_predict_layer_sweep", body)
