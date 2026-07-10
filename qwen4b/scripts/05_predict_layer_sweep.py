"""Step 05 — per-layer ablation sweep, comprehension metric.

Mirror of step 02 but using `measure_verb_continuation` (post-disambiguator
P(verb_object)) as the metric instead of Δ at the disambiguator. Hunts the
layers that actually contribute to verb-parse reanalysis.

Selectivity = vc_gps_inflation − control_inflation.
"""

from _ablation import sweep_predict_layers  # noqa: E402
from _common import run_step  # noqa: E402


def body(model, tokenizer, device: str) -> dict:
    return sweep_predict_layers(model, tokenizer, device)


if __name__ == "__main__":
    run_step("05_predict_layer_sweep", body)
