"""Step 13 — random single-head ablation null distribution.

Samples random (layer, head) pairs, zero-ablates each, and measures the
same statistics the project's claims are stated in (NV Δ-drop, vc_gps
inflation, selectivity). The resulting percentiles are the calibration
for every "this effect is noise" / "this effect is real" statement.

    uv run python qwen14b/scripts/13_null_distribution.py [N_SAMPLES]
"""

import sys

from _common import run_step  # noqa: E402


def main() -> None:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    def body(model, tokenizer, device: str) -> dict:
        from _extra import run_null_distribution

        return run_null_distribution(model, tokenizer, device, n_samples=n_samples)

    run_step("13_null_distribution", body)


if __name__ == "__main__":
    main()
