"""Step 14 — random single-head null for the comprehension (vc) metric.

Calibrates the step-06 claim: the best head found inside the step-05 layers
is a max over K swept heads, so it must beat the null max-of-K threshold,
not the single-head percentile. Cheap probe (NV pairs + controls only),
hence many more samples than step 13.

    uv run python qwen4b/scripts/14_null_distribution_vc.py [N_SAMPLES]
"""

import sys

from _common import run_step  # noqa: E402


def main() -> None:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    def body(model, tokenizer, device: str) -> dict:
        from _extra import run_null_distribution_vc

        return run_null_distribution_vc(model, tokenizer, device, n_samples=n_samples)

    run_step("14_null_distribution_vc", body)


if __name__ == "__main__":
    main()
