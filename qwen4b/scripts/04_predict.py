"""Step 04 — comprehension test of the top Δ-surprisal candidates.

Behavioural probe: after the disambiguator has been emitted, does the
model still predict a plausible verb-object? Compares baseline against
ablation of {L25.H7}, {L35.H0}, and their combination.

Crucial step in the project: confirms or denies that the heads identified
in steps 02-03 are *comprehension* heads or merely *surprise* heads.
"""

from _ablation import sweep_predict  # noqa: E402
from _common import run_step  # noqa: E402

_TARGETS: list[list[tuple[int, int]]] = [
    [(25, 7)],
    [(35, 0)],
    [(25, 7), (35, 0)],
]


def body(model, tokenizer, device: str) -> dict:
    return sweep_predict(model, tokenizer, device, _TARGETS)


if __name__ == "__main__":
    run_step("04_predict", body)
