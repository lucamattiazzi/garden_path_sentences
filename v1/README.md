# Garden-Path Sentences — Mechanistic Interpretability

This directory contains V1. Run all commands below from `v1/`; paths in its
historical documentation are relative to this directory. See the
[repository overview](../README.md) for V2.

A mechanistic-interpretability investigation into whether a decoder-only LLM does genuine syntactic reanalysis on **garden-path sentences** such as *"The old man the boat"*, and whether the underlying circuit can be located and selectively ablated.

The historical results establish predictive differences and intervention effects;
they do not yet identify syntactic reanalysis. See the [reviewed protocol](research/PROTOCOL.md)
and [numerical audit](research/results/legacy-audit.md) for limitations and the new
NP/S–NP/Z behavioural pilot.

## Layout

```
model_finder/   # Surprisal sweep across candidate models + the two GPS datasets
research/       # Historical audit, reviewed protocol and role-comprehension pilot
qwen4b/         # Full mech-interp pipeline on Qwen3-4B-Base (steps 01-11)
qwen14b/        # Full mech-interp pipeline on Qwen3-14B-Base (steps 01-13)
qwen32b/        # First-round 32B triage (confounded; kept for the record)
```

## Quick start

```bash
uv sync
uv run python qwen4b/scripts/01_baseline.py        # ~1 min
uv run python qwen4b/scripts/02_layer_sweep.py     # ~10 min, biggest
uv run python qwen4b/scripts/03_head_sweep.py 25   # ~3 min
uv run python qwen4b/scripts/04_predict.py
uv run python qwen4b/scripts/05_predict_layer_sweep.py
uv run python qwen4b/scripts/07_lexical_baseline.py
uv run python qwen4b/scripts/11_patching.py        # ~2 min, the most informative step
```
