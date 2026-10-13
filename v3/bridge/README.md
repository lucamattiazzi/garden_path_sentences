# V3 behavioral bridge: a failed measurement gate

[Italian offline report and figures](results/bridge-v3-20261008-02/report.html)
([Markdown](results/bridge-v3-20261008-02/report.md)).

We tried to connect the pre-disambiguation intervention to later agent/patient
questions on new synthetic sentences. The **Gemma 3 1B pretrained + fixed four-shot
A/B format did not pass the development competence gate**, even on simple clauses.
The planned causal patches and held-out test inference were therefore not run.

| Simple development clauses | Active | Passive |
|---|---:|---:|
| BF16, correct in both option orders | 8.3% | 0.0% |
| FP32 diagnostic, correct in both orders | 33.3% | 33.3% |
| Required threshold, per voice | 75% | 75% |

BF16 produced 46 exact A/B ties in 96 simple-clause measurements. They count as
errors under the frozen rule. The 25% strict accuracy must not be interpreted as
ordinary below-chance behavior: with half credit for ties it is 49.0% overall.
A separate post-hoc full-FP32 diagnostic (TF32 disabled) eliminated these ties but
still achieved only 47.9% active / 50.0% passive accuracy per option order.
The probability mass on A/B was about 84%, so low answer-format mass alone does
not explain the failed gate.

This limits this measurement setup. It does not show that Gemma lacks role
representations, and it does not invalidate V3's prior predictive measurements.
The next useful step is to validate a different readout or a more capable model
on development before spending compute on causal patches. Preserve the test set.

The [direct-name follow-up](../direct/README.md) has now been completed on
development only. It also fails its gate, with particularly weak passive-agent
answers; no new test or causal inference was performed.

## Frozen scope and provenance

- [Original protocol](PROTOCOL.md): twelve synthetic verb families, six development
  and six test; participant reversals, active/passive, two question roles and two
  option orders. Layer 17 fixed, passive-only causal plan, position alignment and
  norm-matched controls. No human validation or intransitive factorial control.
- `bridge-v3-20261008-02`: **480 development baselines**, no interventions/test
  inference, 22.2 seconds in the GPU function, one A10.
- [Numerical diagnostic protocol](PRECISION.md): `bridge-precision-20261008-01`,
  **192 development-only measurements**, full FP32, 14.3 GPU-function seconds.
- Source snapshots and raw data are saved for both; independent audits passed.
  These durations exclude provisioning. Both completed Modal apps are stopped.
- Initial app `ap-jOB0kNP7eQuzQUJje7j8Yh` failed during import before model loading
  because Modal relocated the entrypoint; the path handling was corrected before
  the measured run. No scientific result came from that attempt.

## Reproduce

Run from the repository root. Use a fresh run ID; existing results are immutable.
The numerical diagnostic entrypoint has fixed parent/destination IDs documenting
this specific comparison and refuses to overwrite its destination.

```bash
uvx --from modal modal run v3/bridge/modal_app.py --run-id fresh-bridge-id
uvx --from modal modal run v3/bridge/modal_app.py --run-id bridge-v3-20261008-02 --fetch
uv run --no-project --with numpy --with matplotlib --with markdown python v3/bridge/analyze.py v3/bridge/results/bridge-v3-20261008-02
```

The original failed branch is complete as specified; `metadata.complete` and
`results.gate.passed` intentionally mean different things. The main report also
audits and incorporates the saved FP32 diagnostic when its directory is present.
No existing V1/V2/V3 scientific source was changed by this extension.
