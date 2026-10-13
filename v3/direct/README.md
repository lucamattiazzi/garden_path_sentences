# Direct participant answers — development validation

[Italian HTML report and plots](results/direct-v3-20261008-01/report.html)
([Markdown](results/direct-v3-20261008-01/report.md)). Run: `direct-v3-20261008-01`.

The direct-name format still **fails the predeclared competence gate** on Gemma 3
1B pretrained in FP32. Accuracy is 75% for both queried roles in active sentences,
but 33.3% for agents and 83.3% for patients in passive sentences. Correctness under
both participant reversals is respectively 50%, 50%, 0%, and 66.7%; all four strata
had to reach 75%. Only six development families were used.

A post-hoc diagnostic finds the same preferred name for agent and patient questions
in 10/12 passive sentences (6/12 active). This describes behavior, not its mechanism.
There is useful sentence sensitivity in parts of the task, but not robust role
comprehension sufficient for the planned causal bridge.

All candidate answers have exactly two appended tokens: the noun and the period.
No unequal-length candidates or BF16 ties explain this result. Candidate scores
are full sequence log probabilities after the shared `Answer: the` prefix, not
free-generation answers. The two candidates' conditional probability mass has a
95.9% median on simple clauses. No-context reversal controls passed the audit.

## Scope and next decision

The [protocol](PROTOCOL.md) was frozen before inference, after the earlier A/B
failure. One prompt format, no model selection, no held-out inference, no patches.
This is adaptive development evidence, not confirmation. The comparison with A/B
changes several aspects of the task, so it is not an isolated causal comparison
of answer formats. V3's prior predictive findings remain separate and unchanged.

Recommendation: stop prompt iteration on this 1B checkpoint for this bridge.
Validate competence on a more capable checkpoint under a new frozen development
protocol before causal work. Do not lower the threshold or select only the easy
active/patient subsets after observing these results.

96 observations / 192 scoring forwards, one Modal A10G, 15.2 seconds inside the
GPU function (provisioning excluded). The app completed; raw measurements,
source snapshots and independent audit are saved. Existing scientific sources
from V1/V2/V3 and the previous bridge were preserved.

The [larger-model follow-up](../scale/README.md) has now tested 4B and 12B with
identical materials. The 12B checkpoint passes all four development strata; this
qualifies it for a separately frozen causal experiment, not a retrospective
reinterpretation of the 1B measurements.

## Reproduce

From the repository root, with the existing Modal profile and Hugging Face secret:

```bash
uvx --from modal modal run v3/direct/modal_app.py --run-id fresh-direct-id
uvx --from modal modal run v3/direct/modal_app.py --run-id direct-v3-20261008-01 --fetch
uv run --no-project --with numpy --with matplotlib --with markdown python v3/direct/analyze.py v3/direct/results/direct-v3-20261008-01
```

One A10G, ten-minute ceiling, no automatic retries; use a fresh ID for inference.
The analyzer recomputes token sums, gold margins, family pairs and the gate, and
checks identical no-context scores under opposite gold labels. Tests cover scoring
alignment, counterfactual labels, reversal completeness, candidate-order invariance
on a tiny causal model and tamper detection in saved scores.
