# Gemma 3 12B causal bridge

**Completed on Modal:** [Italian HTML report and figures](results/bridge12-20261009-01/report.html)
([Markdown](results/bridge12-20261009-01/report.md)), run `bridge12-20261009-01`.

On six held-out families, the sign-aligned cross patch changes the predictive
`was`/comma margin by **8.069 nats [5.338, 10.582]**, exceeding norm-matched lexical
and random controls by 7.952 and 7.544 nats respectively. The mean correct-answer
margin changes by **−0.074 [−0.132, −0.021]**: this is not a zero QA effect.

A post-hoc role breakdown reveals opposing shifts: **−0.463** for agent questions
and **+0.315** for patient questions. **None of the 48 held-out cross interventions
changes the preferred answer.** In passive clauses this pattern is compatible with
a bias toward the grammatical subject/patient across both question types. It is
not evidence for selectively transferring the requested semantic role; paired
controls and six-family uncertainty limit that interpretation.

The contextual development gate passed (83.3% both-reversal correct for comma/agent,
100% in the other three strata). On held-out no-comma/gap sentences, final QA can
remain accurate despite a strong pre-verb preference against `was`. Predictive
preference, answer confidence and answer choice are distinct endpoints.

192 baselines, 576 interventions; 389.2 seconds inside one A100-80GB function,
45.8 GiB peak GPU memory. Identity errors: zero. Maximum control-norm relative
error: 8.56e-7. Source, chronology, gate and raw-effect audits passed. The Modal app
is stopped. The report includes descriptive family-bootstrap intervals, both
directions, controls and the explicitly exploratory role breakdown.

This is the causal follow-up to [larger-model competence validation](../scale/README.md).
The 12B checkpoint passed all simple-clause development questions. This experiment
adds a separate contextual competence gate before any causal patch or test inference.
Read the frozen [protocol](PROTOCOL.md) for the stopping and analysis rules.

The primary endpoint is the intervention's effect on correct-versus-incorrect
full direct-answer log probability. The same forwards also score `was` versus
comma at the pre-disambiguation noun. Passive sentences, both role questions and
participant reversals; clean/gap positional alignment; both patch directions;
identity, norm-matched lexical controls and three norm-matched Gaussian directions.

Block 32 is fixed by relative depth from V3's 1B block 17, not selected from these
outcomes. This is a comma-cue effect, not the old optional-minus-intransitive
factorial interaction. No claim of abstract roles or garden-path reanalysis follows
from a predictive effect alone. Six test families limit the precision of inference.

## Run and audit

From the repository root, using the existing Modal profile and Hugging Face secret:

```bash
uvx --from modal modal run v3/bridge12/modal_app.py --run-id fresh-bridge12-id
uvx --from modal modal run v3/bridge12/modal_app.py --run-id bridge12-20261009-01 --fetch
uv run --no-project --with numpy --with matplotlib --with markdown python v3/bridge12/analyze.py v3/bridge12/results/bridge12-20261009-01
```

One A100-80GB, 80 GiB CPU RAM, full FP32, TF32 disabled, eager attention,
20-minute ceiling, no automatic retries. Unique output directories; exact pinned
checkpoint; source/material snapshots before loading. No prior scientific source
bundle is edited. The analyzer independently checks the competence gate, chronology,
per-token answer sums, intervention deltas, family/control aggregation and norms.
