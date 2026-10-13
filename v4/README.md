# V4: first mention, queried-event subject, and semantic agent

V4 follows up the role-opposite confidence changes in the V3 12B causal bridge.
The old result alone cannot identify a role representation: its common preference
for the first participant is approximately as large under lexical and random
perturbations as under the comma patch.

The [frozen protocol](PROTOCOL.md) uses twelve new event/participant families,
four constructions and two questions to cross first mention with queried-event
subject and semantic agent. Six families are development; six are fresh
confirmation. The embedded actors/verbs and demonstrations are reused.

A model must answer both participant reversals correctly in at least five of six
development families in every construction/question stratum. Run 12B first, then
27B only if 12B fails this competence gate. Never select on confirmation results.
All inference is FP32, without TF32, at a fixed relative decoder depth.

The relative constructions distinguish subject of the queried event from first
mention; their matrix subject is still the first participant. They are formal,
synthetic and not human-validated. This is a limited instrument, not a general
assay for all grammatical roles.

## Reproduction

From the repository root, using the existing Modal profile and HF secret:

```bash
uvx --from modal modal run v4/modal_app.py --run-id fresh-v4-12b --size 12b
# Only if the development gate fails:
uvx --from modal modal run v4/modal_app.py --run-id fresh-v4-27b --size 27b
# Fetch a completed or interrupted bundle:
uvx --from modal modal run v4/modal_app.py --run-id existing-id --fetch
uv run --no-project --with numpy --with matplotlib python v4/analyze.py v4/results/completed-id --output v4/results/comparison-20261009
```

The Modal app uploads an explicit scientific-source allowlist. Raw measurements,
model revision, source snapshot/hash, material split, package versions, identity
checks, elapsed time and peak GPU allocation are saved per run. The analyzer
recomputes token sums, effects, gate chronology, family contrasts and uncertainty.

## Implementation amendment

`v4-12b-20261009-01` stopped after twelve development baselines on the inherited
1e-5 causal-prefix invariance check. Unequal answer lengths produced different
forward shapes. The V4 measurement module now right-pads candidates to equal
length with masked padding and scores only real tokens; the tolerance is unchanged.
The interrupted bundle remains available. A targeted test verifies equal forward
shapes and unequal scored suffix lengths. No scientific choices changed.

## Results, 2026-10-09

[Offline HTML explanation and figures](results/comparison-20261009/report.html)
· [independent audit and family estimates](results/comparison-20261009/audit.json)

Both models fail the unchanged gate on **passive-relative agent** questions.
No V4 confirmation baselines or causal interventions were run.

| Development stratum | 12B, both reversals | 27B, both reversals |
|---|---:|---:|
| Active agent | 6/6 | 6/6 |
| Active patient | 6/6 | 6/6 |
| Passive agent | 6/6 | 5/6 |
| Passive patient | 6/6 | 6/6 |
| Object-relative agent | 6/6 | 6/6 |
| Object-relative patient | 6/6 | 6/6 |
| Passive-relative agent | **0/6** | **0/6** |
| Passive-relative patient | 5/6 | 6/6 |

Completed bundles: `v4-12b-20261009-02` and `v4-27b-20261009-01`, each with
96 development observations, zero confirmation observations and zero patches.
This limits the instrument; it does not establish absence of role representations.

The exploratory reanalysis of V3's six held-out families gives a common
first-participant effect of 0.389 [0.272, 0.515] for the comma patch, 0.366
[-0.159, 1.047] for lexical controls and 0.392 [0.209, 0.582] for random controls.
Paired comma-minus-control effects are 0.023 [-0.697, 0.586] and -0.003
[-0.234, 0.215]. This provides no evidence of a specific comma advantage for
that common bias; wide intervals do not demonstrate equivalence. These are
post-hoc transforms of the same V3 data, not fresh replications. Question-dependent
effects remain distinct; the original predictive effect is not invalidated.

## Scope after 13 October 2026

Only the original experiment with the garden-path context is retained as a
historical instrument check. Subsequent validation/paraphrase/targeted branches
without that context were removed. They are not the continuation of this project.
The active question is surprisal at disambiguating words and causal mechanisms
for processing the ambiguous sentence; see the repository README.
