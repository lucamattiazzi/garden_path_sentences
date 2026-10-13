# V4: first mention, event subject, and semantic agent

Frozen before any V4 model inference on 2026-10-09. Motivated by post-hoc V3
12B analysis, not an independent replication of that analysis.

## Materials and model selection

Twelve new event/participant families, sorted by SHA256 `gps-v4:` + event;
first six development, remaining six confirmation. Embedded verbs/actors and
four demonstrations are reused. No human validation. Two participant reversals,
two questions (agent/patient), four constructions sharing the prefix through
first participant:

- active: While the guard watched[,] the surgeon assisted the dentist.
- passive: While the guard watched[,] the surgeon was assisted by the dentist.
- object_relative: While the guard watched[,] the surgeon whom the dentist assisted smiled.
- passive_relative: While the guard watched[,] the surgeon by whom the dentist was assisted smiled.

First-participant agent signs: +, -, -, +. Queried-event subject signs: +, +, -, -.
The queried event is `assisted`: the relative constructions still have the first
participant as matrix subject of `smiled`. This design does not separate first
mention from every grammatical subject. Construction complexity remains a confound.

Use pinned Gemma3 12B pretrained, then only if its development gate fails, pinned
27B pretrained. Stop at first passing model or after 27B failure. Full FP32, no
TF32, eager attention, fixed layer round(17/25 * (decoder_blocks - 1)). No layer,
prompt, or threshold search. Revisions in run.py. One GPU at a time, 30 minute
limit per run, no automatic retries. Infrastructure failures may be repaired and
rerun but must be disclosed.

Development gate: clean comma condition only, at least 75% of six families
correct under BOTH participant reversals in EACH construction x question (eight
strata, at least 5/6). Ties fail. Failing gate stops after 96 baselines without
confirmation inference or any interventions. No confirmation-driven model choice.

## Confirmation and controls

For a passing model only: six confirmation families, clean comma and comma-less
gap-position conditions. Remove the comma token while preserving subsequent
position IDs; assert exact token/position alignment. Patch the first participant's
last token, captured during the complete question-answer forward, in both
directions. Score complete participant name + period continuations after
`Answer: the`, using the same four fixed demonstrations as V3.

Four patch kinds: cross-condition, identity, lexical actor replacement matched
to cross L2 norm, and one norm-matched isotropic random direction. Random seed is
SHA256(`v4:{id}:{reversal}:{direction}`) first eight bytes, shared across questions
and constructions. This is one fixed random realization per item/reversal/direction;
intervals do not include uncertainty over random seeds. Identity tolerance 1e-4.
Successful run: 288 baselines and 768 interventions. No development interventions.

## Estimands and uncertainty

Let E_agent and E_patient be patch-induced changes in first-participant versus
second-participant answer log probability, for the two question types. Reverse
sign for gap-to-clean, average directions and participant reversals within family.
Pair question types before bootstrap. C=(E_agent+E_patient)/2 measures common
first-participant preference; D=(E_agent-E_patient)/2 measures question dependence.

Prespecified family-level contrasts:

- first_mention: mean C across constructions;
- event_subject: mean C times (+,+,-,-);
- semantic_agent: mean C times (+,-,-,+);
- construction_interaction: mean C times (+,-,+,-);
- role_selective: mean D times (+,-,-,+), equal to the mean signed correct-answer margin effect.

Report cross, lexical, random and paired cross-minus-control estimates for each.
Bootstrap six families, 10,000 replicates, seed 0, percentile 95% intervals.
No exclusions based on outcome. Report every contrast, no multiplicity-adjusted
binary discovery claims. Also report preferred-answer flips and baseline accuracy.
The was/comma logit contrast is ancillary and counterfactual for relative clauses:
it is not their actual next-token continuation or a primary reanalysis endpoint.

A question-dependent effect is not by itself proof of an abstract role circuit.
Null, failed-competence, and non-specific-control outcomes are reportable endpoints.
The old V3 common-bias decomposition remains explicitly exploratory/post-hoc.

## Numerical implementation amendment before completed development (2026-10-09)

Run v4-12b-20261009-01 stopped on the unchanged 1e-5 causal-prefix logit
invariance assertion during development. V4 now right-pads candidate forwards
to the same tensor length with attention-mask zero, trims logits to the actual
sequence before scoring, and retains the original invariance tolerance. This
addresses different GEMM shapes for unequal tokenized answer lengths. No prompts,
materials, gate thresholds, model selection rules or statistical contrasts changed.
The interrupted run is retained with its original source snapshot and metadata.
