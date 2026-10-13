# V3 behavioral bridge — frozen before inference

## Question and interpretation

Does replacing the pre-disambiguation noun's residual state affect both the
next-token preference and a later question about the main event's agent/patient?
This is a new synthetic, few-shot task on the same checkpoint and layer as V3.
It is not the published V3 material set, a human-validated garden-path test, or a
replication of V3's optional-minus-intransitive factorial interaction.

The primary endpoint is the change in correct-versus-incorrect answer log odds,
averaged across both option orders, roles, reversals and patch directions within
optional-verb family, then equally across held-out families. The secondary
endpoint is the same intervention's effect on log P(was) − log P(comma) before
the main verb. Both are measured in one teacher-forced forward pass.

## Materials and split

Twelve authored frames in `materials.py`; each has three animate participants,
a distinct embedded optional verb, two reversals of the main-clause participants,
active/passive voice and agent/patient questions. Answer options are the two
main-clause participants. Thus role changes require changing the answer despite
an unchanged noun inventory. This rules out a fixed noun rule but not surface
word-order/voice rules; successful QA alone does not establish abstract syntax.

Six development and six test optional-verb families are selected by fixed SHA256
ordering (`gps-bridge:`). Event predicates may recur across splits. No human
validation or training-data exclusion is claimed. Four fixed demonstrations
balance A/B, voice and role, using predicates absent from experimental frames.

Conditions: simple main clause; no sentence; no-comma native; no-comma with a
one-position gap from the second NP onward; comma control. Gap and comma must
have matching common token IDs and position IDs after removal of the comma.
The no-context prompts have identical text under counterfactual gold labels.

## Mandatory development competence gate

Before any test inference or causal intervention, evaluate all development
baselines. Require at least 75% of simple-clause questions to be correct in BOTH
answer orders, separately for active and passive voice; require median P(A)+P(B)
to be at least 0.10 on these simple clauses. Strict positive log odds defines
correctness; ties fail. This is a measurement-instrument gate, not a statistical
significance test. Accuracy means ranking the specified A/B tokens, not sampled
free-form answers. The answer labels must each be a single appended token.

If either requirement fails, stop. Save development results with status
`instrument_gate_failed`; perform no patches and do not inspect test behavior.
Do not lower the threshold or revise demonstrations after seeing this run.
An unsuccessful gate cannot establish absence of role knowledge in the model.

## Fixed causal design if the gate passes

Use pretrained `google/gemma-3-1b-pt`, revision
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, BF16, eager attention, block 17
(zero-based, post-block residual). No layer or model search.

Patch only passive sentences, at the last token of the second NP before `was`.
Capture donor and recipient using the complete QA prompt shape; causal attention
masks later tokens. Check prefix tokenization matches the full-prompt prefix.
Directions: comma → gap and gap → comma. Negate the reverse direction before
averaging, so a positive effect means a shift toward the comma condition.

Controls: identity; replacement of the embedded actor with another human noun;
three seeded Gaussian directions. Match control displacement norms to the cross
patch in BF16 and record realized norms. Use the same random direction for all
question roles and answer orders sharing item, reversal and direction. An identity
error above 1e-4 in either endpoint aborts the run. Entire residual replacement is
not evidence for a role-specific feature. A later QA null can reflect redundant
information or recomputation after the intervention.

Report both directions and their average, raw cross effects and paired advantages
over lexical/random controls. Bootstrap 10,000 resamples of family means with
seed 0 as inherited from `v2.experiment.family_interval`; intervals are descriptive
with only six test families. Random seeds, roles and answer orders are not
independent samples. No ratio normalization or outcome-based filtering.

## Size, provenance and stopping

Completed bridge: 960 baselines, 1,152 interventions. Gate failure: 480 development
baselines, zero interventions, no test inference. Development is kept separate
from test. Source text and hashes, generated materials, package versions, device,
raw prompts, token IDs, position IDs and scores are saved. The source snapshot
precedes model loading. Existing output directories cannot be overwritten.

One Modal A10, 20-minute function ceiling, no automatic retries. `complete=true`
means the specified branch completed, not that the competence gate passed.

## Relation to prior results

A positive result would connect a local predictive intervention to this QA task,
subject to controls. It would not identify a unique syntax layer, prove garden-path
reanalysis, or recover the optional-minus-intransitive interaction of V3. The new
few-shot context prevents direct amplitude comparisons with V3's bare prefixes.
A failed gate means this instrument/checkpoint combination is unsuitable for the
planned causal interpretation; V3's previously measured predictive effects remain.
