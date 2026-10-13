# Gemma 3 12B direct-answer causal bridge

Frozen after 12B passed the direct-answer development gate and before any
measurement of this bridge. No held-out behavior has been inspected during
checkpoint selection. This protocol is conditional on a new contextual competence
gate; a pass on simple clauses does not alone validate garden-path QA.

## Fixed model and material

Pretrained google/gemma-3-12b-pt, revision
295efb63d01a7017928f273a94ebb86105c9526f. Full FP32, TF32 off, eager attention,
48 decoder blocks, seed 0, Torch 2.10.0 / Transformers 5.9.0. Full multimodal
wrapper with text input only, exact checkpoint-loading checks. No layer search.

Use existing twelve synthetic frames and the existing six-development / six-test
optional-verb-family split. Four direct-name demonstrations, same questions and
answer suffixes as the development instrument. Passive main clauses only: this
matches the was/comma predictive endpoint. Two participant reversals and two
queried roles. No active subset or item is selected based on new measurements.

Conditions: simple main clause, native no-comma, no-comma with a position-ID gap,
and comma. Gap shifts IDs by one from the second NP through the answer. Clean and
gap must share identical token IDs and position IDs after removing the actual
sentence comma. These are new synthetic items, not human-validated NP/Z materials.

## Mandatory gate before test or any patch

On development, require >=75% of families correct under BOTH participant reversals
for each queried role in BOTH simple and comma conditions (four strata; five of
six pairs each). The full-answer log margin must be strictly positive; ties fail.
If the gate fails, stop with 96 development baselines, zero interventions, no test
inference. Do not change prompt, threshold or layer after observing this gate.

## Intervention

Use fixed block 32, zero-based post-block residual. This is a prespecified relative
depth transfer: round((17/25)*(48-1)) = 32 from V3's 26-block checkpoint. It is not
claimed to be a homologous layer or a best layer in 12B.

Patch the last token of the second NP before `was`. Capture full residual donor
states in the complete teacher-forced prompt shape. Verify invariance of the noun
state and earlier next-token logits across the two future candidate answers.
Directions: clean → gap and gap → clean. Controls: identity, norm-matched embedded
actor replacement, and three seeded norm-matched Gaussian displacements. Gaussian
vectors are shared across queried roles within item/reversal/direction/seed.
Record realized norms. Identity effects above 1e-4 abort the run.

Score BOTH endpoints in the same forward passes: (1) full correct-minus-incorrect
answer sequence log probability after `Answer: the`, including noun and period;
(2) log P(was) − log P(comma) at the pre-disambiguation noun. Validate token-stable
boundaries and single-token predictive alternatives. No per-token normalization.

## Analysis and claims

Primary: held-out QA-margin effect of cross patch, averaged equally over random
seeds within controls, both roles, reversals, and sign-aligned directions within
family, then equally across six families. Negate reverse-direction effects so
positive means a shift toward comma-condition behavior. Report raw cross and
paired cross-minus-lexical / cross-minus-random effects for both endpoints; report
each direction separately as well. Do not normalize by baseline effect size.

Descriptive bootstrap 95% intervals over six family means: 10,000 resamples,
seed 0. Roles, reversals, directions and Gaussian seeds are not independent units.
No outcome-based filtering. Save per-family values and raw individual measurements.
Report native/gap/comma behavior and baseline positional differences separately.

This measures a comma-cue intervention, NOT V3's optional-minus-intransitive
factorial interaction. There is no intransitive control in this bridge. A positive
QA effect would connect a residual intervention to this readout, not establish
abstract role representations or garden-path reanalysis. A QA null may reflect
redundant information, subsequent recomputation, or an ineffective selected layer.
A high unpatched accuracy can coexist with a predictive garden-path preference.

## Size and execution

Passing gate: 192 baselines and 576 intervention observations (each scores two
candidate forwards), plus lexical donor captures. Gate failure: 96 baselines only.
Keep development and held-out test separate. Snapshot sources and materials before
loading; save raw prompts, token/position IDs, per-token answer scores and effects.
One A100-80GB, 80 GiB CPU RAM, 20-minute ceiling, no automatic retries. Existing
output IDs are immutable. All previous frozen source bundles remain untouched.
