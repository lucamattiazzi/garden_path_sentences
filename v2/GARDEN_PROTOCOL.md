# Gemma 3 NP/Z follow-up: frozen exploratory protocol

This protocol and the twelve materials in `garden_followup.py` are fixed before
the follow-up inference. It is an exploratory synthetic assay, not a preregistered
human-validated study. Six new optional-transitive verb families are development
items; six disjoint families are a held-out test of this particular assay. Test
data become development evidence after this run and must not be reused as fresh
confirmation of subsequent design changes.

## Question and measurement

Does a pre-disambiguation residual patch transmit a continuation preference that
depends on whether the preceding verb permits an object, beyond a punctuation
effect shared with strongly intransitive controls?

Each family crosses embedded-verb class with the presence of a comma:

| Verb class | No comma | Comma |
| --- | --- | --- |
| Optional transitive | While the artist painted the colorful canvas | While the artist painted, the colorful canvas |
| Intransitive control | While the artist sneezed the colorful canvas | While the artist sneezed, the colorful canvas |

The identical candidate completions follow all four prefixes:

- Main-clause reading: `was beside the table.`
- Object reading: `beside the table, the assistant waited.`

The second completion is grammatical under the optional-transitive, no-comma
object reading. It is intentionally incompatible with the comma and intransitive
readings. Both candidates describe material near a table, but their events and
lengths differ. Define the raw margin as summed log P(main completion) minus
summed log P(object completion). It is a contrast between these two strings,
**not a normalized probability of a parse**. Token counts and both raw log
probabilities are retained. Separately report the factorial interaction in
log P(`was`), which measures the immediate finite-verb prediction. Full-completion
scores include later teacher-forced tokens and must not be called a pure measure
of processing at the first disambiguator.

For either baseline outcome, the interaction is:

```text
(optional comma - optional no-comma)
  - (intransitive comma - intransitive no-comma)
```

A positive value indicates a larger comma effect when an object reading was
available. It reduces a shared punctuation explanation; it does not remove
verb-class-specific punctuation, lexical frequency, plausibility or positional
interactions. "Intransitive" names the intended ordinary sense in these sentences,
not a claim that the word can never occur transitively in English.

## Intervention and selection

Capture each decoder block's output at the final token of the target noun,
before the completion. Capture uses the same teacher-forced input shape as
scoring; causal masking prevents the captured state from reading the completion.
Each candidate has its own shape-matched capture. Patch the same noun position
in the recipient. Comma donors and recipients can have different absolute token
indices; both are logged. These position differences remain a possible confound.

On the six development families, measure comma → no-comma and no-comma → comma
patches at every non-final block, for both verb classes and both completions.
The final decoder block is excluded from selection and evaluation. Its direct
connection to the readout makes it unsuitable as evidence of a parsing circuit.
Earlier whole-residual patches can also move generic predictive information;
excluding the last block does not make earlier patches syntax-specific.

For each direction compute the change in the main-minus-object margin, orient
it toward the comma reading (positive for comma → no-comma; negate the reverse),
then subtract the corresponding intransitive-control effect. Average the two
directional differences. Select the block with the largest signed mean across
development families, resolving ties toward the earlier block. Do not select by
absolute effect or by test performance. The selected block is written to the raw
log **before any test-family forward pass**. Negative development values still
produce a selection but do not constitute supporting evidence.

Compute only that block on the six test families; no test layer curves are
calculated. Report the two directional effects separately as well as their mean.

At the chosen block, on development and test separately, compare:

1. Cross-condition patches within the same lexical family and verb class.
2. Identity patches, required to differ from baseline by no more than 1e-4 nats.
3. A same-condition lexical direction from changing only the embedded actor,
   rescaled to the cross-condition displacement norm.
4. Three Gaussian random directions with deterministic seeds, each rescaled to
   the same displacement norm. Within a family, block and direction, the same
   random vector is used for both candidate completions.

Controls are constructed in FP32 and rounded to the model dtype before injection.
Log the requested norm, actual post-cast norm, raw direction norm and scale; report
the median and maximum relative mismatch. Do not describe norms as exactly matched
in BF16. Lexical-direction scaling may itself move states off the data manifold;
random directions are generic perturbation controls, not valid alternative parses.
Three directions are a small descriptive control ensemble, not an exhaustive null.

## Analysis and decision

The independent unit is the paired lexical/verb family, not each direction,
candidate or random seed. Average random controls within family. Report means and
10,000-draw percentile family bootstrap intervals with fixed seed 0, including
paired cross-condition-minus-lexical and cross-condition-minus-random contrasts.
There are only six families per split; these intervals are unstable, descriptive
and conditional on the hand-selected materials. There is no population-level
significance claim and no correction for all possible hypotheses in this project.

Evidence worth following would combine a positive held-out baseline interaction,
positive held-out bidirectional intervention interaction, and a positive paired
advantage over both perturbation controls. Inspect directional asymmetry and
family dependence even if the average is positive. No threshold automatically
establishes a garden-path mechanism or a syntactic-role feature. A weak or
control-sized result is a reason to improve the assay before increasing model size
or searching heads. Preserve every raw observation; no post-hoc item exclusions.
Invalid alignment, nonfinite scoring, zero control direction with nonzero target
norm, or failed identity checks abort the run and leave a recorded error.

## Remaining limitations

- Materials are authored for this assay and have not been judged by humans.
  The contrasted events and temporal clauses need naturalness review.
  Optional object omission and compound-noun plausibility vary between families.
- The two verb classes differ lexically and semantically. A factorial interaction
  alone cannot identify a syntactic computation.
- The two full completions have unequal lengths and different events. The finite
  verb baseline helps interpret the contrast but does not remove that limitation.
- Patching copies the full residual vector. It does not isolate an abstract role,
  prove where that role was computed, or distinguish repair from reanalysis.
- This assay uses one model checkpoint and one NP/Z construction. It cannot
  support model-size or architecture-general claims.

Human validation and a more selective intervention remain necessary before a
mechanistic claim. Keep this continuation experiment separate from retrospective
syntactic-role decoding in the parallel logical-analysis track.
