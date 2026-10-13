# Garden-path research: reviewed baseline and behavioural pilot

Status: exploratory protocol, written after inspecting the historical results.
This is not a preregistration. The new materials have not been independently
validated by human readers; no new model finding follows from preparing them.

## What the historical results support

Qwen3 and Gemma3 assign different probabilities to GPS and normal strings.
Interventions change these predictions. The continuation proxy does not identify
syntactic comprehension, incomplete reanalysis, or the absence of a circuit.
The original 50 items and the additional 10 NV items remain unchanged so historical
results can be interpreted. “Canonical” and “novel” are historical set labels,
not verified properties of every sentence or of its training-corpus exposure.

Reproduce the audit and rebuild the corrected HTML report from `v1/`:

```bash
uv run python -m research.audit
uv run python report/build_report.py
```

The audit writes [results/legacy-audit.md](results/legacy-audit.md) and
[results/legacy-audit.json](results/legacy-audit.json). It fingerprints all 98 input
JSON files, flags non-finite scores and mismatched baseline observations, recomputes
item-bootstrap intervals and leave-one-out means, joins model comparisons by item
ID, and compares NV head effects with the NV random-head reference. These operations
do not rerun inference or repair historical logs.

Specific corrections:

- Stored `effect_drop` means ablated minus baseline Δ; a reduction is negative.
- Step 10 ranks attention differences but correlates **GPS attention**, not the
  GPS-minus-normal difference, with continuation cost. Those p-values are exploratory.
- Qwen14B L10.H25 changes NV Δ by −0.134 nats, outside the 60-head sampled range.
  This contradicts the blanket claim that no head exceeded that reference. It is
  not a selection-corrected result.
- Qwen4B step 02 contains a non-finite baseline observation (#11) and differs from
  step 01 on #5. Historical provenance does not establish the cause. New shared
  intervention measurements abort on any non-finite score, including controls,
  instead of silently changing the sample across conditions.
- The final-layer critical-state patch copies the readout for a single-token target.
  The ambiguous-word state precedes disambiguation. Neither observation identifies
  a syntax-specific revision operation.
- Pooled mean ablation is another intervention, not a guarantee of remaining
  in-distribution. Disagreement with zero ablation is sensitivity, not proof of an artifact.
- Random heads provide an empirical effect reference, not a null of no causal
  influence. Max-of-K resampling omits the preceding data-dependent layer selection
  and cannot extrapolate beyond the observed tail.

## Pilot question

Can a model identify the subjects of two events after an NP/S or NP/Z ambiguous
prefix, while remaining sensitive to a reversal of the participants and robust
to answer order? Do incremental prediction differences accompany errors on this
task? This supplies behavioural prerequisites for later interventions; it does
not by itself distinguish incremental reanalysis from recomputation during QA.

### Materials and split

[materials/pilot.json](materials/pilot.json) contains 24 hand-authored items:
12 lexical families, each with two different participant pairs. Each template
has four exploration families and two validation families:

| Template | Exploration ambiguity verbs | Validation ambiguity verbs |
|---|---|---|
| NP/S | knew, heard, believed, remembered | suspected, assumed |
| NP/Z | watched, washed, hunted, visited | called, trained |

The default run uses the 16 exploration items (8 families). Validation requires
an explicit `--split validation` run and contains 8 items (4 families). Do not
choose heads, prompts, exclusions or thresholds from the validation results and
continue calling them held-out. If materials change after inspecting model
results, assign a new dataset version and obtain a fresh validation set.

Each item contains:

- **GPS**: temporarily ambiguous string.
- **Normal**: same events and actors with `that` or a comma providing the cue.
- **Role-reversed normal**: exchange the two actors, preserving the word inventory
  and questions while reversing the correct answers.
- **No-context diagnostic**: the questions and answer options without a sentence;
  this measures answer/lexical preferences, not comprehension accuracy.

Two questions cover the two event subjects. Each question is presented with both
orders of the same A/B candidates. This balances subject positions and answer
labels and makes simple word-presence strategies insufficient for the complete
task. It does not rule out all surface heuristics. Role reversal can also alter
plausibility; both directions need human ratings.

Example:

```text
GPS:     The doctor knew the nurse had arrived at the clinic.
Normal:  The doctor knew that the nurse had arrived at the clinic.
Reverse: The nurse knew that the doctor had arrived at the clinic.

Who knew about the arrival?
Who had arrived at the clinic, in the situation described?
Options in both orders: the doctor / the nurse
```

### Measurements

1. **Incremental prediction**: whole-word surprisal at the designated critical word
   and each following word, GPS and normal. Each word is scored only from its
   preceding prefix. Inspect the trajectory and individual items; do not label it
   reanalysis solely because it peaks and then declines.
2. **Role-answer margin**: `log P(correct label) − log P(incorrect label)`, averaged
   over the two option orders. A/B must each tokenize as one token with the same
   unchanged prompt tokenization. Base models receive a fixed plain-text prompt;
   no generation or chat template is used.
3. **Order robustness**: both answer orders must favour the correct actor strictly.
   Ties are counted as failures. Ordinary accuracy over presentations is also saved;
   an always-A model scores 50% there and 0% on order robustness.
4. **Paired GPS cost**: GPS minus normal correct-answer margin. Negative means worse
   discrimination in GPS. This sign differs from the legacy surprisal-cost proxy.

Questions and answer orders are repeated measurements. Average within each lexical
family, then use a paired family bootstrap (10,000 draws, seed 0). The family mean
is the estimand; the same observations must exist in every condition. Report
per-template and pooled summaries plus raw prompts, scores and item trajectories.
The few pilot families give coarse uncertainty estimates, not publication-ready power.

### Go/no-go criteria for the next experiment

Use these as operational pilot criteria, not significance thresholds:

1. Obtain independent judgments of grammaticality, intended parse, first point of
   disambiguation, question answerability and plausibility in both actor orders.
   A structurally valid JSON file is not linguistic validation. Revise materials
   before collecting the confirmatory set; record reasons for every exclusion.
2. Require at least 80% order-robust accuracy on **both normal and role-reversed
   controls**, reported separately for each template, before interpreting GPS
   failures as anything beyond task-format or general comprehension difficulty.
3. Inspect no-context preferences, question types and leave-one-family-out results.
   Set sample size for a larger study from variability across families and a
   prospectively chosen smallest meaningful effect, not from whether p < .05 appears.
4. Freeze the expanded materials, measurement and selection rule before evaluation
   on new validation families. Failure to detect a difference is not equivalence.

### Run

Structural validation needs no model download:

```bash
uv run python -m research.behavioral --validate-only
```

Run a pilot with an exact Hugging Face checkpoint commit (replace the placeholders):

```bash
uv run python -m research.behavioral \
  --model Qwen/Qwen3-4B-Base \
  --revision <40-character-checkpoint-commit> \
  --split exploration \
  --dtype bfloat16 \
  --output research/results/qwen3-4b-exploration-v1.json
```

Add `--local-files-only` to prevent downloads. The runner refuses to overwrite
an existing result and saves the checkpoint revision, runtime versions, device,
dtype, dataset/code SHA-256 hashes, answer token IDs and full per-order prompts.
The output remains labelled an unvalidated exploratory pilot even when all code
checks pass. A failure during scoring produces no completed results file.

The local `.venv` observed during review referenced a removed uv Python 3.14.5
interpreter. Repair the environment with `uv sync` before relying on the commands
above. This does not alter any saved experimental results.

## Causal follow-up after behavioural validation

Select candidate components using exploration families only, and evaluate the
same interventions on untouched validation families. Measure the paired
GPS-minus-normal intervention effect on answer margins and on incremental
predictions. A head that merely promotes an answer label should fail the order
and actor-reversal controls.

Patch comparable states in both directions; separate pre-disambiguation positions
from disambiguator, spillover and question positions. Include attention outputs,
MLP outputs and targeted groups. Full residual copying remains a coarse positive
control. Use matched donor examples and syntax-irrelevant interventions to test
whether the effect is specific to the proposed structural information.

To distinguish prior preference, repair and reanalysis, test whether post-cue
answers depend on the same reading-specific information encoded before the cue,
or on information computed at later positions. Neither successful follow-up QA
nor a successful residual patch establishes that distinction alone.

## Related work

- [Hanna & Mueller (NAACL 2025), Incremental Sentence Processing Mechanisms in
  Autoregressive Transformer Language Models](https://aclanthology.org/2025.naacl-long.164/):
  directly relevant feature-level analysis, alternative parses and comprehension
  controls. This pilot is not a reproduction of their stimuli or results.
- [Zhang & Nanda (ICLR 2024), Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042):
  dependence of localisation results on metrics and intervention methods.

## Targeted checks

```bash
uv run python -m unittest discover -s tests -p test_core.py
uv run python -m unittest discover -s tests -p test_audit.py
uv run python -m unittest discover -s tests -p test_behavioral.py
```

Synthetic models/scorers test measurement mechanics; they supply no evidence about
the syntactic abilities of a pretrained model.
