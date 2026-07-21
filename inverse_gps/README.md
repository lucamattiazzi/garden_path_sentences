# Inverse garden paths — LLM-hard, human-easy

New research angle: hunt for sentences that produce a **garden-path-like
failure in LLMs but not in people** — the mirror image of the question the
rest of this repo asks.

## Why this is a promising direction

The literature already shows the two processing systems come apart in one
direction: LM surprisal *underestimates* human garden-path difficulty
(van Schijndel & Linzen 2021; Arehalli, Dillon & Linzen, CoNLL 2022), and on
the large-scale SAP self-paced-reading benchmark LM surprisal fails to
explain syntactic disambiguation difficulty at all (Huang et al., JML 2024).
Recent comprehension-based work shows LLMs also *misinterpret* garden paths
where humans recover (Amouyal et al., ACL 2025; arXiv:2510.07141 reports
frontier models near ceiling on non-GP structures but ~50% on GP ones).

If we can also show the opposite dissociation — constructions where the
statistical parser is pulled off the rails while the human syntactic parser
is not — we get a **double dissociation**, which is much stronger evidence
that the model's "reanalysis" circuit (located in the main pipeline) is a
statistical mechanism rather than a human-like parser. And the traps are a
natural input for the same mech-interp toolchain (patching, head ablation):
does the circuit found on classic GPS also light up on LLM-only traps?

## Trap templates (`sentences.py`, 30 items)

| Template | Trap | Why LLM-hard | Why human-easy | Literature |
|---|---|---|---|---|
| `IDIOM` (10) | Literal use of a V+NP idiom chunk ("spilled the beans **onto** the floor") | Idiomatic continuation dominates the training distribution; figurative and literal readings compete inside the model | Context has already disambiguated; no syntactic reanalysis is ever required | Idiom figurative/literal tug-of-war in LLMs (arXiv:2506.01723); idiom surprisal advantage |
| `QUOTE` (10) | Verbatim famous-quote opening continued off-script but grammatically ("Houston, we have a **solution**") | Verbatim memorization distorts LM surprisal (Oh & Schuler, TACL 2023: memorized sequences are a systematic source of LM/human misfit) | Continuation is semantically seamless; predicted human cost is mild amusement, not reanalysis. **Weakest "not-in-people" claim → prediction is a magnitude gap, not zero human effect** | Oh & Schuler 2023 |
| `ROLE` (10) | Grammatically unambiguous but role-implausible sentences ("The dog was bitten by the man") | LMs weigh plausibility over argument structure (psycholinguistic evaluation of argument-role sensitivity, arXiv:2410.16139) | Surface syntax is unambiguous; humans at normal reading pace are near ceiling. **Caveat:** humans noisy-channel-correct some implausible sentences (Gibson et al. 2013) — the human baseline must be measured, not assumed | arXiv:2410.16139; Christianson et al. 2001; Gibson et al. 2013 |

Every item is a minimal pair: the control breaks the trap (idiom noun
swapped, quote de-memorized, roles restored) while preserving the
continuation, so all the repo's minimal-pair machinery applies unchanged.

## Detectors

From **this repo** (same logic, same tokenization handling via
`model_finder.surprisal`):

- **D1 Δ-surprisal** at the forced continuation,
  `S(target|trap) − S(target|control)` — mirrors `01_baseline`.
- **D3 free continuation** — greedy decode after the trap prefix, flag
  attractor capture — mirrors the predict steps (04–06, 09).
- (Once a trap class is confirmed: attention patterns, activation patching,
  mean ablation and the null-distribution significance machinery, steps
  10–14, apply as-is.)

From the **literature**:

- **D2 capture score**, `logP(attractor) − logP(target)` on trap vs control
  prefix — the BLiMP-style wrong-continuation preference measure, plus a
  trap-specificity contrast.
- **D4 comprehension probe** — forced-choice QA after the sentence, the
  Christianson et al. (2001) lingering-misinterpretation paradigm as applied
  to LMs by Amouyal et al. (ACL 2025) and arXiv:2510.07141.
- **Human side** — reading-time prediction: the human analogue of D1 is a
  self-paced-reading effect at the target region. For classic GPS these
  exist in the SAP benchmark (Huang et al. 2024); for our traps the
  *prediction* is ≈0 human effect (see below).

## Establishing the "not in people" half

Three levels, in increasing cost:

1. **By design** — in IDIOM/ROLE traps the wrong reading is not a parse the
   human incremental parser ever needs to abandon: context (IDIOM) or
   unambiguous surface syntax (ROLE) rules it out without reanalysis. This
   is an argument, not a measurement.
2. **From the literature** — human ceiling on unambiguous implausible
   sentences (with the Gibson noisy-channel caveat), human literal-idiom
   reading costs known to be small once context biases literal.
3. **Measured** (the publishable version): a small self-paced-reading study
   on the 20 surprisal-instrumented items (trap/control × target region,
   ~40 participants) plus comprehension questions on all 30. Prediction:
   human RT effect ≈ 0 at target and QA accuracy ≈ ceiling, while the LLM
   shows Δ > 0, capture > 0, and a QA gap.

## Success / falsification criteria

The angle *works* if, for at least one template, across several models:
`mean Δ > 0` with capture specificity `> 0` (01), attractor capture in free
generation (02), and a trap-vs-control QA gap (03) — while the human checks
above hold. It *fails informatively* if models sail through the traps
(evidence their parse commitment is more context-sensitive than the
statistical-capture story predicts), or if humans turn out to garden-path
too (the trap classes then need redesign, e.g. stronger disambiguating
context).

## How to run

```bash
uv run python inverse_gps/sentences.py                                  # dataset sanity check, no model
uv run python inverse_gps/scripts/01_screen.py       --model Qwen/Qwen3-4B-Base
uv run python inverse_gps/scripts/02_continuation.py --model Qwen/Qwen3-4B-Base
uv run python inverse_gps/scripts/03_qa_probe.py     --model Qwen/Qwen3-4B-Base
```

Any HF causal LM from `model_finder/models.py` works via `--model`; outputs
land in `inverse_gps/logs/<step>__<model>.{log,json}`.

## Open extensions

- **TOKEN traps**: adversarial subword segmentations (rare compounds,
  hyphenation, numbers) — genuinely LLM-specific but needs different
  instrumentation than word-level surprisal.
- **Frequent-bigram attractors spanning a syntactic boundary** where the
  attractor parse is *ungrammatical* — candidate class, but hard to design
  without also garden-pathing humans (compound-noun traps like "rain
  forest fires" catch humans too, so they were excluded here).
- **Instruct-model QA** with chat templates, matching the setup of the
  frontier-model comprehension results (arXiv:2510.07141).
- **Circuit overlap**: run steps 10–14 of the main pipeline on confirmed
  traps — is the GPS circuit re-used when the model garden-paths on
  memorized text?

## Key references

- van Schijndel & Linzen (2021), *Single-stage prediction models do not
  explain the magnitude of syntactic disambiguation difficulty*, Cognitive
  Science.
- Arehalli, Dillon & Linzen (2022), *Syntactic surprisal from neural models
  predicts, but underestimates, human processing difficulty from syntactic
  ambiguities*, CoNLL — [aclanthology.org/2022.conll-1.20](https://aclanthology.org/2022.conll-1.20/)
- Huang et al. (2024), *Large-scale benchmark yields no evidence that
  language model surprisal explains syntactic disambiguation difficulty*,
  JML (SAP benchmark) — [tallinzen.net/media/papers/huang_et_al_2024_jml.pdf](https://tallinzen.net/media/papers/huang_et_al_2024_jml.pdf)
- Oh & Schuler (2023), *Why does surprisal from larger transformer-based
  language models provide a poorer fit to human reading times?*, TACL —
  [aclanthology.org/2023.tacl-1.20](https://aclanthology.org/2023.tacl-1.20/)
- Amouyal, Meltzer-Asscher & Berant (2025), *When the LM misunderstood the
  human chuckled: Analyzing garden path effects in humans and language
  models*, ACL — [aclanthology.org/2025.acl-long.403](https://aclanthology.org/2025.acl-long.403/)
- *Comparing human and language models sentence processing difficulties on
  complex structures* — [arXiv:2510.07141](https://arxiv.org/abs/2510.07141)
- *A psycholinguistic evaluation of language models' sensitivity to argument
  roles* — [arXiv:2410.16139](https://arxiv.org/abs/2410.16139)
- *Tug-of-war between idioms' figurative and literal interpretations in
  LLMs* — [arXiv:2506.01723](https://arxiv.org/abs/2506.01723)
- Christianson, Hollingworth, Halliwell & Ferreira (2001), *Thematic roles
  assigned along the garden path linger*, Cognitive Psychology.
- Gibson, Bergen & Piantadosi (2013), *Rational integration of noisy
  evidence and prior semantic expectations in sentence interpretation*,
  PNAS.
