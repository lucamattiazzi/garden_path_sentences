# V3: next-token NP/Z preferences and a positional diagnostic

Frozen before V3 inference. This is an exploratory extension of V2, with a fixed
checkpoint and fixed layer. It is not a preregistration or a new human study.

## Materials and scope

Use all 24 NP/Z item pairs in Hanna & Mueller's published equal-word-length
materials, pinned to GP-mechanisms commit
`3ea8380a08e6ea601cf65f1b624f74ef29fc765b`. Each optional-verb family occurs twice;
there are 12 independent resampling units, not 24 independent verbs. Sort the 12
optional verbs by SHA256 of `gps-v3-split:<verb>`; the first six families are
instrument-development, the other six evaluation. Both splits have 12 items.
Both splits are fixed before inference. No V3 outcome changes the assay, layer,
controls, inclusion, or evaluation schedule. Report splits separately.

Use the published ambiguous and intransitive (`sentence_post`) prefixes. Add a
comma before the second noun phrase, preserving all words. The three-word
continuation comes from the original ambiguous sentence's main clause, aligned
at its target noun. The equal-length adaptation removes some earlier modifiers;
our combined texts are adaptations, not newly human-validated materials. Repeated
control verbs and some nouns cross splits: only the optional-verb families are
disjoint. Public availability does not establish absence from Gemma training.
Source hashes, item IDs and original sentences are retained. No outcome-based
exclusions; any failed numerical/tokenization precondition aborts the run.

## Primary measurement: two single next-token alternatives

At the last noun, measure `log P(was) - log P(comma)` and retain the two log
probabilities separately. Both alternatives must be exactly one appended token;
` was` has a leading space; `,` is attached directly. No full-string score and
no per-token length normalization enter this primary endpoint. The same pair
was used for NP/Z in the authors' behavioral evaluation. It is a preference
between two tokens, not a posterior over complete syntactic parses; other tokens
and readings remain possible.

Capture and score from the same prefix-only forward shape. One block output at
the last noun is patched. Evaluate only block 17 (zero-based) from V2, excluding
the final block by assertion. No V3 layer search. Do not interpret it as the
unique location where syntax is computed.

## Three conditions and contrasts

- `native`: original no-comma prefix and normal position IDs.
- `comma`: insert the comma before the second NP; normal position IDs.
- `gap`: original no-comma tokens, with position IDs increased by one from the
  second NP onwards, including any teacher-forced continuation.

Assert that removing the one comma token gives exactly the native token IDs,
and that its remaining position IDs equal the gap condition. This holds within
verb class; equal word count across verb classes need not give equal token count.
All token IDs and position IDs are recorded.

The gap manipulates positional encoding with no additional token. It is an
artificial diagnostic, not a natural alternative sentence. It does not reproduce
the extra token's attention contribution or eliminate all positional/lexical
confounds. In short sequences, no sliding-window boundary is crossed.

Three predeclared contrasts: comma vs native; comma vs gap (aligned position IDs
for common tokens); gap vs native (position-only intervention). For each contrast
subtract the intransitive cue effect from the optional-verb cue effect. The
baseline interaction for comma/native is algebraically the sum of the other two;
patch effects need not be additive.

## Interventions and controls

For each contrast, run both directions, multiply the reverse effect by minus
one, subtract the intransitive effect and average the two directions. Controls:
identity, a lexical actor replacement using the actor from the other published
item in the same optional-verb family, and three deterministic Gaussian
perturbation directions. Lexical/random displacements are norm-matched to the
cross-condition displacement in FP32 and cast to BF16; log actual norm error.
They are perturbation references, not a null distribution of alternative parses.
Random seeds are paired across primary/trajectory stages. Controls are averaged
within source item before family aggregation; the two source items per optional
verb also remain inside one resampling unit. Identity tolerance is 1e-4 nats.

Primary endpoint: the bidirectional optional-minus-intransitive interaction for
`was` minus comma under `comma_aligned`, plus its paired advantage over lexical
and random controls. Retain ordinary and position-only contrasts as diagnostics,
each token component, and each patch direction. All results are reported even if
primary estimates are negative or control-sized. No formal significance claim.

## Secondary: actual continuation trajectory

Score the first three words of the published main clause for every condition,
retaining every token's log probability, token ID and piece. Capture the
pre-continuation noun state with the exact same teacher-forced shape as scoring;
causal masking excludes future words from that state. Use these captures for
identity/cross/control patches of the three-word sequence. Report each word and
the sum separately. This avoids attributing an effect on the entire string to
its first disambiguator. Future words condition later predictions in the normal
teacher-forced way; these are not independent observations or proof of repair.

## Analysis and execution

Bootstrap whole optional-verb families, 10,000 draws, seed 0, separately by split.
Use paired differences for cross-minus-control intervals. These six-family
intervals are exploratory, conditional on the selected materials and prior
fixed layer. Publish counts of negative and positive next-token margins by
condition without relabeling them as correct/incorrect parse decisions.

Gemma 3 1B pretrained revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`,
BF16, eager attention, no chat template; one A10, 20-minute ceiling, no retries.
Use a source/data allowlist, separate V3 results volume, immutable run ID,
source snapshot before model loading, raw JSONL, partial-failure metadata, and
local fetched results. No model-size comparison or role-probe training in V3.
A positive result motivates improved behavioral/causal validation, not SAE
feature naming. A small or unstable result motivates revising the assay before
scaling. The inspected V3 evaluation sample becomes development evidence for
any later redesign.

## Sources

- Hanna & Mueller (2025): https://aclanthology.org/2025.naacl-long.164/
- Authors' materials and behavioral code: https://github.com/hannamw/GP-mechanisms
- Underlying materials: authors describe their equal-length file as an adaptation
  of Arehalli et al. (2022); our export records both source file hashes.
- No explicit license file was present in the pinned authors' repository; this
  project does not assert a new license for those third-party materials.
