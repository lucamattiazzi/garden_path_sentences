# GPS title case study — frozen before inference, 13 October 2026

Question: where is word surprisal concentrated in the title, and which reading
is elicited after the complete string? This is an exploratory case study, not
a circuit localization experiment or a validation of the title as a human GPS.

Six fixed cases in probe.CASES: original old/man pair; exact lowercase title
without article or final punctuation; title with The and period; an adverbial
cue before models; an additional finite verb after noun models.
Do not silently correct the exact title. Report all six cases, including failure.
The adverb and noun-reading controls change context; differences are not a pure
syntax effect. No sample size, confidence interval or scaling conclusion from
these hand-picked examples.

PT: pinned Gemma 3 12B, raw text with one BOS, FP32, TF32 off, eager attention.
Store all token IDs, offsets and log probabilities; aggregate subword surprisals
within each whitespace word, verify no token spans multiple words. Include
punctuation in the final word and label it. Report the second 'the' in the old/man
pair and every word of the title; human and migrations are candidate diagnostics,
not prespecified proven disambiguators. Top-five next tokens are diagnostic only.

IT: separately pinned Gemma 3 12B instruction-tuned, native chat template,
deterministic greedy answer, maximum 256 generated tokens, one neutral prompt
asking paraphrase, grammatical subject and main verb. The prompt gives no intended
parse. Use all six cases. An answer is prompted linguistic behavior, not access
to hidden reasoning and not proof that the PT model performed that interpretation.

One A100-80GB, sequential checkpoint loads, 20-minute function ceiling, no retries.
Existing named Hugging Face secret and weight cache; save source snapshot,
checkpoint revisions, software versions, raw responses and partial results.
No layer/head sweep or causal patch in this case study.
