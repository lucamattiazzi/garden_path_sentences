# Direct participant answers: development-only instrument validation

Frozen before inference, after the failed A/B bridge and its FP32 diagnostic.
This is an adaptive development follow-up, not independent confirmation.

Use the same Gemma 3 1B pretrained revision, full FP32, eager attention and TF32
disabled. The four demonstrations express the same meanings as the A/B task but
answer with participant names. Remove option labels and option lists entirely.
No prompt search, model search, generation sampling or outcome-based exclusion.

Use only the six existing development families: two participant reversals, active
and passive voice, agent and patient questions; simple clauses and no context.
96 observations, each scoring both candidate completions. No held-out inference
or causal patches in this validation run, regardless of the result.

Prompt ends `Answer: the`; completions are ` <participant>.`. Score the sum of
teacher-forced log probabilities of ALL appended tokens including the final period.
Require token-stable prompt boundaries; save token IDs and individual log scores.
These are complete sequence probabilities, not per-token length-normalized scores.
Different token counts are allowed and reported, not filtered. Reversing the two
participants counterbalances lexical/length priors; success requires getting both
reversals right, not merely a positive mean log-odds difference. This does not make
all tokenization or lexical confounds disappear. Report joint candidate probability
mass but do not interpret pairwise ranking as free-generation accuracy.

Freeze the gate at >=75% correct in BOTH participant reversals, separately in each
of the four voice x queried-role strata on simple clauses. Ties count as incorrect.
With six families this requires at least five successful pairs per stratum. The
no-context condition has identical prompts with opposite labels under reversal;
its pairwise ranking accuracy should be 50% without ties and its both-reversal
success must be zero. No candidate-order effect exists: completions are scored
independently, never presented as an ordered list in the prompt.

Report raw accuracy, both-reversal success, family margins and descriptive 95%
bootstrap intervals over six families (10,000 resamples, seed 0), separately by
voice, queried role and condition. Families are the independent units; reversals
are not independent repetitions. A gate failure says this checkpoint/readout is
not reliable enough for the planned causal interpretation; it is not evidence
that the model lacks role representations. A gate pass licenses designing a
separately frozen bridge, not retrospectively reopening the failed A/B experiment.

Snapshot sources, generated prompts and hashes before loading the model. One Modal
A10; 10-minute ceiling, no retries. Save partial metadata on failure, never overwrite
an existing run. Old V1/V2/V3 sources and test behavior remain untouched.
