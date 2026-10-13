# Post-hoc precision diagnostic

Written after `bridge-v3-20261008-02`, before numerical follow-up inference.
The BF16 run failed its instrument gate. Among its 96 simple-clause development
measurements, 46 have exactly tied A/B log probabilities. We therefore cannot
interpret the 25% strict-positive accuracy as ordinary below-chance performance:
ties count as incorrect under the frozen rule.

Re-evaluate only the same 96 simple and 96 no-context development prompts with
full FP32 weights and computation, eager attention, TF32 disabled. Use the same
pinned checkpoint, tokenizer, demonstrations, gold labels and option orders.
Save individual measurements and compare margins, ties, strict accuracy and
both-order correctness against the existing BF16 run.

This is a post-hoc numerical diagnostic, not a new confirmation experiment.
There is no test inference, no causal patch, no model/layer search and no threshold
change. Even if FP32 meets the original threshold, do not reopen the stopped BF16
causal experiment automatically. Changes to a later experiment need their own
frozen protocol. The diagnostic can determine whether BF16 ties explain the gate
failure; it cannot measure the model's general role competence.

One A10, ten-minute function ceiling, no retry. Sources and original observations
hash are saved before loading the model. Exact duplicate no-context prompts are
retained to preserve the counterfactual label balance of the original run.
