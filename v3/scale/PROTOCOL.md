# Larger-model competence validation — frozen before inference

Adaptive development follow-up to the failed Gemma 3 1B direct-name run.
Reuse EXACTLY `v3.direct.experiment.materials`, `decisions`, and the full sequence
scorer. No modification to demonstrations, sentences, labels, score normalization,
role/voice strata, threshold, or six-family development split. 96 observations /
192 forward passes per model. No test inference and no causal patch in this run.

Predeclare this sequence and stop at first passing checkpoint:
1. google/gemma-3-4b-pt, cc012e0a6d0787b4adcc0fa2c4da74402494554d
2. Only if 4B fails: google/gemma-3-12b-pt,
   295efb63d01a7017928f273a94ebb86105c9526f
Stop after 12B even if both fail. Do not pick the best-looking subset or lower the
gate: >=75% correct under BOTH participant reversals in EACH voice x role stratum.
A runtime/load error is not a scientific gate failure and does not license model
selection; repair runtime issues without inspecting or changing the prompt.

Use pretrained rather than instruction-tuned checkpoints to retain the experiment's
model family. Full FP32, eager attention, TF32 disabled, seed 0, Transformers 5.9.0
and Torch 2.10.0. Load the full Gemma3ForConditionalGeneration wrapper, pass only
text, and use its language-model layers for access. No image inputs or chat
formatting. Require no missing/unexpected/mismatched checkpoint weights. This
larger architecture is not identical to the 1B architecture: this is a checkpoint
comparison, not an isolated causal manipulation of parameter count.

4B: one A10G, 32 GiB CPU memory, 15-minute function ceiling.
12B: one A100-80GB, 80 GiB CPU memory, 20-minute function ceiling.
No retries or concurrent model execution. Source snapshots and complete generated
materials precede model loading. Existing output IDs cannot be overwritten.

Report ALL attempted checkpoints, the original 1B result for context, both-reversal
and per-sentence accuracy, uncertainty by family, token lengths, loading checks and
actual run duration. Prompt/material identity with the saved 1B run is audited.
Selection uses reused development data; passing is not held-out confirmation.

If one passes, it becomes the candidate for a separately frozen causal-bridge
protocol. Do not treat behavior on simple clauses as proof of a garden-path effect,
reanalysis, or an abstract syntactic representation.

Sources: official model cards https://huggingface.co/google/gemma-3-4b-pt and
https://huggingface.co/google/gemma-3-12b-pt ; runtime documentation
https://huggingface.co/docs/transformers/model_doc/gemma3 . Revisions resolved via
the official Hugging Face model API before this experiment. Modal published GPU
rates checked 2026-10-09: A10 $0.000306/sec; A100-80GB $0.000694/sec, plus CPU,
memory and storage. They are resource rates, not a claimed final invoice.
