# Larger Gemma 3 checkpoints — fixed direct-answer validation

[Italian comparison report and figures](results/comparison-20261009/report.html)
([Markdown](results/comparison-20261009/report.md)).

**Completed:** 4B failed; **12B passed all four strata at 100%** (all participant
reversals correct). The planned checkpoint search stopped at 12B; 27B was not run.
The [separately frozen 12B causal bridge](../bridge12/README.md) has also completed
on held-out families: large predictive effects, role-dependent QA-margin shifts,
no change in preferred answers under the cross patch. Passing the development gate alone is not a causal result.

This extension reuses the exact 96 development-only materials, demonstrations,
labels, scoring and competence gate from `v3/direct`. It changes the pretrained
checkpoint, using 4B first and 12B only after a failed 4B gate. The series stops
at the first pass or after 12B; see the frozen [protocol](PROTOCOL.md).

The primary criterion is >=75% correct in both participant reversals in every
voice/queried-role stratum (five of six families required). No test inference,
causal patch, prompt search or threshold adjustment occurs during this selection.
The 1B result is included as a historical baseline and was not rerun.

Full FP32, TF32 disabled, eager attention. Larger checkpoints use the complete
Gemma 3 conditional-generation wrapper with text inputs only. Loading must have
no missing, unexpected or mismatched weights. Local tests exercise the same path
through a tiny multimodal model; actual checkpoint loading is recorded separately.

## Reproduce

Run from the repository root with the existing Modal profile/Hugging Face secret:

```bash
# New inference needs a fresh ID. Run 12B only after a failed 4B competence gate.
uvx --from modal modal run v3/scale/modal_app.py --size 4b --run-id fresh-scale-4b
uvx --from modal modal run v3/scale/modal_app.py --size 12b --run-id fresh-scale-12b
# Retrieve saved data without inference:
uvx --from modal modal run v3/scale/modal_app.py --run-id scale-4b-20261009-02 --fetch
# Rebuild comparison, independent audits, Markdown, SVG, PNG and offline HTML:
uv run --no-project --with numpy --with matplotlib --with markdown python v3/scale/analyze.py v3/scale/results/scale-4b-20261009-02 v3/scale/results/scale-12b-20261009-01 --output v3/scale/results/comparison-20261009
```

One A10G for 4B (15-minute ceiling) or A100-80GB for 12B (20-minute ceiling);
no automatic retries or concurrent model execution. The incomplete initial 4B
attempt is retained: inference finished, but the final metadata JSON write failed
because Transformers returned sets. Serialization was repaired with a regression
test; the repeated 4B run produced byte-identical raw observations. It is not counted
as an independent scientific replication.

The report audits each completed run, identical material snapshots across models,
checkpoint revisions, empty loading-error lists and the predeclared selection
sequence. Sources from earlier completed experiments are unchanged. Six reused
development families do not support a claim of held-out generalization, abstract
role representations, or a causal effect of parameter count alone.
