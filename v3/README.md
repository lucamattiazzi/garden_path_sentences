# V3: one-token NP/Z preferences and positional controls

**Completed on Modal:** [Italian HTML report with figures](results/npz-v3-20261008-01/report.html)
([Markdown](results/npz-v3-20261008-01/report.md)), run `npz-v3-20261008-01`.
The GPU function took 133.7 seconds on one A10; the app is stopped.

Held-out aligned-position patch interaction: **6.599 nats [4.065, 8.734]**.
Paired advantages: **6.223 [3.706, 8.148]** over lexical controls and
**6.564 [3.957, 8.786]** over random controls. The position-only interaction is
**−0.141 [−0.898, 0.648]**. All six test-family means are positive in the aligned
contrast. These are descriptive six-family bootstrap intervals, not a proof of
reanalysis or a syntax-specific feature.

In the actual continuation, the aligned patch interaction is concentrated on the
first word: **3.815 [1.974, 5.605]** nats, versus 0.027 and −0.018 on words two
and three. This localizes the measured predictive effect in the continuation;
it does not establish recovery of an initially incorrect parse.

Identity errors were zero. Raw-effect, independent family-aggregation and source
snapshot audits passed. BF16 norm matching remains approximate (maximum 2.491%
error in the aligned contrast; 6.958% across all contrasts).

V3 replaces V2's unequal-length primary continuations with the single-token
contrast `was` versus comma on published NP/Z materials. It measures effects at
**fixed block 17 of the same Gemma 3 1B checkpoint**. It also records every token
in the first three words of the original main clause.

This track investigates continuation preferences. It does not identify a parse
posterior, a unique syntax layer, a reanalysis circuit, or general role decoding.

## Behavioral follow-up

The [behavioral bridge](bridge/README.md) tested agent/patient questions on new
synthetic sentences. Its development competence gate failed in BF16 and in a
separate FP32 diagnostic, so no causal patch or held-out inference was performed.
See the [offline report](bridge/results/bridge-v3-20261008-02/report.html). This
limits that QA instrument; it does not overturn the predictive results above.

The subsequent [direct-name validation](direct/README.md) also failed its
competence gate, with a specific weakness on agents in passive sentences.
No held-out inference or causal patches were run.

The [larger-model comparison](scale/README.md) subsequently found that 12B passes
the unchanged development competence gate, while 4B fails. A separately frozen
[12B causal bridge](bridge12/README.md) has completed after that selection:
strong predictive effects and opposing QA-margin shifts by role, with no cross-patch
answer flips on the held-out set. See its report for controls and interpretation.

## Design

- 24 published items, grouped into 12 optional-verb families; 6 development and
  6 test families, with two items per family. Control verbs/nouns can recur across
  splits; only optional-verb families are disjoint.
- Optional versus intransitive verbs, with and without a comma.
- Additional no-comma condition with a gap in position IDs. It aligns common
  tokens' positional IDs with the comma condition without adding a token.
- Both patch directions, identity, norm-matched lexical control, and three
  norm-matched Gaussian directions, averaged within item and family.
- Prefix-only capture for the next-token endpoint. Shape-matched teacher-forced
  captures for the secondary three-word trajectory; future tokens remain masked.
- 3,456 interventions and 288 baseline measurements. No layer sweep or new probe
  fitting, no outcome-based item selection, no additional model sizes.

Read the frozen [protocol](PROTOCOL.md) before interpreting the results.
`materials/npz.json` includes source IDs, source file hashes, the pinned upstream
commit, original sentences, and exact adapted prefixes. These are adaptations of
published materials, not newly human-validated sentences. Source provenance does
not rule out training exposure or guarantee every verb-class label in every sense.

## Run and reproduce

From the repository root, using the existing Modal profile and `huggingface`
secret. One A10, 20-minute ceiling, no retries, separate `gps-v3-results` volume.
Only the explicit files in `v3.run.SOURCES` are uploaded.

```bash
# Use a fresh ID; existing remote runs cannot be overwritten.
uvx --from modal modal run v3/modal_app.py --run-id my-v3-run
# Retrieve existing results without GPU inference:
uvx --from modal modal run v3/modal_app.py --run-id my-v3-run --fetch
# Rebuild the audit, Markdown, offline HTML and scientific plots:
uv run --no-project --with numpy --with matplotlib --with markdown python v3/analyze.py v3/results/my-v3-run
# Reproduce materials from pinned, hash-checked upstream files:
python v3/prepare_materials.py
```

The run saves `source.json`, metadata (including partial-failure status), raw
`observations.jsonl`, and `results.json`. Analysis independently recomputes primary
family effects and checks every raw effect against its baseline. The two source
CSV files are retrieved by the preparation script; no upstream code is executed.
No new license is asserted for third-party materials.

## Sources

- [Hanna & Mueller, NAACL 2025](https://aclanthology.org/2025.naacl-long.164/).
- [Authors' code and data](https://github.com/hannamw/GP-mechanisms), commit
  `3ea8380a08e6ea601cf65f1b624f74ef29fc765b`.
- The authors describe the equal-length file as an adaptation of Arehalli et al.
  (2022). Our exact transformations and source hashes are recorded locally.
