# GPS title case study

Completed on 13 October 2026. This returns to the original question: surprisal at
a disambiguating word and the behavior elicited by a temporarily ambiguous sentence.
No general role benchmark and no circuit search are included in this case study.

[Protocol](PROTOCOL.md) · [Raw results](results/title-20261013-01/results.json)
· [Metadata](results/title-20261013-01/metadata.json)
· [Verification](results/title-20261013-01/verification.json)
· [Public explanation](https://research.grokked.it/mech-interp/garden-path-sentences/)

Six fixed sentences. Gemma 3 12B PT: per-token and per-word surprisal, FP32/eager,
TF32 off. Gemma 3 12B IT: separately prompted grammatical explanation, native
chat format, greedy decoding. Checkpoint revisions are frozen in `probe.py`.
One A100-80GB; the function completed in about 114 seconds and the app stopped.

For `The diffusion of language models human migrations.`, PT surprisal at `human`
is 14.220 nats; with `actually` before `models`, 4.832. The exact lowercase
article-free title gives 13.765, but its highest-probability next token after
`models` is a special image token. All logits were scored without filtering.

IT fails the intended parse on both title variants; with `actually` it recognizes
`models` as a verb. It also fails the old/man control, so the prompted parse is
not a validated measure of internal comprehension. The result does not establish
that PT and IT use the same reading or identify a causal circuit.

```bash
python -m unittest gps_title.test_probe
uv run --no-project --with matplotlib python -m gps_title.analyze
# Requires existing Modal profile and named huggingface secret. Use a fresh ID:
uvx --from modal modal run gps_title/modal_app.py --run-id fresh-title-id
# Fetch a completed run, without inference:
uvx --from modal modal run gps_title/modal_app.py --run-id title-20261013-01 --fetch
```

Raw outputs, exact prompt, token IDs/offsets, source snapshot and version metadata
are retained. Final punctuation is included in the last word's surprisal.
The supplementary published JSON is generated from these files by `analyze.py`.
