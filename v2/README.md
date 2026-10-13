# V2 — controlled garden-path experiments

The only active V2 track is incremental GPS prediction and intervention.
Independent role probes and UD validation were removed on 13 October 2026.

[Protocol](GARDEN_PROTOCOL.md) · [Follow-up](FOLLOWUP.md) ·
[Public explanation](https://research.grokked.it/mech-interp/garden-path-sentences/)

The original pilot scores `was` and the following word in ambiguous and
comma-disambiguated NP/Z sentences, patching before/after the disambiguator.
The follow-up crosses optional-transitive/intransitive verbs and comma/no-comma,
selects a block on development, and evaluates on six held-out families. It uses
bidirectional residual patches, identity, norm-matched lexical and random controls.
Its primary score compares full continuations, not just the critical token.

Run from the repository root:

```bash
uvx --from modal modal run v2/followup_modal.py --track garden --run-id fresh-gps-id
# Fetch without GPU inference:
uvx --from modal modal run v2/followup_modal.py --track garden --run-id existing-id --fetch
# Recompute GPS-only robustness diagnostics from saved observations:
python -m v2.robustness
# Original pilot analysis and figure:
python -m v2.analyze v2/results/pilot-20261008-03
python v2/plot.py v2/results/pilot-20261008-03
```

Pinned checkpoint and upload allowlist are in `followup_run.py`. The original
source snapshots and metadata of retained GPS runs are unchanged. They can contain
historically shared role code; no current entry point launches that removed track.
`Measurements`, patch hooks and family-bootstrap helpers in `experiment.py` are
also dependencies of the V3 GPS experiments.
