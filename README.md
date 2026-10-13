# Garden-Path Sentences — finding the disambiguation circuit

**This project studies ONLY garden-path sentences (GPS).** Its original question
is whether we can identify a causal circuit that enables a language model to
process a temporarily ambiguous sentence when disambiguating evidence arrives.
The primary behavioral measurement is **surprisal at the actual disambiguating
word**, followed separately by spillover on subsequent words.

```text
Ambiguous prefix:   The old man
Clearer prefix:     The old people man
Disambiguating word:            the
Later continuation:                boat.
```

`S(the | The old man) = −log P(the | The old man)`.
The garden-path cost is `S_ambiguous − S_clearer`, in nats. The ambiguous word
`man`, the disambiguating word `the`, and the later word `boat` are different
measurement/intervention sites. A probability change is an operational signal;
it does not alone identify a complete syntactic parse or a comprehension circuit.

## Evidence and current work

- [V1](v1/README.md): original critical-word surprisal, layer/head ablations,
  continuation scoring and activation patching. Original logs are preserved.
  [Independent audit](v1/research/results/legacy-audit.md).
- [V2 GPS](v2/README.md): comma × verb-class controls and development/test split.
  Full-completion margins are a secondary diagnostic, not critical-token surprise.
- [V3 GPS](v3/README.md): single-token diagnostic, positional controls and actual
  continuation trajectory. `was` versus comma is not the same endpoint as the
  surprise of each sentence's actual disambiguator.
- [V3 12B bridge](v3/bridge12/README.md): historical auxiliary check on completed
  GPS sentences; strong prediction effects, zero preferred-answer changes among
  48 cross patches. This does not directly measure revision of the initial attachment.
- [V4 original](v4/README.md): historical GPS-context instrument gate, failed;
  no causal patches. It is not a general role-analysis research direction.
- [Title case study](gps_title/PROTOCOL.md): the exact string
  `diffusion of language models human migrations`, with separate punctuation/
  article controls, token surprisal and elicited final interpretation.

**A specific reanalysis/comprehension circuit has not yet been established.**
Candidate interventions must be tested on ambiguous versus matched clear
sentences, at prespecified pre/post-disambiguation sites, with identity, lexical,
random and positional controls and independent evaluation materials. Behavioral
success must concern the same ambiguity, not an unrelated agent/patient task.

## Report and reproduction

[Public report in Italian, three levels](https://research.grokked.it/mech-interp/garden-path-sentences/)
· [Markdown source](<reports/Garden path evidenze e limiti.md>)
· [Build and publication](deploy/gps/README.md)

V1 commands run from `v1/`; V2/V3 and the title case study run from the repository
root. Follow each version's protocol and use fresh run identifiers. Saved results
are historical evidence, not permission to reselect hypotheses on a held-out set.

## GPS-only cleanup, 13 October 2026

Removed the independent V2 role probes/UD materials, non-GPS V4 follow-ups,
V5–V9, and mixed role-analysis reports/talk packages. Runtime V2 launches now
accept only the garden-path track. The recovery location and removed paths are
recorded in `scope-cleanup-20261013.json`; the verified backup is outside this repo.

Original `source.json`/metadata snapshots of retained GPS runs remain unchanged:
they are immutable provenance and can contain code formerly shared with the
removed track. Some simple-sentence QA helpers remain solely because the saved
GPS bridge and original V4 runner import them. They are not an independent
research objective. No new general role experiments belong in this repository.
