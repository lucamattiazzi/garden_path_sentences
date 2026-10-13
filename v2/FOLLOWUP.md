# Gemma 3 garden-path follow-up

Completed on Modal, 8 October 2026; no new inference in this scope cleanup.
[Protocol](GARDEN_PROTOCOL.md) · [Raw aggregate results](results/garden-followup-20261008-02/garden-followup.json)
· [GPS robustness diagnostics](results/robustness-20261008.json).

Development selected block 17 (zero-based). On six held-out families,
bidirectional factorial patch interaction was 6.671 nats [5.297, 8.305],
versus 0.356 for norm-matched lexical and 0.488 for random-direction controls.
Both directions and every test family had positive interactions.

This endpoint is a difference of **full-completion log-probability margins**,
not surprisal at `was` alone. Unequal continuation lengths affect the reading of
baseline preferences. The block-selection bootstrap shows a broad sensitive
region: block 17 wins 25.64% of development resamples, not a unique syntax layer.
No initial wrong parse or subsequent correction was established by this score.

The next GPS phase V3 uses a single-token diagnostic and separately measures the
actual continuation. The original scientific question remains surprisal at the
actual disambiguating word and the causal mechanism for processing the GPS.
