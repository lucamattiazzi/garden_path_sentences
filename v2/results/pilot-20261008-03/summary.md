# Descriptive results: pilot-20261008-03

Model: `google/gemma-3-1b-pt`; revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`.
GPU: NVIDIA A10; inference/analysis wall time: 90.7 s.

All results are exploratory. Layer indices are zero-based.
Intervals are family-bootstrap, pointwise, with no layer-selection correction.

## Incremental prediction

Positive cost means the word was less expected without the comma.

| Word position | Mean cost, nats | 95% interval | Families |
|---|---:|---|---:|
| pre_disambiguation | 4.596 | [3.952, 5.332] | 8 |
| post_disambiguation | -0.095 | [-1.344, 0.987] | 8 |

## Bidirectional patching

Mean change in recipient log P(word), nats. Lexical donor changes the first actor only.
The final block is a readout-copy control; exclude it from localization claims.

| Layer | Site | Recipient | Syntax/punctuation donor | Lexical donor |
|---:|---|---|---:|---:|
| 0 | pre_disambiguation | clean | 0.014 | 0.005 |
| 6 | pre_disambiguation | clean | 0.005 | -0.087 |
| 12 | pre_disambiguation | clean | -2.444 | -0.101 |
| 18 | pre_disambiguation | clean | -4.512 | -0.187 |
| 24 | pre_disambiguation | clean | -4.609 | -0.179 |
| 25 | pre_disambiguation | clean | -4.596 | -0.182 |
| 0 | pre_disambiguation | ambiguous | 0.026 | -0.005 |
| 6 | pre_disambiguation | ambiguous | 1.209 | 0.033 |
| 12 | pre_disambiguation | ambiguous | 4.594 | -0.055 |
| 18 | pre_disambiguation | ambiguous | 4.644 | -0.255 |
| 24 | pre_disambiguation | ambiguous | 4.620 | -0.246 |
| 25 | pre_disambiguation | ambiguous | 4.596 | -0.262 |
| 0 | post_disambiguation | clean | 0.023 | 0.015 |
| 6 | post_disambiguation | clean | 0.044 | -0.040 |
| 12 | post_disambiguation | clean | 0.642 | 0.086 |
| 18 | post_disambiguation | clean | 0.295 | -0.220 |
| 24 | post_disambiguation | clean | 0.137 | -0.279 |
| 25 | post_disambiguation | clean | 0.095 | -0.286 |
| 0 | post_disambiguation | ambiguous | -0.043 | -0.036 |
| 6 | post_disambiguation | ambiguous | -0.049 | -0.221 |
| 12 | post_disambiguation | ambiguous | -0.231 | -0.359 |
| 18 | post_disambiguation | ambiguous | -0.199 | -0.701 |
| 24 | post_disambiguation | ambiguous | -0.132 | -0.744 |
| 25 | post_disambiguation | ambiguous | -0.095 | -0.751 |

Maximum identity-patch discrepancy: 0 nats.

Full curves, individual scores, patch norms and intervals are in the JSON artifacts.
These measurements do not establish a syntax-specific circuit, online role revision,
or a model-size trend. See the protocol before drawing those conclusions.
