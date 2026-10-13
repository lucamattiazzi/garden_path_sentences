# Historical results audit

Recomputed from saved JSON files; no checkpoint inference was rerun.
Item-bootstrap intervals use 10,000 draws, seed 0. Random-head comparisons are descriptive,
not tests of absence of a circuit or corrections for the complete selection procedure.

## Baseline integrity

- **qwen4b**: 2 non-finite fields; changed/missing baseline items: [5, 11].
- **qwen14b**: 0 non-finite fields; changed/missing baseline items: none.
- **qwen32b**: 0 non-finite fields; changed/missing baseline items: comparison unavailable.
- **gemma3_1b**: 0 non-finite fields; changed/missing baseline items: none.
- **gemma3_4b**: 0 non-finite fields; changed/missing baseline items: none.
- **gemma3_12b**: 0 non-finite fields; changed/missing baseline items: none.
- **gemma3_27b**: 0 non-finite fields; changed/missing baseline items: none.

## Continuation proxy and sensitivity

Historical ‘canonical’ and ‘novel’ labels do not certify corpus novelty or linguistic validity.

| Model | Set | n | Mean [95% CI] | Median | Leave-one-out mean range | Most influential omitted item |
|---|---|---:|---|---:|---|---|
| qwen4b | canonical | 10 | 1.748 [0.727, 2.848] | 1.410 | 1.341–2.000 | #23 → 1.341 |
| qwen4b | novel | 10 | 1.097 [-0.190, 2.370] | 0.801 | 0.728–1.522 | #52 → 1.522 |
| qwen14b | canonical | 10 | 1.280 [0.186, 2.404] | 1.090 | 0.997–1.627 | #18 → 1.627 |
| qwen14b | novel | 10 | 0.999 [-0.429, 2.393] | 1.252 | 0.641–1.366 | #52 → 1.366 |
| gemma3_1b | canonical | 10 | 1.698 [-0.299, 3.902] | 1.305 | 0.901–2.268 | #23 → 0.901 |
| gemma3_1b | novel | 10 | 1.657 [0.450, 2.863] | 1.489 | 1.326–1.984 | #53 → 1.326 |
| gemma3_4b | canonical | 10 | 1.953 [0.410, 4.039] | 1.130 | 1.049–2.296 | #23 → 1.049 |
| gemma3_4b | novel | 10 | 1.144 [-0.422, 2.769] | 1.162 | 0.626–1.504 | #53 → 0.626 |
| gemma3_12b | canonical | 10 | 1.519 [0.010, 3.527] | 0.514 | 0.672–1.849 | #23 → 0.672 |
| gemma3_12b | novel | 10 | 1.818 [0.391, 3.328] | 1.441 | 1.365–2.144 | #53 → 1.365 |
| gemma3_27b | canonical | 10 | 1.477 [-0.156, 3.811] | 0.629 | 0.437–1.875 | #23 → 0.437 |
| gemma3_27b | novel | 10 | 1.758 [0.232, 3.484] | 0.923 | 1.230–2.063 | #55 → 1.230 |

## Paired scale comparisons

Differences are larger minus smaller model, matched by item ID.

- gemma3_27b − gemma3_1b, canonical: -0.221 [-1.660, +1.052].
- gemma3_27b − gemma3_1b, novel: +0.101 [-1.199, +1.290].
- qwen14b − qwen4b, canonical: -0.468 [-1.325, +0.451].
- qwen14b − qwen4b, novel: -0.098 [-0.854, +0.634].

## Random-head reference checks

- **qwen14b**, NV Δ: sampled range [-0.067, 0.088], n=60.
  - L10.H0: -0.079; controls -0.000. Outside the sampled range; no confirmatory significance claim.
  - L10.H11: +0.110; controls -0.011. Outside the sampled range; no confirmatory significance claim.
  - L10.H25: -0.134; controls +0.006. Outside the sampled range; no confirmatory significance claim.
  - L21.H28: -0.090; controls +0.025. Outside the sampled range; no confirmatory significance claim.
  - L3.H7: -0.072; controls -0.009. Outside the sampled range; no confirmatory significance claim.
  - L3.H35: +0.090; controls +0.003. Outside the sampled range; no confirmatory significance claim.
- **gemma3_1b**, NV Δ: sampled range [-0.584, 0.638], n=60.
  - L2.H3: -1.832; controls +0.072. Outside the sampled range; no confirmatory significance claim.
- **gemma3_1b**, continuation: best 0.176, max-of-8 p95 0.732; exceeds reference: False.
- **gemma3_4b**, NV Δ: sampled range [-0.212, 0.234], n=60.
- **gemma3_4b**, continuation: best 0.133, max-of-16 p95 0.217; exceeds reference: False.
- **gemma3_12b**, NV Δ: sampled range [-0.112, 0.064], n=60.
  - L0.H13: +0.110; controls -0.009. Outside the sampled range; no confirmatory significance claim.
  - L26.H4: +0.099; controls +0.041. Outside the sampled range; no confirmatory significance claim.
  - L26.H8: +0.070; controls +0.030. Outside the sampled range; no confirmatory significance claim.
- **gemma3_12b**, continuation: best 0.137, max-of-32 p95 0.183; exceeds reference: False.
- **gemma3_27b**, NV Δ: sampled range [-0.123, 0.086], n=60.
  - L54.H14: +0.087; controls -0.027. Outside the sampled range; no confirmatory significance claim.
- **gemma3_27b**, continuation: best 0.155, max-of-64 p95 0.294; exceeds reference: False.

## Patching checks

- **qwen4b**: gap 1.422; final critical-state recovery 1.000; peak ambiguous-word recovery 0.627.
- **qwen14b**: gap 1.140; final critical-state recovery 1.000; peak ambiguous-word recovery 0.630.
- **gemma3_1b**: gap 1.677; final critical-state recovery 1.000; peak ambiguous-word recovery 0.576.
- **gemma3_4b**: gap 1.548; final critical-state recovery 1.000; peak ambiguous-word recovery 0.505.
- **gemma3_12b**: gap 1.669; final critical-state recovery 1.000; peak ambiguous-word recovery 0.452.
- **gemma3_27b**: gap 1.617; final critical-state recovery 1.000; peak ambiguous-word recovery 0.586.

Final-state recovery for a single-token target copies the source readout by construction.
Intermediate patching effects do not isolate syntactic information.

## Provenance

98 input files are fingerprinted in `legacy-audit.json`.
Historical logs do not record enough checkpoint/runtime provenance to certify rerun equivalence.
