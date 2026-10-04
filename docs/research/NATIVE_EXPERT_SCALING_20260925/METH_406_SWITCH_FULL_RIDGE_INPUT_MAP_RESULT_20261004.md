# M406: fixed full affine ridge input interface rejected

Frozen9518b9f. Raw SHA256 `f1220d8c7eaa4fcc0d0749952c9eeec49cfebb937f679a4969f3599110cd6fd0`.

ALL7 apparatus gates PASS, ALL12 local input eligibility FAIL. ONE full768 affine
ridge map per bank, development-only penalty=(1e-5*smax)^2, all96 consumed362
paired cases/18 development and6 validation books. All trace keys/SHA and twelve
development spectra exact405. No hyperparameter/rank grid or validation selection.

| Stack | Bank | Median relativeL2 | 95th relativeL2 | Error / mean-only |
| --- | ---: | ---: | ---: | ---: |
| encoder | 0 | 0.520769 | 0.843924 | 0.721520 |
| encoder | 1 | 0.755589 | 0.923104 | 0.964662 |
| encoder | 2 | 0.703522 | 0.911740 | 1.013816 |
| encoder | 3 | 0.796966 | 1.006074 | 1.050111 |
| encoder | 4 | 0.850183 | 1.003314 | 1.038641 |
| encoder | 5 | 0.865864 | 1.087300 | 0.897453 |
| decoder | 0 | 0.979056 | 2.193769 | 2.244684 |
| decoder | 1 | 1.081935 | 1.482799 | 2.490664 |
| decoder | 2 | 1.319020 | 1.712496 | 2.935276 |
| decoder | 3 | 1.530408 | 2.055738 | 2.767290 |
| decoder | 4 | 1.721145 | 2.426196 | 3.069670 |
| decoder | 5 | 1.922550 | 2.648756 | 3.065412 |

Median errors0.520769-1.922550,95th0.843924-2.648756 exceed0.25/0.50 bounds.
Squared-error ratio0.721520-3.069670 exceeds0.50 in every bank. All decoder
maps are worse than development-mean-only prediction on these validation books.
Finite/F32-vs-F64 checks pass; this is a scientific screening rejection.

MAIN6.078000s /maximum checkedRSS262623232B;
twelve retained failed F32 matrix/mean maps28,385,280B;7,077,888 matrix coefficients.
No source weight reads/model inference/network/GPU/T4. No native cost/DRAM claim.

Decision: reject THIS fixed full affine ridge interface before source function/
output-space/selector/model construction. Identity403, rank32 PCR404 and this
full ridge map are closed on their stated calibration/local criteria. The
unregularized full405 map was never fitted due conditioning. No broad claim
that all nonlinear/other-data/weight-based alignment is impossible. More affine
tuning needs a new stated uncertainty and independent validation, not gate easing.

Return to another-family applicability before extending this same-family union.
Ling-mini-2.0 current official metadata identifies20/D2048/256/top8/shared512
and GQA/QK norm/half RoPE/sigmoid-group routing. It is not useful>256 proof.
Proposed407 verifies immutable actual tensor headers/custom source/config/index,
all bank/core/head/control bytes and an explicit complete active cost ledger
before source acquisition or generic port. Actual pretrained quality/native
reference/operators/adaptation/usefulness/rate/physical DRAM remain required.
