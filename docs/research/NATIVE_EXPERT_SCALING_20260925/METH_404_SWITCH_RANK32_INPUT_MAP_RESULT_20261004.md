# M404: fixed rank32 input interface rejected

Frozen653ad03. Raw SHA256 `d153bcd12ef3be6fe7264b6b41f666b3abff9508aa7181c58afa007a20185cf6`.

ALL6 apparatus gates PASS; ALL12 input-eligibility gates FAIL. One fixed rank32
affine PCR per bank, development18 books only; validation6 predeclared books.
No rank/hyperparameter search. All consumed362 case0 documents; calibration
holdout is not new original-relative model quality. Trace SHA/pairing exact403.

| Stack | Bank | Median relative L2 | 95th relative L2 | Error / mean-only error |
| --- | ---: | ---: | ---: | ---: |
| encoder | 0 | 0.657538 | 0.775729 | 0.865628 |
| encoder | 1 | 0.741241 | 0.887214 | 0.935860 |
| encoder | 2 | 0.653361 | 0.843963 | 0.904253 |
| encoder | 3 | 0.726021 | 1.091208 | 0.900842 |
| encoder | 4 | 0.819737 | 0.929768 | 0.904968 |
| encoder | 5 | 0.872177 | 0.962501 | 0.843537 |
| decoder | 0 | 0.712320 | 0.851359 | 0.648420 |
| decoder | 1 | 0.648790 | 0.728013 | 0.811860 |
| decoder | 2 | 0.740829 | 0.822931 | 0.839194 |
| decoder | 3 | 0.845144 | 0.922164 | 0.786336 |
| decoder | 4 | 0.885695 | 0.945813 | 0.703915 |
| decoder | 5 | 0.896625 | 0.978143 | 0.633437 |

Median errors0.648790-0.896625 exceed0.25;95th0.728013-1.091208 exceed0.50.
Squared error0.633437-0.935860 of development-mean-only prediction exceeds0.50.
These predeclared local gates fail despite modest improvement over mean-only.
All rank32 conditioning and finite/F32-vs-F64 numeric checks pass. Do not promote
from development PCA energy or improved error relative to raw coordinate identity.

MAIN1.578000s; maximum checkedRSS102068224B.
Actual twelve F32 factors/means2,433,024B;589,824 active matrix coefficients
(two768x32/32x768 factors per bank). Frozen protocol incorrectly stated twice
the active count; raw/controller arithmetic is correct. See scope clarification.
NPZ archive overhead/hash recorded separately. BLAS1, no source-weight/model
inference/network/GPU/T4. No accepted-rate or DRAM measurements.

Decision: reject THIS fixed rank32 interface before actual transplanted function,
output alignment, selector or model construction. Does not reject all learned
cross-source transfer. Twelve factors are retained diagnostic failed maps.

Next proposed405: more consumed cases with SAME source/prefix pairs, preserving
the18/6 book split, to test actual rank/conditioning of a possible full768 affine
input map. Four cases/book give2088 encoder/1008 decoder development positions;
counts alone do not guarantee independent rank. This new full-basis hypothesis
must have new capture/fit protocol and local gates before observations. Neither
success nor failure inherits old model quality/rate or proves extra capacity.
