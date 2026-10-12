# Necessary input-recovery geometry,not recovered source states

13 October2026. Execute after full-width timing ends;no benchmark interference.
All3 fixed sites0/12/23,exact source in_proj first3072 gate rows x2048 columns,
actual stored BF16 coefficients promotedF64. No learned/DEV row or rank choice.
Measure full Gram/eigenbasis/spectrum/condition/inverse noise gain. MuP gate
and upstream input multiplier are nonzero scalars:rank/condition unaffected,
absolute scaled inverse gain/actual BF16 buffer rounding NOT identified here.

Unrounded linear map gate=G*h+e,full column rank implies least-squares bound
||h_hat-h||<=||G^+||*||e||. Rank/condition is only a NECESSARY geometry check,
not evidence of small actual BF16 hidden error or a closed compact encoder.
Current captured gate is BF16 after source projection/MuP rounding;future
recovery needs actual multiplier dtype/quantization intervals and OTHER operand
replay. No source gate values used to fit/test this matrix-only preflight.

Producer:all Gram/eigenvectors/eigenvalues/singular values saved. Numerical
Gram1e-12/orthogonal max1e-10/eigen residual1e-10/negative PSD relative1e-10.
Fixed robust rank counts sigma>1e-6*sigma_max;full2048 andcondition<=1e6 permits
an OFFLINE recovery preflight,not recovery admission. No numerical floor
silently establishes positive rank;zero/negative eigenvalues retained.

Independent audit:exact source rows,full blockwise Gram,eigenbasis/residual,
direct rectangular source SVD ALL2048 singular values (normalizedsigma error
1e-8) and classification/inverse gain/multipliers/hash controls before-after.
If weak/deficient,select true in_proj input capture;otherwise separately freeze
uncertainty-qualified recovery protocol. No model forward/GPU/T4/native/RESERVED/
hidden recoveries/optimizer/quality/n/speed claim. Producer900s/audit1200s,
OS6GiB/output512MiB,fresh tracked E namespace,CPU0..5/sixBLAS threads.
