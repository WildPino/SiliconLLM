# FIRST actual changed-quadratic features/covariance and independent audit

Prospective8 October2026, implementing [NEXT](CHATBOT_QUADRATIC_KERNEL_NEXT_20261008.md).
Uncertainty: qualify actual256 quadratic columns/covariance in ONE source-informed
rank16 private format before response fitting. Exact source R/T BF16 factors
are already independently qualified. No response coefficients or new chatbot.

Original3845 FIT_x/weights/W/mu/r and old linear G saved kernel are used as
immutable inputs; no original geometry/linear feature/Gram/factor replay.
No source/core archive/forward/H/J/responses, old predictions or optimizer.
Original per-occurrence masses promoted F32->F64, multiplicity weighting sum3166.
No novel states/factor labels/scales here. Old linear mu_F64/r_F64 remain intact.

Encode NEW quadratic mu32=float32(mu),r32=float32(r) BEFORE features;
z=(x−promoted mu32)/promoted r32, F64 smooth formula. Native rounded F32
intermediates are not asserted equal. For e=0..15,j=0..15:
Qraw[:,16e+j]=W_e*(R_ej z)*(T_ej z),
sigma_j=sqrt(sum a*Qraw_j^2/3166), B_Q=sqrt(a)*(Qraw/sigma).
ALL256 sigma finite>1e-12; any dead column closes this design with saved Qraw/
sigma, no column drop/rank/scale retry. Save new center/scale asF32, Qraw/sigma/
B_Q/G/L asC-orderF64. Input R/T rows copied unchanged, no re-encoding.

NEW G=saved_linear_G+B_Q B_Q^T, alpha=trace(G)/(1000000−1), same once-only
nominal real feature-Gram condition rule. Require positive finite trace/alpha,
symmetry<=1e-12, trace closure relative<=1e-12, quadratic trace versus256*3166
relative gap<=1e-12. Cholesky new G+alphaI, positive diagonal/reconstruction
<=1e-10. No old factor or parameter inverse. Nominal real1e6 upper deduction
is not measured/certified condition2 of rounded saved matrices. Factor/scales
alone do not demonstrate useful capacity or response fidelity.

FIRST audit BEFORE fitting: independent scalar struct F32 constants, BF16
exponent/mantissa promotion, gate-squared/gate-up form with alternate matrix
orientation; ALL Qraw relative discrepancy<=1e-11 plus absolute1e-11.
ALL sigma by math.fsum weighted scalar energies; B_Q by scalar normalized
columns. Then SUM sixteen separate16-column Gram blocks plus reused oldG,
ALL-matrix relative<=1e-11 plus absolute1e-11, positive saved factor/all lower
triangle/reconstruction<=1e-10. Scalar trace/alpha checks rel2e-12/abs1e-12.
No original/producer kernel acquisition rerun or new response/fit label.
Save FIRST audit provenance/metrics/decision; audit failures retained.

CPU-only Python3.12.10 isolated/no-site/no-bytecode, five restricted
NumPy2.4.6/psutil7.2.2 roots/OpenBLAS1/CPU0..10/launcher11. Primary180s/family300s/
conservativeOS4GiB/output512MiB/log2MiB through actual held-handle exit;
FIRST audit same time/memory/log, empty directory/output1MiB. No GPU/T4/network/
new resources. Exact actual code/protocol/input/runtime/full Git freeze BEFORE
new columns/covariance; FIRST audit/typedUTC known-instance closure before fit.
Common launcher retains first faults/partials and three foreign tracked SHAs;
launcher final receipt/stdout tail outside its last peak snapshot disclosed.

Necessary ALL24 budget remains295104512MAC/592186464 logicalB/1080411488 stored
payload; no native/physicalDRAM/rate. Old source-D projection residual92.1%..95.5%
does not substitute for free-C response fit. Original novel1%/ALL-category3%
and full export/native/fresh own-history/tasks+accepted50 SAME artifact/useful
n/RAM/LUT winner+mass/actual further families/scales remain mandatory.
Only fully qualified new design permits ONE separately frozen convex response
fit on reused labels/RHS, no alpha/lambda/rank/codec/epoch ladder.
Exact actual command stored in terminal.command via chatbot_directional_launch.py.
