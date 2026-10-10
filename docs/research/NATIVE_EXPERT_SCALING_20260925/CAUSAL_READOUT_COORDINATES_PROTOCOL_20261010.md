# Stored original C causal-coordinate qualification: frozen protocol

10 October2026. One new stored observable; no fit or trajectory replay.
[Decision and ensuing converter](CAUSAL_READOUT_COORDINATE_NEXT_20261010.md).

## Inputs and scope

Exact actual27 packed head/final norm and audited full prefix logits,48 FIT/DEV
histories with adopted offsets handled explicitly. Select all existing output
labels only:4422FIT+4386DEV=8808. No teacher distributions consumed or quality
selection on DEV; no source weights/old masters/optimizer/GPU/native call.
Reuse already qualified packed-only original C baseline and its completed audit.
Current new paired-head probe/full audit is COMPLETE and exposes encoder room.

Read H=old head F32 decoded to F64, gamma=old final norm. One thin QR H=QR,
then one256x256 SVD R=U diag(s) VT. Save all factors and calibration durably.
Reconstruct fhat by solve(R,(z Q)^T), no ridge/pseudoinverse fallback, using
exact old row_offset + metadata positions. Preserve features and residuals
per case before progressing. This is approximate old normalized C state, not
bit-exact raw state. Lost directions are not silently invented.

Numerically full rank requires all256 s>=1e-8 smax AND positive sigma_H_lower.
If that prerequisite fails, keep factors, report NOT_QUALIFIED with0 reconstructed
labels, and audit the complete supported calculation. No criterion-triggered
omission of the full audit; inversion itself has no defined admitted result.
If eligible, reconstruct ALL8808 even if later residual/bound gates fail.

## Error algebra and assumptions

From measured factor residuals and Frobenius orthogonality errors:
sigma_R_lower = sqrt(1-err_U) sqrt(1-err_V) smin - ||R-U diag(s) VT||_F,
sigma_H_lower = sqrt(1-err_Q) sigma_R_lower - ||H-QR||_F.
Square roots use max(0,1-err); a nonpositive lower precludes recovery admission.
These follow from singular-value perturbation and norm inequalities. All are
F64 numerical evaluations, not directed-rounding interval certificates.

For conventional unexceptional F32/FMA rounding, unit u=2^-24, gamma_n=n u/(1-n u).
Original dotf has32 FMA steps per lane plus7 scalar summations; gamma256 is a
conservative allowance. Original scalar RMS sum gets gamma512; final reciprocal/
sqrt/products have a conservative four additional unit-rounding factors.
Native norm upper C=16 max|gamma| (1+u)^4 / sqrt(1-gamma512).
Native dot error L2 upper=gamma256 C ||H||_F.
For EVERY returned fhat, using full residual r=H fhat-z:
||fhat-f_native|| <= (||r||_2 + native_dot_error_upper)/sigma_H_lower.
The relative bound divides by ||fhat||-absolute_bound when positive, else inf.
Bound derivation is conditional on the stated F32 arithmetic model; factor/
residual arithmetic is independently verified F64, not interval-certified.
No claim that stored heads provide exact native bits or exceptional FP flags.
Actual new-head/native histories must later validate any fitted artifact.

## Frozen gates

QR/SVD relative<=1e-10; all Q/U/VT Frobenius orthogonality errors<=1e-10.
Full-V maximum score residual<=1e-4; every feature error L2 upper<=.05;
every relative bound<=.01; recovered norm<=C+.05. Pass is expressly APPROXIMATE.
All metric/decision verification absolute1e-10 + relative1e-10, stored normal
equations zQ=fhat R^T absolute<=1e-9. No head quality/task/50-token admission.
No fit or downstream optimizer until full factor/custody/feature audit passes.

## Cost, stopping, provenance, complete audit

Code/protocol/inputs frozen before observation. New unique consumed score files
are sealed, not the unrelated87GB collection. Forecast tens of seconds to a few
minutes, with hashing part of cost. Held600s including pre/post hashes; worker
reserve60s,OS3GiB/output256MiB/log8MiB. Separate full audit600s/OS3GiB/output2MiB/
log8MiB/noGPU. No simultaneous owned benchmark; foreign publisher untouched.
Record exact live handles/PID creation and terminal receipts. First mechanical
fault preserves stage, calibration/factors, complete case records and error;
repairs use a new namespace/code/binding with no completed calculation replay.
Stop only completion, prerequisite for inversion, or mechanical/resource cap.

Stored audit checks all input/output hashes, exact old head/norm/positions,
complete QR/SVD reconstruction and orthogonality without refactorization,
ALL8808 feature normal equations and full-V residual/bound/per-case maxima,
384 independent math.fsum scalar head witnesses, original four readout function
bodies in engine/header, decisions, exact calls/resources. It never runs native,
source, learner, QR/SVD, optimizer or GPU. Count full stored head reconstructions
separately for worker/audit; factors are one QR +one SVD in worker,zero in audit.

Next, only if qualified: a newly frozen FIT-only convex J255x256 categorical
encoder and offline head H_new=A J, fixed current native core/final norm.
Quality of the actual new C artifact remains the end criterion; reconstructed
feature learning is diagnostic/conversion work, not a substitute for generation,
useful conditional experts/routing/DRAM/50 and cross-family validation.
