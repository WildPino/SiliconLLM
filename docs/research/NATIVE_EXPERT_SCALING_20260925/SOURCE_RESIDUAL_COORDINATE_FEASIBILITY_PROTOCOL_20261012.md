# FIT-only layer-specific angular residual coordinates

12 October2026. Prospective control, full goal ACTIVE/INCOMPLETE. New variable:
source RESIDUAL basis at each boundary, not recurrent-channel/state pruning,
head-only fitting or another epoch on the failed global-P shadow.

Reuse existing raw BF16 source residuals,48 matched histories,seven boundaries
0/4/8/12/16/20/24.24 FIT cases11148 positions,24 DEV11399 positions. Source
parent capture runtime/identity gaps remain; actual stored projection audit and
all bytes adopted. No source/shadow/GPU/generator/native/optimizer/T4 call.

## Fixed estimator and numerical checks

For each FIT case normalize every raw h by sqrt(mean(h^2)+source epsilon1e-5).
Equal-case uncentered covariance G=mean_cases(U^T U/T). Top256 eigenvectors are
a boundary-specific orthonormal decode basis; maximal-absolute-entry sign fixed.
This optimizes only the FIT angular quadratic reconstruction objective. Save
complete covariance/eigensystem/selected F64 and F32 bases; no DEV selection,
regularization/grid/search, centering or teacher-label adaptation.
All full eigenvectors orthogonality<=1e-10,eigen-equation/trace relative residual
<=1e-10,PSD numerical tolerance1e-10 relative. F32 basis promoted to F64 for
reconstruction metric comparison<=1e-5; no F32 runtime arithmetic/C claim.

All48 cases/boundary evaluate new versus existing fixed P:angular relative RMS,
mean-removed angular relative RMS,retained angular energy and actual source/
restricted reconstructed RMS ratio (with epsilon). Report complete negative
results; do not infer task information from squared Euclidean energy.

Predeclared TRIAGE proxies on DEV at EVERY boundary:mean angular relative RMS
<=10%,worst<=20%,mean centered relative RMS<=10%. These are reconstruction
budgets, not chatbot/quality or a capacity theorem. A pass only earns composed
operator testing with actual native cost. Failure motivates changed read/write
metric, conditional/nonlinear/affine encoding or larger active representation.
Keep any mechanically valid failed spectrum as reusable evidence.

## Radius and changing-basis residual transport

Source h=Pz+r has additional radius/direction in r. Exact on-subspace norm/gamma
folding assumes r=0 and does not recover that information. Report measured
source/restricted RMS ratio; no oracle source norm admitted to a deployed model.

With different per-boundary decode P_a,P_b, residual code transport is P_b^T P_a,
not identity. Save six actual256x256 F64 transports/singular spectra/overlaps.
These are coordinate transfers, not a proof the four intervening source blocks
can be replaced cheaply. Future affine charts require explicit centers/bias/
constant-coordinate format, and general dual bases require their actual norm
metric and read/write maps. Include their matrix/storage cost in engine work.

## Budget and custody

CPU NumPy,6 threads,CUDA hidden,900s/OS4GiB/outputs512MiB. Full actual source
stream/protocol/P/runtime inputs before/after exact. Exclusive new outputs;
freeze source/protocol,actual binding and invocation before run. Complete all7
FIT eigensystems/336 case-boundary comparisons/6 transports even proxy failure,
unless resource/numerical execution fault; retain first fault stage/receipt.
Principal runtime binaries,not full DLL tree. Foreign publisher/engine/work
preserved, no timing isolation claim. Full independent stored covariance/
eigen/reconstruction/transport audit required before promoting proxy results.
Full useful conditional LUT/ternary/chatbot/same-artifact50/useful-n/DRAM and
multi-family goal unchanged; no spectral result substitutes for it.
