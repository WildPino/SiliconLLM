# M444: fixed global output subspaces fail validation; large norm imbalance

Controller frozen24317ce, protocol clarified/resealedbaaf45c BEFORE first import/
forward/eigensolve. Terminal exit0, zero optimizer updates. [Protocol](METH_444_SWITCH_FUNCTION_CONTRAST_PROTOCOL_20261004.md),
[raw](meth444_switch_function_contrast_result.json) SHA256
9fbb825172be8d21a4b40d2c619a9906972d6d0c3bbe1519b487879f28d47545.
ALL8 apparatus gates PASS:24 original complete native replays exact, all128
first-anchor WI/WO outputs exact independent primitive,3072 native functions
collected, covariance/private-energy/eigen/full-recomposition/entropy/KL checks
qualified. Basis fit on18development anchors only,6validation anchors untouched.

Define barF(x) as uniform128-function response mean and d_e=F_e-barF. Fit a
single common output PCA basis from development private responses. Mean and
projection coefficients remain offline ORACLES, not runtime functions.

| Fixed rank | Development private energy | Validation private energy | Mean paired native KL |
| --- | ---: | ---: | ---: |
|0/common only|0%|0%|.212477951|
|32|96.5183%|62.8015%|.096090711|
|64|97.1456%|66.9517%|.080129051|
|128|97.9143%|71.9834%|.064902816|
|256|98.8311%|79.0997%|.045385097|

No nonzero rank passes all4 predeclared geometry gates: development energy>=95%
passes, but validation energy>=95%, mean KL<=.01 and every-anchor mean KL<=.05
all FAIL. Original native uniform interventional JS mean.168167641nats; common
only JS is zero within1e-10, rank256 JS.120830776. JS preservation alone cannot
establish correct identity assignment, truth accuracy or useful routed capacity.

**Close this18-anchor absolute-energy development PCA recipe.** A single basis
of these fixed ranks does not preserve the sampled validation contrasts even
with oracle means/coefficients. This is not an impossibility theorem for all
linear bases, different sampling/metrics or nonlinear/input-conditioned methods.
PCA is optimal for development Frobenius error, not validation or downstream KL.

**Post-outcome diagnosis, not a changed gate:** saved per-anchor energies show
development books0 and14, both decoder position0, contribute44.4991% and43.9362%
of development private squared norm: together88.4354%. Rank32 retainsabout99.66%
for those two anchors, but only51.87..80.46% for the other16development anchors.
The weighted96.52% aggregate therefore does not describe uniform anchor quality.
Validation anchors contain no position0. Both weighting and input coverage are
plausible contributors; this observation does not establish which correction
will succeed. Do not retrospectively call the original recipe PASS.

Next NEW445: a cheap, separately frozen algebraic diagnostic on the exact retained
response matrix. Compare equal-relative-anchor covariance with the old absolute
covariance, and compute each input's oracle local contrast spectrum. This separates
norm domination from intrinsic local dimensionality without new model forwards.
At a fixed input,128 centered functions have rank<=127 by algebra; a cheap method
for predicting an input-dependent subspace is still a separate runtime problem.
Any promising geometry must subsequently pass the source-native posterior gates.

109.937s total/admission12.828s/numerical97.109s,2,062,761,984B peak,
11,106,913,127B hashed,7output archives634,450,124B. Full6x128x32128 F32 native
and all5rank matrices retained, geometry includes all24x128x768 source features.
CPU0/Torch1/BLAS1, no supervised/gradient fit/GPU/T4/network/C edit/rate/DRAM.

No runtime compression savings, complete task quality/SAMEartifact50, useful
RAM-scale n, other family/~100B proof. Qualified original artifacts intact;
goal ACTIVE/INCOMPLETE.440 remains closed; no foreign readout rescue.
