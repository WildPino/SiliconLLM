# NEXT: changed quadratic features, augmented covariance, ONE fixed transfer fit

8 October2026. Source factor extraction/FIRST audit COMPLETE; new feature
design/kernel/response coefficients are NOT implemented/observed.
[Actual factors](CHATBOT_SOURCE_QUADRATIC_RESULT_20261008.md): sixteen distinct
R/T BF16 bases, eight selected source/core atoms per parent, exact source rows,
all128 choices/688128 scores independently verified. Scalar source-D projection
residuals92.1%..95.5% are a limitation, not a free-C response-class result.
Goal stays complete pretrained CHATBOT -> engine.c, fresh own-history/tasks
AND accepted>=50 SAME artifact/useful n/RAM/LUT winner+mass/DRAM/family-scales.

## FIRST changed-feature geometry, before response coefficients

New uncertainty: whether actual donor-conditioned quadratic features can
augment the qualified affine class in the SINGLE complete-cost rank16 format.
Use immutable original3845 FIT_x/multiplicity weights/FIT_W/FIT_U/mu/r/linear
G and independently qualified actual R_BF16/T_BF16. No source/core forward,
old geometry/Gram/factor/feature experiment/selector/response/optimizer replay.
No donor D or full source archive is needed; the factor bytes are now saved.

Preserve old linear columns Phi_L[e]=W_e[1,(x−mu_F64)/r_F64] exactly as the
saved original G. Encode mu32=float32(old mu),r32=float32(old r) FIRST;
these new F32 center/scale bytes belong to the priced candidate. Define
z=(x−promoted mu32)/promoted r32 in F64 for the smooth conversion field.
This deliberate F32-center formula must not be silently substituted for old
Phi_L or asserted equal to actual rounded F32 inference.

For each parent e, construct ONLY NEW quadratic columns

    Q_raw[i,e,j] = W[i,e]*(R[e,j] z_i)*(T[e,j] z_i), j=0..15,
    sigma[e,j] = sqrt(sum_i a_i*Q_raw[i,e,j]^2 / sum_i a_i),
    Q = Q_raw/sigma; B_Q = sqrt(a)*Q.

Original exact-x weighting sum3166; original per-occurrence F32 mass promoted
to F64, including cross-case equal-x mass differences. Freeze arithmetic,
array order and cutoff BEFORE features. All256 sigmas finite>1e-12 required;
zero/dependent column evidence retained, no dropped columns/reduced rank/retry.
Dependence itself does not invalidate positive-ridge uniqueness. No novel
state or response is a feature-scale/factor-selection label.

NEW augmented covariance

    G_new = saved_G_linear + B_Q B_Q^T,
    alpha_new = trace(G_new)/(1000000−1),
    L_new L_new^T = G_new + alpha_new I.

This follows the SAME prewritten nominal real condition-scale rule, once,
on a genuinely changed feature class; no alpha/lambda/precision/grid search.
It is not a measured/certified condition2 of rounded matrices. Require positive
finite trace/alpha, symmetry<=1e-12, positive factor diagonal and relative
reconstruction<=1e-10. Save Q_raw/Q or B_Q,sigma,mu32/r32,G_new,L_new and typed
SHA/layout receipts. Old G is a reused summand; do not rerun old kernel/factor.
FIRST independent changed-quadratic feature/Gram/factor/scalar/provenance audit
must precede fitting. Avoid calling saved old audits as a new result.

Proposed CPU-only worker180s/family300s/4GiB/output512MiB/log2MiB through exit,
five restricted NumPy/psutil roots/OpenBLAS1/CPU0..10. No GPU/T4/network/new
resources. Actual code/input/runtime binding before features/factor; retain
first faults/partials and close this design on qualification/resource failure.

## ONE subsequent fixed convex response fit, still not executable

Only qualified augmented covariance allows ONE response objective:

    min_Theta ||[B_L,B_Q]Theta−sqrt(a)*(Y−shared)||_F^2
              +alpha_new||Theta||_F^2.

Reuse original cached full-FIT source BF16 Y/shared F64/batchedF32 and the
already saved zero-prior RHS where byte/formula provenance agrees. No old
source/shared/fit predictions/optimizer/minimum replay; this new Θ has new
columns and a new covariance. Solve with saved new factor, no inverse,
fold affine A/bias from old mu_F64/r_F64 and C=Theta_Q^T/sigma. Free C is
broader than the extraction's source-D/scalar projection witnesses.
Factor BF16 rows stay EXACT, C/A encode BF16,bias/center/scale F32. No H-J
prior/regularizer variant here; source curvature informed factor selection.

Frozen numerical stationarity/dual/folded/primal and independent encoding/error
checks must qualify the actual fit; same original response criteria govern.
Reuse source FULL1495 novel exact denominator, not a new source-energy pass.
FIRST64 original novel predictions (both smooth F64 and actual encoded-F32
conversion arithmetic) can supply exact necessary FULL1% rejection. Stop
remaining novel/ALL24/export/native if rejected. These are consumed calibration/
diagnosis, not fresh quality. A passed necessary prefix is not full admission;
original full novel1%/ALL-category3% and complete artifact gates still apply.

Actual fitting code/objective/normalization/codec/arithmetic/inputs/first audit
and finite budget must be frozen BEFORE any new response coefficient/prediction.
Proposed fitter300s/family540s/4GiB/output512MiB/log2MiB; same isolated CPU
runtime. No alpha/atom-count/rank/epoch/codec ladder after outcome. Failed
response recipe remains retained/closed, not a global capacity refutation.

## Return to the complete pipeline

Only qualified actual local response permits ALL24 conversion plus explicit
export and canonical native chat integration. Then fresh EXCLUDED own-history
dialogues/tasks and accepted>=50 SAME artifact; physical LUT winner+mass/DRAM
and useful larger n/RAM; actual further families/~10B/~100B as resources allow.
Native quadratic operator/export are not implemented yet. Logical budget
295104512 MAC/592186464B/1080411488 stored payload remains a necessary dimension
proposal, not measured physical traffic or rate. No admitted compact chatbot.
