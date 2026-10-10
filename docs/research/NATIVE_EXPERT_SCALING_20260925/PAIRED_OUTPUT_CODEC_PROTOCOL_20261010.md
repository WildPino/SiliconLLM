# One output-weighted paired rank codec: frozen protocol

10 October 2026. FIT-only control, before any codec fitting/quality observation.
Goal INCOMPLETE. No source history, optimizer, full engine, DEV/RESERVED or T4.

## Uncertainty and decision

[Actual51/readout attribution](ORIGINAL_LATENT_READOUT_ATTRIBUTION_RESULT_20261010.md)
identified a coupled raw-projection/norm/head defect. [Coherent cached FIT
features](SOURCE_CACHED_FIT_RESULT_20261010.md) are now qualified:24 cases,
4422 old tokens/logits and same-call norm/head EXACT; full audit all hashes,
411 parameters,4422 frames and548 scalar head witnesses PASS. Reuse these
immutable observations, the source head and original final norm/matvec bodies.

New variable: paired encoder/head selected on actual postnorm features by one
explicit output-distribution quadratic, with a representation admissible under
original RMSNorm. This is not another unchanged recovery dose, width/rank/scale
grid or arbitrary-head oracle. A readable injected representation is a necessary
control before causal SSM/SWA/ternary recovery; it is not a converted chatbot.

The test finishes all FIT cases and the independent stored audit even if
scientific quality or precision gates fail. Numerical decomposition/first
mechanical faults are retained. No completed source capture or training replay.

## One calibration algorithm

[Derivation](OUTPUT_WEIGHTED_RANK_CODEC_DERIVATION_20261010.md).
Source dimension2048, rank255, target256, vocab65537, epsilon1e-5, s=.01953125.
Use actual observed BF16 f and W decoded to reals, M=sW. Each case has mass1/24;
each of its m labels mass1/(24m), so each of12 FIT domains has mass1/12.
No mean subtraction/bias fitting. Compute F64

    C = sum_j w_j f_j f_j^T,
    qbar = sum_j w_j softmax(actual_cached_logits_j),
    G = M^T diag(qbar)M - (M^T qbar)(M^T qbar)^T.

This fixed mixture Fisher is NOT average context Fisher, exact KL or a claimed
KL bound. Full-V KL/argmax remain separate observed quality criteria.

C and A eigensystems use NumPy F64/OpenBLAS6 CPU threads. G/source-head decoder
factorization and compact real scores use Torch F64 CUDA, TF32off, seed0. Block
head4096 rows; compact labels16. No autograd or optimizer. Symmetrize C,G,A.

PSD numerical allowance: smallest C/A eigenvalue >= -1e-10 times largest.
Covariance support retains lambda_C>1e-12*lambda_C_max; rank>=255; discarded
positive covariance trace<=1e-8 total. Save raw eigensystems before adjudication.
Clamp only numerical negative C eigenvalues to zero for square roots; do not
silently ridge/invert tiny values. Let C_s be retained covariance, C_drop its
discarded positive part, R=C_s^(1/2), R_plus=C_s^(+1/2), A=R G R.
Top255 A eigenvalues must exceed1e-12 its maximum. Choose U descending, largest
absolute entry of each column positive (first coordinate breaks ties). Pair

    B = U^T R_plus, T=R U, K=M T, phi=B f.

This supplies the rank optimum restricted to the declared covariance support.
Direct actual-FIT quadratic must agree with

    J = trace(G C_drop) + sum_(i beyond top255)lambda_A_i

within1e-7 relative to trace(G C). With numerical negative C values, the clamp
is an explicitly bounded numerical approximation, not an exact PSD certificate.
Discarded energy/condition, whitening and both spectrum tails remain reported.

## Original RMS representation and native arithmetic

rho=1; a=2 max_FIT||phi||/sqrt(256), no gain grid. Construct

    u=[phi/a; sqrt(256-||phi/a||^2)], gamma_C=ones,
    W_C=a sqrt(1+epsilon)[K,0].

Save F64 B/K/phi and actual F32 u/head/gamma. The domain is empirical,
||phi/a||<=16; FIT radius<=8 by gain rule. No future-input guarantee; no clipping
or DEV refit. Carrier reachability/conditioning by the causal core is unproved.
Head F32 finite required. Original state/norm/head are F32, not F16.

Compile isolated DLL with unchanged hsum256/dotf/matvec/rmsnorm bodies from
engine.c, D256/V65537, OMP_PFOR empty. clang -O3 -mavx2 -mfma -march=znver2
-shared, no fast-math. Save compiler/source/body hashes/DLL/log before binding.
This consumer ONLY injects final states into final norm/head. It executes
4422 actual original final-readout rows, not the complete causal engine.

Two arithmetic views of the SAME pair: real F64 K phi versus original F32
norm/matvec(u). Save both full-V score streams losslessly, native normalized
features, phi/state and case metadata before reporting any scientific verdict.
No second codec or gain is selected after evaluating these scores.

## Frozen scientific and precision gates

For BOTH real and native views, donor-relative case-average KL<=.01 and
disagreement<=.01; EVERY case KL<=.05/disagreement<=.05; EVERY domain case
average KL<=.05/disagreement<=.05. These source-readout fidelity gates are
FIT screens; eventual held-out generation/tasks and useful50 remain required.

Numeric: whitening B C B^T versus I and B R versus U^T relative<=1e-8;
surrogate identity1e-7; native normalized features versus analytic F64 norm of
the ACTUAL stored F32 state relative<=1e-5 per case; maximum all-V native versus
real score absolute<=1e-4; case-average KL(real||native)<=1e-6; every case
native-to-real ID disagreement<=.01. Metrics double logsum cross-check1e-10.
Finite states/scores and dimensions are required. Native rounding is distinct
from representational loss, even if both are small or both fail.

Decision: numeric FAIL -> PAIRED_CODEC_NUMERIC_FAIL; numeric PASS but real/native
quality failure -> PAIRED_CODEC_FIT_QUALITY_FAIL; all PASS ->
PAIRED_CODEC_FIT_PASS_PENDING_DEV. No quality admission from this FIT result.
All outcomes receive complete audit; no criterion relaxed after observation.

## Independent stored audit, costs and stopping criteria

Every bound input and terminal output hash; exact sealed case records/native
function bodies; all eigensystem residuals/orthogonality/root/encoder identities;
all decoder K rows reconstructed from source M/T; all qbar coordinates through
independent logaddexp normalization;32 C and32 G scalar math.fsum witnesses;
all encoded phi/states; all real/native scores verified by CPU F64 matrix
arithmetic; every full-V metric through independent logaddexp/dot calculation;
all flags/decisions/counts and held resource caps. No re-fit, eigensolver replay,
native binary call or source history. Honest audit counts: one decoder
factorization verified and8844 stored linear head rows reconstructed.

Forecast a few minutes for FIT statistic/factorization/scoring plus full audit;
no exact runtime claim before measurement. Held fit900s/reserve30s, OS worker+
launcher<=6GiB, GPU allocated<=4GiB/reserved<=5GiB, namespace+result<=5GiB,
log<=8MiB. Score payload3,477,655,368B (4422x65537x(8+4)); matrices/metadata extra.
Held audit900s/OS<=3GiB/result<=2MiB/log<=8MiB/no GPU. Native compile preparation
45s subprocess cap; observed2.875s/no readout call. Input hashing is included
in held families; binder/compile preparation counted separately.

Hard deadline/resource/unexpected-child/malformed input faults preserve the
actual first stage/counters/completed cases. Repair -> new code/binding/namespace,
missing-only completion; do not reinterpret observation timeout as exit. No
concurrent owned benchmark; exact publisher daemon exempt, uploader must finish.
Retain exact unified session/Win32 PID/create_time until terminal and gone.

## Reproduction and next branch

Tool `benchmarks/native_expert_scaling/paired_output_codec.py`, isolated Python
3.12 flags `-I -S -B -X utf8`. --prepare produces native directory/preparation
JSON; --bind uses --binding/--protocol/--native-preparation. Freeze code/protocol/
binding before --binding/--binding-sha/--freeze/--directory/--out launches fit.
After terminal, same frozen tool --audit/--source-result/--out with same binding.

Freeze fitted pair and complete audit before any matching new DEV observation;
reuse the already qualified53-label DEV case when exact source custody matches.
Scientific FAIL still deserves a held-out/geometry diagnosis under fixed rules;
do not jump to unchanged joint training or a universal D256 conclusion.
Useful original SSM/SWA/LUT/ternary chatbot, RAM-driven n/structured CPU ID-mass/
physical DRAM, same-artifact>=50token/s and second family/scale remain the goal.
