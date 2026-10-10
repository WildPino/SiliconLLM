# A paired rank codec selected by output geometry

10 October2026. Analytical proposal while coherent FIT capture is LIVE.
No numerical fitting, benchmark, DEV selection or quality admission in this memo.
Reuses [RMS-compatible carrier](RMS_COMPATIBLE_OUTPUT_CODEC_20261010.md) and
[readout Fisher geometry](TARGET_READOUT_GEOMETRY_20261010.md).

## New variable

The old basis balances embedding/head Euclidean Grams; it neither uses actual
postnorm features nor selects directions by their effect on token probabilities.
A new codec can pair an encoder B with its decoder K, rather than compress raw
states and separately project a diagonal norm. Let f_j be observed BF16 source
postnorm features decoded to reals, q_j the actual same-call teacher softmax,
and M=sW, s=.01953125, source BF16 head W decoded to reals.

We want a linear rank-k surrogate K B f_j, then represent B f_j through the
existing256-coordinate final RMSNorm. A causal history/core capable of producing
these coordinates remains a separate question. No final-state information may
be treated as an input available to a deployed compact chatbot.

## One explicit output metric

Give each FIT case equal mass1/24 and each of its m_c labels weight
w_j=1/(24 m_c). Thus all12 domains also have equal mass; long responses cannot
dominate calibration merely by length. Let qbar=sum_j w_j q_j,
H=diag(qbar)-qbar qbar^T, and C=sum_j w_j f_j f_j^T.
Use uncentered C: the original final head has no free
bias. H removes a common logit shift. Its pullback is

    G = M^T H M
      = M^T diag(qbar) M - (M^T qbar)(M^T qbar)^T.

qbar is calibrated ONLY on FIT; it is not uniform and not fitted on DEV.
This is a fixed mixture-distribution quadratic surrogate, not exact KL and
not the average context Fisher. In fact

    H - sum_j w_j F(q_j)
      = sum_j w_j (q_j-qbar)(q_j-qbar)^T >= 0.

That identity does NOT upper-bound the desired context-dependent loss when
each error vector differs with j: error and q are correlated. Full-V KL and
argmax decisions must still be measured. Small quadratic training error alone
does not admit useful quality, and this metric is not justified as a Taylor
approximation at the old KL~6.52/13.57 failures.

## Closed-form rank choice for this surrogate

On the support of C, let C^(1/2) be its PSD square root and C^(+1/2) its
Moore-Penrose inverse square root. Diagonalize

    A = C^(1/2) G C^(1/2),

and let U contain its k leading orthonormal eigenvectors. Define the PAIRED
encoder and decoder

    B = U^T C^(+1/2),
    K = M C^(1/2) U.

For C nonsingular, B C B^T=I. Singular C restricts these identities to its
empirical support. Numerical rank handling/PSD tolerances must be declared
before consuming data; tiny eigenvalues must not be inverted silently.
No ridge or hidden normalization may be added after observing DEV failure.

For any linear L, the empirical surrogate is exactly

    J(L) = sum_j w_j (M f_j-L f_j)^T H (M f_j-L f_j)
         = ||H^(1/2)(M-L)C^(1/2)||_F^2.

The squared singular values of H^(1/2)M C^(1/2) are the eigenvalues of A.
Its rank-k truncated SVD gives L=K B and J(L)=sum_(i>k)lambda_i,
on the empirical support. The discarded-eigenvalue sum therefore certifies
the best rank-k value for THIS fixed quadratic objective, not best KL,
not best nonlinear codec, not DEV optimality and not a universal width limit.
H can be singular; the construction still supplies a minimizing feasible L.
Directions with zero weight are not identified uniquely by this objective.

## RMSNorm compatible realization

Set k=255, d=256, epsilon=1e-5, phi=B f. For fixed a>0, rho>0 and
||phi/a||^2<=d rho^2 define

    u = [phi/a ; sqrt(d rho^2-||phi/a||^2)],
    gamma_C = 1,
    W_C = a sqrt(rho^2+epsilon) [K,0].

Original RMSNorm(u) has denominator sqrt(rho^2+epsilon), hence
W_C RMSNorm(u)=K phi in real arithmetic. This identity does not require B
orthogonal. It extends the previous carrier construction to a whitened,
output-sensitive encoder/decoder pair without adding a final native operator.
Source BF16 rounding, actual target RMS/head precision and carrier production
remain independent residuals. A large/ill-conditioned B can amplify state error;
whitening is not automatically a good causal-training target.

A candidate finite-domain gain rule is rho=1 and
a=2 max_FIT ||phi||/sqrt(d). It puts observed FIT information radius<=sqrt(d)/2
and carrier>=sqrt(3d/4). This is a declared empirical domain, not a global source
bound or a guarantee for future histories. DEV domain violation is a retained
failure; no clipping or DEV gain refit. The native F32 head must also remain
finite and preserve the chosen pair within fixed precision gates.
This gain rule is prospective, not consumed or numerically selected yet.

Original engine uses float state/norm/head and matvec (engine.c lines329,
347,388,439-440,642-645). Its current head is F32; avoid assuming a half-head
format because a different source/runtime experiment used lower precision.
The packed expert weight codes do not change this final-head datatype.

## Why a readable codec still needs a causal accuracy test

Let a compact causal core produce u+xi rather than the exact carrier state u.
Write xi=[xi_info;xi_carrier], r0^2=rho^2+epsilon, and

    beta = (2 u^T xi + ||xi||^2)/(d r0^2).

In real arithmetic, whenever 1+beta>0, its logits are EXACTLY

    l(u+xi) = (K phi + a K xi_info)/sqrt(1+beta),

so the error splits into a denominator-induced rescaling and information error:

    delta = (1/sqrt(1+beta)-1) K phi
            + a K xi_info/sqrt(1+beta).

Quotient out the common-logit direction before interpreting either term as
distribution error. This formula does not turn their separate norms into an
exact KL decomposition; the two effects can cancel. For small xi, the Jacobian
form is a K xi_info - K phi (u^T xi)/(d r0^2). Carrier error can therefore
change token probabilities even with a zero last head column. A learner must
control this denominator or the paired representation is lost again.

The exact encoder/carrier Jacobian inside its domain is

    J_u(f) = [ B/a ; -phi^T B/(a^2 sqrt(d rho^2-||phi/a||^2)) ].

Its carrier derivative diverges at the domain boundary. The prospective2x FIT
radius rule leaves an explicit FIT margin, but says nothing about unseen
histories. Whitened coordinate error and carrier/radial error should be measured
separately in any later recovery. Euclidean latent error alone still cannot
certify that the final categorical information survives.

## Cost and decision still to freeze

FIT statistics require reading4422 paired features/logit rows, streaming qbar,
and two2048-square operators/eigensystems; G needs the full65537x2048 source
head but no new source forward. Avoid forming H as a65537-square matrix.
Use the diagonal-minus-rank-one formula; record numerical method, precision,
actual time/memory and independent witnesses. One metric/rank pair, no rank
or scale grid. Numerical costs have not yet been benchmarked or capped here.

Before fitting: require completed FIT capture/full audit, declare one algorithm,
rank/conditioning/precision thresholds, finite held caps and donor-relative
FIT/DEV full-V gates. Freeze the result before any matching new DEV capture;
reuse the one qualified DEV case when source operators/custody match.
If this pair loses substantial output information, inspect the residual modes
and conditional readout possibilities. Do not infer that adding useful ternary
experts behind the same fixed head can expand its linear output span.

This is a better-defined mathematical control, not a transferred chatbot.
Original SSM/SWA/LUT/ternary history recovery, own-history generation/tasks,
same-artifact50token/s, useful-n/DRAM and second-family validation remain missing.
