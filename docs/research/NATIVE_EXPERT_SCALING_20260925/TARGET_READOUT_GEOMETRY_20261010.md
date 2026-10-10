# Projected targets, normalization and decoder geometry

10 October2026. Algebraic analysis while the stored audit finishes. No new
candidate measurements, fitted maps, source queries or optimizer updates.
Companion to the [executable attribution protocol](ORIGINAL_LATENT_READOUT_ATTRIBUTION_PROTOCOL_20261010.md)
and [primary literature investigation](LITERATURE_CONVERSION_FOLLOWUP_20261010.md).
The attribution criteria remain unchanged. Goal INCOMPLETE.

## The actual projection is not an exact change of coordinates

Use column vectors and real arithmetic to expose the approximation. Source
residual h has dimension d=2048. P has shape2048x256 and approximately orthonormal
columns; write D=256, z=P^T h and Q=PP^T. The ideal identities below assume
P^T P=I. Actual basis defect, BF16 rounding, packed F32 and native AQ are separate
numerical terms, already recorded; they are not silently included in an identity.

Let G be the donor's diagonal final normalization weight, W its effective head
including its fixed output multiplier, and r(h)=sqrt(||h||^2/d+eps_D). Let G_C
and W_C be the actual51 diagonal final weight and head, and
r_C(z)=sqrt(||z||^2/D+1e-5). The retained-target intervention compares

    l_D = W G h / r(h)
    l_C = W_C G_C P^T h / r_C(P^T h).

The initializer in `original_falcon_learner.initialize` sets the head to WP and
the compact normalization diagonal to diag(P^T G P). This does not retain the
off-diagonal entries of P^T G P, the discarded complement of h, or the donor
normalization denominator. Later actual updates can also change both operands.
Transferring parameter values through P therefore supplies an initialization;
it does not establish a function-preserving residual/readout transport.

## An exact decomposition of the idealized readout error

Define DeltaW=W_C-WP, using the actual compact head and gamma in each arm.
For orthonormal P, direct expansion gives

    l_D-l_C =
      W G (I-Q) h / r
      + W (G P-P G_C) z / r
      + W P G_C z (1/r-1/r_C)
      - DeltaW G_C z / r_C.

The four terms have different causes:

1. Discarded donor directions that its readout still observes.
2. Incompatibility between the projected normalization and a diagonal compact
   normalization, including changes of the actual learned compact gamma.
3. Change in the norm denominator after projection.
4. Actual head adaptation away from the projected initializer.

These are algebraic terms, not four independently measured causes. They can
cancel. Their norms cannot be added as an exact KL decomposition. The new
intervention measures their combined output against retained teacher logits;
it does not isolate each term and does not use an optimal compact decoder.

Even if G=I and the head remains WP, term1 and term3 generally survive. For a
particular h with negligible epsilon, equality of the denominators requires
||P^T h||^2/||h||^2=D/d. This condition can vary by history and position. The
recorded projection energy is useful for this geometric question, but is not
a fraction of knowledge or preserved chatbot quality.

For a square orthogonal basis, no direction is discarded and the norm is
preserved. A general diagonal G still becomes a dense conjugate P^T G P;
absorbing this conjugate into neighboring weights can preserve a suitable full
coordinate change. Rectangular compression and deletion require additional
approximation. This is consistent with the distinction in the primary
[SliceGPT paper](https://arxiv.org/html/2401.15024v2), not a claim that our
rectangular P implements that paper's transformation.

## What output loss observes

Softmax ignores a common logit shift. Let Pi=I-11^T/V. All readout-error terms
should therefore be interpreted through Pi when discussing probability
expressivity. A large constant component of l_D-l_C can have zero probability
error; a small task-relevant component can change a decision.

For q=softmax(l_D), delta=l_C-l_D, smooth real arithmetic gives the local
expansion

    KL(q || softmax(l_D+delta))
      = 0.5 delta^T [diag(q)-q q^T] delta + O(||delta||^3).

The Fisher matrix is positive semidefinite and annihilates the constant-shift
direction. This is a local Taylor approximation, not a bound at the observed
KL~6.52 or a replacement for the full-V measurements. Argmax disagreement is
also discontinuous near ties and must remain a separate metric.

The donor RMS readout Jacobian before finite precision is

    J(h) = W G [ I/r - h h^T/(d r^3) ].

Thus the local residual metric is J(h)^T F(q) J(h), not the Euclidean energy
of h alone. It varies with the history and teacher distribution. A task-aware
target would need to account for this geometry rather than credit mean-only
residual repair as recovered varying information. No such target is fitted here.
The earlier balanced384 recurrent study optimized a declared Cartesian-time
pair proxy; it did not optimize this final full-V Fisher geometry. That
distinction identifies a possible new variable, not permission to replay a
closed rank/map sweep or a prediction that another linear map will suffice.

## Two different limits on adding experts

For the fixed actual head, all logits remain in col(W_C), with dimension at
most256 before quotienting constant shifts. More conditional functions can
improve which256-vector is reached; they do not enlarge that fixed linear
span. This rank statement does not prove that useful chatbot distributions
need more than256 coordinates. A union of conditional readout subspaces could
change the span, but its selector, useful functions, precision and complete
CPU/head cost would require a new demonstrated conversion path.

A separate information statement concerns a fixed encoder Z=f(H). If Y is a
token sampled from the teacher distribution given H, the unconstrained optimum
over *all* probability decoders of Z satisfies

    inf_p E_H KL(q(Y|H) || p(Y|Z)) = I(Y;H|Z).

The optimum is p(Y|Z)=E[q(Y|H)|Z]; this follows by adding and subtracting
log q(Y|Z) and averaging. Extra expert parameters behind the same encoder
cannot eliminate genuinely lost conditional information. However, the finite
retained dataset does not identify this conditional mutual information without
additional assumptions. Distinct continuous Z values could simply be memorized.
The actual fixed-head injection tests a restricted decoder; it is not an
estimate of this infimum. A different learned causal encoder may retain a
different sufficient statistic. Neither projection energy nor our failed
intervention can refute compact learned states generally.

## Consequence for the next decision

### A separate operator mismatch in redundant FFN shards

For scalar donor preactivations g and u, the exact real-arithmetic identity is

    g u = sum_(s,t in {-1,+1}) s t ReLU(s g) ReLU(t u).

The current four sign replicas exploit this bilinear decomposition, but a
SwiGLU neuron contributes SiLU(g)u=g u sigmoid(g). Summing all four signed
dReLU replicas reproduces gu, not SiLU(g)u, unless the missing sigmoid factor
is supplied. Projection, normalized sparse selection and AQ/ternary decisions
introduce further independent approximations. Coverage alone cannot repair a
changed activation. This does not refute approximation by learned experts and
input-dependent softmax routing; it identifies what they must approximate.

If a region-specific redundant function uses a coefficient c_j approximating
sigmoid(g_j), its full-coverage activation error is exactly

    sum_j down_j * g_j u_j * (sigmoid(g_j)-c_j).

The sigmoid derivative is at most1/4. If c_j=sigmoid(g_j^0) and the routed region
guarantees |g_j-g_j^0|<=delta_j, the output norm is bounded by

    0.25 sum_j ||down_j|| |g_j u_j| delta_j.

That conditional bound does not certify any current router region, sparsity or
quantizer. It states a measurable route for the user's redundant-leaves idea:
store function variants valid on declared regions, then test coverage,
coefficient error and held-out applicability. Active work and region-selection
cost still count. No such variants have been constructed or admitted here.

The [granular upcycling paper](https://arxiv.org/html/2410.07524v2) separates
shard coverage from router attenuation. Its weight-scaling derivation uses
squared ReLU; its SwiGLU scaling evidence is empirical. Its formula therefore
does not establish an exact SiLU-to-dReLU conversion for our initializer.

### Ordered diagnosis

First close the actual51 stored adjudication and run the already fixed two-head
intervention. If the target is readable substantially better, study how the
original internal path reaches a compatible representation. If it is not,
distinguish the projected target's coordinates, normalization and decoder
coupling before adding width or a new conditional head. Preserve mixed and
domain-specific outcomes. No unchanged long dose, DEV refit, T4 campaign or new
runtime operator is selected by this memo. Useful own-history chatbot quality
and same-artifact50 accepted tokens/s remain the final tests.
