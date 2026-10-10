# Information in a quantized direction: focused follow-up

10 October2026. Primary papers consulted during the held full24-case campaign.
No concurrent numerical experiment, no change to consumed campaign criteria.
This is an algebraic diagnostic note, not a result of the evolving trajectory.

## What two primary papers establish

ProxQuant compares gradients evaluated at quantized coordinates with gradients
evaluated at continuous coordinates followed by a proximal operation. Its toy
counterexample has two losses whose derivatives coincide at the discrete
points but whose discrete minimizers differ. Thus those derivatives alone need
not identify the better discrete choice. Its stationary-point theorem requires
a smooth loss and differentiable regularizer; the paper explicitly discusses
smoothing its nonsmooth quantization penalty. ResNet/LSTM experiments and this
theorem do not establish convergence for our deterministic ternary, activation
quantization and top-eight routing composition. See Figure1, sections2 and5.1,
Remark5.1 in [Bai, Wang, Liberty, ProxQuant](https://arxiv.org/pdf/1810.00861).

Shekhovtsov and Yanush derive straight-through methods in stochastic binary
models. Section3 uses Bernoulli weight probabilities and relates a KL mirror
step to additive latent-logit updates. The modeled objective is an expectation
over binary weights; gradient estimation and handling probability constraints
are distinct problems. This supplies hypotheses under which a bypassed
derivative is principled. It does not make every deterministic identity STE an
exact gradient, or immediately apply to learned ternary scales/top-k expert
switching. See section3/equations12–14 and Proposition1 in
[Reintroducing Straight-Through Estimators as Principled Methods for Stochastic
Binary Networks](https://arxiv.org/pdf/2006.06880).

## Our operator algebra: a stable ternary cell is low rank

The actual `original_tensor_learner.quant_weight` computes, for a row of d
latent weights, s=max(mean(abs(w)),1e-5), q=clip(round(w/s),-1,1), and forward
effective weights a=s*q. Scale and q are detached in the surrogate; the effective
weight expression gives identity backward to w. The streamed bank uses the
same quantizer. These are code observations, independent of the papers above.

In ideal real arithmetic, restrict to an open neighborhood with fixed q,
fixed signs of nonzero w, and inactive scale floor. Then

    a(w)=q * (sum_j |w_j|)/d,
    J_a(w)=q * sign(w)^T/d,
    da=q * <sign(w),dw>/d.

The row map has rank at most1. Identity STE substitutes a d-dimensional
identity for this local operator Jacobian. A row direction satisfying
<sign(w),dw>=0 leaves its effective weights unchanged within that cell;
the latent coordinates still change their distances to quantization faces.
With an active constant floor and fixed q, this local map has rank0. Boundaries,
zero coordinates, F32 rounding and changing q require separate treatment.

This operator result does not identify the true complete loss gradient from our
stored surrogate: downstream activation quantization and discrete selection
also use approximations. Nor does it prove that removing all cell-null
directions helps learning. Such a direction can cross a face at a finite step
and create a useful discrete change. Cell-local neutrality differs from the
global homogeneous expert gauge in the existing
[scale geometry note](ORIGINAL_CATEGORICAL_SCALE_GEOMETRY_20261010.md).

## Consequence for reading the campaign

The already audited alpha.001 single-case result changed26,613 ternary symbols
and447,399 scale entries, altered1,006 ranked routing slots, and reduced actual
C weighted KL from138.960856 to58.820600; absolute18/18 answers remained wrong.
Those observations demonstrate an actual finite decrease at that point, not
the correctness of every surrogate Jacobian or a useful converted chatbot.
[Full stored result](ORIGINAL_CATEGORICAL_TRUST_STEP_RESULT_20261010.md).

After the full campaign audit, distinguish three questions using retained
directions/archives/actual C losses: whether proposals decrease their own case,
whether sequential cases interfere on final FIT, and whether improvement
generalizes to DEV/own-history replies. Whole-FIT or task failure cannot be
assigned to quantization solely from the count of changed symbols. Numeric
route mismatch requires attribution before interpreting a rejected direction.

If a concrete directional defect remains, useful next diagnostics are expert
gauge contractions and predicted-versus-actual finite decrease, with scale,
symbol and routing changes separated. An operator-aware or stochastic/proximal
alternative would require its own frozen geometry, runtime and quality tests;
neither paper justifies changing this consumed campaign or repeating the old
optimizer trajectory. No added cost/quality/capacity claim is made here.
