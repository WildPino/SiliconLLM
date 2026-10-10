# New broad-history cross-runtime defect: attribution after the full audit

10 October2026. The full24-case campaign is LIVE; its code/criteria remain
immutable. The following are stored worker observations, not independently
adjudicated campaign conclusions. No additional numerical job runs concurrently.

## The new measured distinction

Case2 apigen023 uses479 history IDs/118 labels at the first newly accepted
state. Dense/compiled loss discrepancy4.83169e-13, state-gradient2.70006e-13,
F32 upstream0, GPU/CPU fixed-head score8.52651e-13. GPU forward/checkpoint route
ID/mass bits identical; blocks[2]*6, all four gradient roles nonzero. This
preserves the local categorical algebra and checkpoint trajectory observed by
the worker; it does not qualify the GPU trajectory as the C trajectory.

GPU/C discrepancies:4 ranked ID slots out of22,992, mass maximum.000844463706,
selected-label score maximum.394166961 and KL maximum.055132711. Argmax0
differences, both mass normalization defects<=1.78814e-7. The saved fields do
not yet distinguish an order-only swap from changed expert membership or locate
the first divergence. A small slot count is not a bound on functional error.

The alpha.0001 actual C candidate improves weighted82.298031->76.281505 and
first43.516856->35.763158. Alpha.001 gives weighted73.276947 but first76.993394.
Constrained selection identifies.0001; acceptance is FALSE because baseline
cross-runtime numeric flags fail. Both trials have all radius/direction/
quantization/route-validity flags true and all118 argmax disagreements with
the donor. The runner conservatively retains the preceding accepted state.
Full candidate/state-chain audit is still required even on this failed bridge.

Case3 apigen002 supplies a distinct boundary:773 history IDs/33 labels, ALL
ranked IDs equal and selected-label score.000321198/KL.000159854 pass, but
full-history mass maximum.000555463 exceeds1e-5. Its feasible alpha.001 actual
C weighted83.606062->58.933532 is likewise rejected by the frozen mass gate.
Thus ranked-ID disagreement is not necessary for the observed bridge failure.
The score comparison covers the33 supervised positions, whereas route/mass
checks cover every773 position/six layers. It does not prove that all earlier
prefix logits match or that an earlier mass discrepancy is harmless.

## Why this is not an attribution to the loss compiler

Original GPU causal operators are numerically separate from original AVX C:
GEMM/reduction order versus `dotf`/`matvec`, recurrent/transcendental arithmetic,
AQ63 threshold decisions, and top-eight selection. CPU ternary weight decisions
and original packing/integer witnesses are already retained, but those witnesses
do not establish that every real-history activation operand is identical.
The native router uses `expf` and sequential F32 normalization; the GPU router
uses Torch softmax and reduction, then normalizes selected mass. Inspect actual
pinned source provenance before testing a repair; original engine.c remains
unchanged. A difference after normalization need not originate in normalization.

In real arithmetic the all-expert softmax denominator cancels from normalized
selected mass. In finite precision and at differing states this identity does
not force equal winners or mass. At the SAME input, if router score discrepancy
is bounded by epsilon in infinity norm, an eighth/ninth score gap>2*epsilon
suffices to preserve membership. All adjacent selected gaps>2*epsilon suffice
to preserve ranking with consistent ties. A causal-input discrepancy adds the
router's sensitivity to that input discrepancy; no same-input bound is currently
measured here. Saved selected probabilities omit the ninth score, so a membership
margin cannot be certified from these files alone.

AQ63 has a similar issue: a small continuous perturbation can cross a half-integer
decision face. The ensuing integer change can propagate through gated expert
outputs and later recurrent states. To assign the observed final-logit difference,
find its earliest operand/decision divergence; final mismatch counts alone do
not identify a cause or prove that all future directions are unusable.

## Next safe evidence and possible corrections

Later provisional directions7/8 add two SAME-recorded-state histories:
both reject and retain identical selected pack/92-parameter metadata. Full209-ID
case7 has IDs exact/mass2.65837e-5/score.334613/KL.128457/one GPU-C argmax change;
full337-ID case8 also has IDs exact but mass.002151817/score2.374393/KL1.285110
discrepancy and zero argmax change. The independent state-chain audit is pending.
These observations do not show a monotonic length dependence or a particular
router/activation culprit; they show why ID equality and argmax equality alone
cannot qualify categorical function parity. All local dense/compiled/checkpoint
gates report PASS. Preserve both full histories and locate missing operands.

First finish the CURRENT24-case capture and exhaustive stored audit. Preserve
all numeric failures, rejected candidates and final quality; do not restart,
shorten the audit or change thresholds. Independent adjudication establishes
which observed bridge failures are real and their breadth across histories.

The [stored-route diagnostic](CATEGORICAL_ROUTE_DIVERGENCE_OBSERVER_20261010.md)
is now IMPLEMENTED/AST and synthetic format fixtures PASS, actual preparation
and execution PENDING full campaign/audit closure. It can identify first
position/layer, order versus membership, and mass differences with matching IDs,
using existing sealed arrays. It needs no model/backward/native replay. Whole
loss/head checks need not be repeated merely to rediscover their current result.

If missing operands prevent attribution, define ONE new observational C/GPU
comparison at the first unresolved divergent point: common input, actual router
scores/margins, activation quantization coordinates/scales and block boundaries.
No broad training rerun is justified before this missing-evidence purpose and
cost are fixed. Original weights/input/execution identities must be pinned.

Two distinct possible remedies require new protocols, not post-hoc admission:

1. A more faithful training forward: reproduce critical C reductions/decisions,
   or use actual native operands/route decisions with an explicitly defined
   surrogate adjoint. Measure trace/checkpoint memory and conversion cost.
2. An explicitly approximate learner whose finite steps are assessed against
   the actual C objective, with declared discrepancy and native descent checks.
   This would give up a claim of exact cross-runtime forward parity; it would
   still require independently valid proposals, original export, full native
   held-out/own-history quality and same-artifact useful speed. The goal permits
   approximation, but this consumed campaign's failed numeric gate remains failed.

Neither remedy is implemented or selected. A relaxed threshold without a
functional/error argument does not resolve the defect. Larger useful-n banks
introduce more selection competitors; no margin or normalization guarantee
follows from the current1152-expert test. The useful chatbot pipeline, CPU
structured selection/mass, physical DRAM, useful50 and family/scale evidence
remain required.

## What a functional approximation guarantee would actually require

Let F_C(W,x) be the declared categorical loss executed in the original C,
and F_G(W,x) its training-forward approximation. Bounds at BOTH endpoints,
|F_C(W,x)-F_G(W,x)|<=epsilon0 and
|F_C(W+delta,x)-F_G(W+delta,x)|<=epsilon1, imply

    F_C(W+delta,x)-F_C(W,x)
      <= F_G(W+delta,x)-F_G(W,x)+epsilon0+epsilon1.

A surrogate finite decrease exceeding epsilon0+epsilon1 would therefore
certify a native finite decrease for THAT case. A baseline-only discrepancy
bound cannot do so. No candidate GPU forwards were captured in this campaign;
the two-endpoint surrogate bound is not established here. Conversely, an actual
before/after C comparison directly measures that finite difference without
requiring such an approximation bound. Neither observation proves descent of
the24-case mean, a true continuous gradient through discrete operations,
held-out quality or robustness at arbitrary n. This algebra describes a
possible future approximate-learning contract; it does not override the
current exact-routing/mass acceptance gate or admit its rejected candidates.
