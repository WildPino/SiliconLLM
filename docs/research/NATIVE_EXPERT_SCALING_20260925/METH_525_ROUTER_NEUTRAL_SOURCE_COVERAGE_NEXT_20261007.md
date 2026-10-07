# Selected next525: find missing source regions without dev residual labels

7 October2026. PROPOSED; no525 code/protocol/geometry/queries. Goal ACTIVE/INCOMPLETE.
[524 result](METH_524_SOURCE_HYPERPLANE_TREE_RESULT_20261007.md) shows source sign
information for117 parents, but dev-zero residual cannot distinguish covered37
from undersampled39/63/73. Keep its failed criterion closed. Do not compile its
445 leaves, lower the threshold or tune a fallback count from consumed errors.

## Algebraic variable: directions invisible to a linear parent router

First recover actual source router score operator and the FFN input/norm/codec
contracts. If scores are linear R*x+b at the same continuous input coordinate,
delta in ker(R) leaves ALL scores, winning parent AND normalized mass unchanged
in exact arithmetic. This is stronger than simply borrowing another parent's
input, as520/521 did. It does not imply F32/I16 BYTE equality after quantization.

Let G be an invertible source norm-scale diagonal if this is the actual input
parameterization. Work with y=G^-1*x, C=R*G. Otherwise recover the correct metric
or explicitly decline this variant. Define P as the orthogonal projection onto
row(C), a=P*y0, v0=(I-P)*y0, radius=||v0|| for each dev-only source anchor.
All y=a+v with C*v=0 and ||v||=radius preserve both original router scores and
the original norm shell in real arithmetic. This is a router affine slice of a
sphere; no empirical source-response fitting is needed to describe it.

For an omitted source WI row i_j, let h_j=(i_j*G)^T be its column vector,
k_j=(I-P)*h_j and c_j=dot(h_j,a).
For radius>0 and dim(ker(C))>=2, its reachable preactivation interval on that
slice is EXACTLY:

    [c_j-radius*||k_j||, c_j+radius*||k_j||].

If zero lies strictly inside, both source ReLU signs are possible with identical
router scores and input norm. A fixed omitted reference sign is then not a
global certificate on that slice, even when all dev residuals are zero.
If k_j=0 or the interval has one sign, this slice cannot expose that boundary.
This is conditional reachability, not proof that actual decoder trajectories
visit every point on the sphere, nor a global quality bound.

## A bounded constructive query, if contracts and bounds permit it

For k_j!=0 choose n=k_j/||k_j|| and desired opposite-side preactivation t.
Let u=(t-c_j)/||k_j|| and w=v0-dot(n,v0)*n. At |u|<=radius the nearest same-score/
same-norm point on that preactivation plane has

    v'=u*n+sqrt(radius^2-u^2)*w/||w||,
    distance^2=2*radius^2-2*(dot(n,v0)*u+||w||*sqrt(radius^2-u^2)).

For w=0 choose a deterministic orthogonal null direction from ordered coordinate
axes, not a random seed. A boundary point t=0 is insufficient; a prospective
fixed interior target, e.g. half the reachable opposite-sign endpoint, provides
a sign margin to price before physical quantization. No sweep of target depths.

Proposed ONE geometric candidate per exposed parent, original523 dev-only anchor,
only omitted neurons outside its512 hinges. Compute all eligible candidate
intervals, select nearest resolved opposite-side target, source-neuron ID tie.
Store ordered raw source identities/directions/certificates BEFORE source labels.
All127 parents included;39/63/73 motivate the variable but are not a cherry-picked
candidate cohort. No consumed source function values select points or predicates.

Recover and propagate conditioning/projection/square-root/interval/target errors
BEFORE selecting. Independently audit the row-space/null-space/norm equations and
all candidate order certificates. A rank/D/conditioning failure is an explicit
applicability failure, not grounds to loosen bounds. Real input quantization,
router reduction and softmax may move scores: independently compute/check actual
parent winner and normalized mass under the exact recovered codec. Do not reuse
old p or assert ideal invariance survived in physical arithmetic.

## Before the first geometry run

Inspect actual source router/norm extents and source-engine operators; re-use
admitted523 WI/hinges/signs/anchor/input data. Determine whether scores and WI
share a coordinate or require a quantified codec bridge. Price one deterministic
rank/null-space construction,127*2560 boundary tests, at most127 selected queries,
all-output bounds and independent audit on CPU10/BLAS1. Proposed geometry-only
ceilings main180s/512MiB, audit240s/512MiB, outputs32MiB; no T4/model/corpus/C.
Budgets and complete wire/tie/rank rules must freeze before values. No geometry
job is authorized by merely writing this proposal without its frozen protocol.

First-stage gates are actual conditional-boundary reachability, resolved numerical
construction/order and physically preserved original parent (actual mass recorded
and charged), not1% function quality. If eligible, separately freeze one new
source-response acquisition at those points, original physical source semantics,
and a source-weight-preserving region construction. No old53943 response replay,
augmented fixed A/L0 readout fit, fresh whole promotion or automatic count trial.
One query per parent is a bounded information probe, not sufficient coverage.

The eventual comparison still needs same-shaped one/multileaf/rotated functions,
real SAME fallback determined by prospective source-domain applicability, ALL
six/rare1% gates, original normalized parent mass cost, and actual useful function
count. A reliable fallback or further coverage may be necessary; observed zero
dev residual never supplies that guarantee. Then kernel/DRAM and whole fresh
donor-relative quality AND SAME>=50/s, much larger useful n/RAM and actual other
families/scales follow. No source-output oracle or omitted scarce denominator.

## Applicability limit

This continuous norm-slice construction requires dim(ker(C))>=2, a nonzero null
component and a feasible sphere intersection. A one-dimensional null space has
only two sphere points and does not fill the interval; decline this variant.
Source WI must have a useful component outside the router row space. These are
weight/geometry prerequisites to measure, not assumed for every family or100B.
Larger original parent sets may saturate router rank; cheap full winner/mass and
source data/calibration costs still grow and remain unresolved. Smooth SwiGLU
does not inherit exact ReLU region folding, even if its router has a null space.
