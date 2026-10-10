# Next diagnosis: paired head image versus its linear encoder

10 October 2026. Analytical next step; NOT launched. Goal INCOMPLETE.
[One paired codec](PAIRED_OUTPUT_CODEC_RESULT_20261010.md) and complete independent
audit are finished. Numeric gates all PASS, FIT quality FAIL:caseKL.804121,
disagreement22.1684%, real/native identical argmaxes. No causal student tested.

## What the failure identifies and what it does not

The covariance/Fisher rank construction correctly attains its declared
support-constrained quadratic optimum: J=.35160991716/total17.15911750908,
about97.95% retained metric energy. Actual KL/IDs still fail. Original F32
norm/head precision agrees closely with real arithmetic. The new uncertainty
is whether K's categorical image/domain loses the relevant information, or
whether the linear feature map Bf chooses poor coordinates inside a useful image.
Do not infer a universal D256 ceiling, or train more experts to the same failed
coordinates before answering this narrower question.

The old Adam1 oracle uses a DIFFERENT head/gamma and raw-feature geometry;
its512 iterations/two scales are complete and will not be replayed. A new
probe is justified only by this newly fitted K, fixed carrier gain/domain and
changed initial coordinates. No gain/rank grid, new teacher observation or T4.

## Exact KL geometry

Let x=phi/a (255 information coordinates), A=aK, p_x=softmax(Ax), R=16.
This ball is the declared carrier representation domain, not a theorem that
the causal core reaches every x. For a fixed actual teacher q,

    F_q(x)=KL(q||p_x),
    g(x)=A^T(p_x-q),
    Hessian=A^T[diag(p_x)-p_x p_x^T]A >= 0.

Thus minimizing over ||x||<=R is convex. No exact KL optimization follows
from minimizing the previous common mixture Fisher objective. At any feasible
x, convexity supplies a numerical lower certificate

    lower=max(0, F_q(x)-g(x)^T x-R||g(x)||), upper=F_q(x).

These are real-arithmetic identities evaluated numerically, not interval
certificates. Keep the best valid lower/feasible upper and their unresolved gap;
a stalled finite solver does not establish an image/rank floor.

For an unconstrained finite optimum x*, moment matching gives
A^T p_x*=A^T q. The exact exponential-family Pythagorean identity is

    KL(q||p_x)=KL(q||p_x*)+KL(p_x*||p_x).

For the ball-constrained optimum, g(x*)+lambda x*=0 with lambda>=0 and
lambda(||x*||^2-R^2)=0. The exact decomposition becomes

    KL(q||p_x)=KL(q||p_x*)+KL(p_x*||p_x)+g(x*)^T(x-x*).

If the ball constraint is active, the last term is
lambda(R^2-x*^T x)>=0. This separates categorical image/domain loss, imperfect
coordinate choice and the boundary constraint at a certified optimum. A finite
uncertified iterate cannot be substituted for x* to claim this decomposition.
F32/native rounding and causal history reachability remain independent tests.

## Smallest control that can change the converter decision

Use one fixed A=aK/gain4.328941201436105 from the sealed paired artifact,
not an old head and not a newly fitted head. Exactly first/middle/last label
in every FIT case:72 labels selected by metadata,12 domains. Warm start at
the stored x=Bf/a, preserving the actual baseline loss. One finite constrained
KL optimization, no scale arm/restart/fallback dose. A returned feasible x can
be represented by the carrier with the same head; any claimed precision
admission needs an actual original F32 norm/head check of the returned states.

Before calls: implement/freeze solver, exact metadata selection, per-label
certificates, artifact hashes, cost/stop criteria and independent stored audit.
Reuse prior projected-descent/backtracking instrumentation if suitable; its
previous concrete cost190.640s worker/196.078s held for TWO72-label512-step arms,
2605/2619 objective-gradient evaluations. That supports a forecast1-3min for
one new warm-start arm, not a measured guarantee. Candidate cap300s plus
separate300s audit; caps/iterations/precision must be frozen with actual code.

Do not average the72 sampled lower bounds and call them a whole-FIT floor.
For case c, L_c=sum_(selected j in c)lower_j/m_c is a valid whole-case lower
bound, since omitted KL terms are nonnegative. Mean_c L_c similarly bounds
the actual24-case/4422-label objective. Domain aggregates use the same actual
label counts. A positive full-cohort bound can therefore be informative without
pretending that three labels are statistically representative of a conversation.

Possible decisions to specify before observation:

- Feasible target coordinates substantially improve the SAME72 baseline:
  evidence of room for a nonlinear encoder/causal target; not whole-FIT/DEV
  quality or learnability. Preserve the exact attained upper, including failures.
- Certified whole-case/domain/cohort lower crosses the corresponding existing
  quality gate: this fixed head/domain cannot meet that gate even with arbitrary
  reachable coordinates. Consider a changed categorical image/conditional
  readout, not another unchanged history dose. No global width theorem follows.
- Both or neither: retain mixed/unresolved branch; do not manufacture a unique
  bottleneck from a finite optimizer or weak lower bound.

## Full goal and held-out custody

The fitted pair is frozen. Matching DEV source final states remain missing
except the already qualified53-label case; their capture must be priced/frozen
and reuse that completed case if source operators/custody match. DEV may evaluate
a fixed method, never select/refit its head/gain or redefine FIT thresholds.
This geometric probe is not a substitute for eventual whole held-out evaluation.

Original SSM/SWA/LUT/ternary causal recovery and useful selectively consulted
functions remain required, followed by export/numerical bridge, own-history
chatbot generation/tasks, same-artifact50token/s, physical DRAM, structured
CPU ID/mass routing and another family/scale. No generic donor runtime port or
injected output test constitutes this end state.
