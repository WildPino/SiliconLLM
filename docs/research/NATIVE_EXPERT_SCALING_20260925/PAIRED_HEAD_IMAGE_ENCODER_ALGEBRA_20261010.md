# From head-image codes to an executable encoder: algebra and next gates

10 October2026. Prospective deductions while the frozen head-image probe runs.
No additional model, gradient, native or optimizer call. These are mathematical
relations and proposed next gates, not a measured chatbot conversion.

## What a categorical encoder must preserve

Fix the actual decoded A=aK and a teacher distribution q. Let
L(x)=log(sum_v exp((Ax)_v)), m=A^T q, and c=sum_v q_v log q_v.
Then KL(q||p_x)=L(x)-m^T x+c. Thus the teacher influences the code objective
through a255-vector m, with entropy c providing the additive loss constant.
Matching all raw source logits in a quadratic metric is a stronger/different
approximation objective. Retaining97.95% of a fixed-mixture quadratic energy
cannot by itself certify categorical fidelity, as the paired FIT result shows.
The source map f -> A^T softmax(sWf) is nonlinear. The current linear Bf was
selected by a quadratic surrogate, not by this sufficient-statistic condition.
Moment matching is necessary at an interior optimum: A^T p_x=A^T q. It is
not proof that a compact causal recurrence can compute the required code.

## A loss decomposition that does not require an exact optimizer

For ANY feasible reference y and candidate x, in exact real arithmetic:

KL(q||p_x) = KL(q||p_y) + KL(p_y||p_x)
             + [A^T(p_y-q)]^T (x-y).

Expand each KL as L minus a linear moment term to obtain the identity; no
assumption of global convergence is needed. At an interior optimum the last
term vanishes. On a ball boundary an exact constrained optimum has g_y=-lambda y,
lambda>=0, so g_y^T(x-y)=lambda(R^2-y^T x)>=0 for feasible x. For a stored
approximate point, the last term can have either sign. Its magnitude is bounded
by ||g_y|| ||x-y||<=2R||g_y||. Never remove that term merely because a feasible
upper is small or the iteration cap was reached.

This separates a fixed-head residual, error in predicting a feasible code's
probability distribution, and the measured residual stationarity term. It is
applicable to later own-history evaluation with the SAME fixed head. A reference
code obtained with donor access is an oracle diagnostic, not an executable state.
The numerical tangent-ball lower used by the probe requires independently
checked losses/gradients and remains a F64 bound, not a rounding interval proof.

## Decision after the complete audit

- If encoder room is verified without an obstructing whole-case/domain lower,
  prefer correcting the encoder/history path. Keep the newly qualified fixed
  native head and train the compact original-operator model directly against
  categorical teacher moments/distributions. The existing actual51 recurrent
  artifact may initialize a NEW paired-head branch; it cannot be declared
  qualified by attaching the head, and the old frozen training run is not resumed.
- If a sufficient whole-case/domain lower is verified, a different encoder alone
  cannot cross that gate for this fixed head/domain. Change the head geometry,
  domain or output representation; do not extend unchanged latent descent.
- If both hold, preserve both: improve the encoder while explicitly correcting
  the remaining head floor. If unresolved, use the final residual gradients and
  particular failed labels to identify the obstruction, rather than replaying
  the same512-step arm indiscriminately.

Before any new training: choose the initialization/state-to-readout convention,
parameter allocation (including effective rank255/carrier radius), safe causal
carrier realization, exact operator derivatives and FIT supervision; freeze a
new namespace/optimizer, matched numerical CPU/native gate, DEV criteria/costs.
First measure the current recurrent artifact with the changed head on the already
available qualified FIT states. A local mapping on full donor f would diagnose
nonlinearity but would leave the expensive source history in the runtime; only
perform it if it changes the compact causal transfer decision.

The central unresolved transfer is a cheap causal state that predicts useful
codes from its OWN history, plus useful ternary conditional functions and CPU
routing. All72 optimized coordinates remain training-side latent variables.
Quality/generation/tasks on held-out data, useful n, physical DRAM, same-artifact
50 tokens/s and another donor family still have to be established. No universal
D256 ceiling, whole-cohort upper or chatbot admission follows from72 FIT codes.

## Use the original RMS readout during causal recovery

The canonical diagnostic state has ||u||=16 and its last coordinate carries the
remaining norm. The original engine need not compute an explicit square root
carrier at inference if a trained causal state can occupy this representation.
For any actual256-vector y, gamma fixed to1 and the frozen head yield effective
x = sqrt(1+epsilon) y[:255] / sqrt(||y||^2/256+epsilon).
Its open radial range approaches16 sqrt(1+epsilon), slightly wider than the
canonical closed radius16 used by this probe. A floor for radius16 would remain
a conclusion about THAT fixed head/domain, not all possible native states.
There is no observed floor here yet. No head/domain conclusion should silently
relax or reinterpret the bound domain.

With r=sqrt(||y||^2/256+epsilon), c=sqrt(1+epsilon), and g=dKL/dx,
dKL/dy = (c/r)[g,0] - c y (g^T y[:255])/(256 r^3).
This is an analytic consequence of original RMS, not a new operator. Joint
causal training must use this derivative and the actual F32 readout gates;
train on the native state rather than assuming an explicit canonical carrier
will be available. The frozen old actual51 norm/head are different artifacts;
a new paired branch changes norm/head explicitly and keeps old evidence intact.
