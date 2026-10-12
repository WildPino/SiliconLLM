# Exact categorical residual correction: prospective next

**Superseded12 October2026:** actual gradient/finite correction plus separate
convergence and full audits are COMPLETE. [Result](EXACT_CATEGORICAL_RESIDUAL_CORRECTION_RESULT_20261012.md)
keeps original coarse nonacceptance and qualifies a separate FIT step;quality
still FAIL. Next broaden existing supervised labels. Text below is historical.

12 October2026. **PROPOSED/UNIMPLEMENTED/UNBOUND**. Selected after completed
global/conditional quadratic geometry and exhaustive frozen-chart output choice.
No long training, T4 or new expert-count expansion selected.

## Decision and changed variable

For25 of DEV72 states no tested final16-chart frozen source encoder/decoder can
preserve the source argmax. A selector alone cannot repair those frozen candidates.
This does not exclude different coordinates inside the same decoder images,
different directions, learned causal encoding or more expressive conditional banks.
The present fitting objective is quadratic; actual finite probabilities can fail
where initial source-local Fisher is tiny. Change the fitting objective to exact
full categorical KL, and verify the gradient/one bounded correction before dose.

Do not repeat the old full categorical native learner: its fresh SSM/global-P
composition was catastrophically inaccurate. Here the new variable is the actual
source-state read/write factorization, with the trained source head held fixed.
No claim that final-state fitting alone repairs internal SSM or composed behavior.

## First finite experiment to implement/freeze

Use actual RAW source h24 on FIT24, first/middle/last LABEL positions (72 states),
chosen from the bound corpus; equal case/label weights. Actual normalized source
head F64 is q reference, not cached BF16 teacher logits. Reuse current global
source-function256 V/W as initial factors and raw center mu=0. No DEV fit,
source forward or oracle-branch selection for training. DEV72 remains the same
fixed output diagnostic, not a new held-out corpus or complete chatbot evaluation.

Column convention: z=W^T(h-mu), y=mu+Vz, r=sqrt(y^T y/2048+1e-5).
A includes actual source final gamma and lm_head multiplier; p=softmax(Ay/r).
Train objective equal FIT mean KL(q||p) over all65537 vocabulary entries.
No initial Fisher surrogate, top-only loss or unpriced conditional source query.

Exact derivatives, with g=A^T(p-q), are

`g_y = g/r - y (y^T g)/(2048*r^3)`

`grad_V = g_y z^T`

`grad_W = (h-mu) (V^T g_y)^T`

`grad_mu = (I - V W^T)^T g_y`.

Sum with declared FIT weights; include BOTH routes through mu and the full RMS
Jacobian. For an oracle source denominator, its derivative is zero, but that
version is a separate diagnostic, not the training/deployment objective.

Validate against finite central differences in deterministic parameter directions
before accepting a step. Bind perturbation construction/scales/tolerances and
small trust-step ladder before observing results. Use only FIT to select a
candidate; inspect DEV only after that selection is immutable. Persist complete
gradients/candidate maps/source probability traces and input hashes; independent
probability/gradient/directional/parameter update audit required, even failure.
No automatic long epoch merely because a local gradient is correct.

## Algebraic constraints and qualification

A generic V/W correction keeps rank<=256 but need not keep W^T V=I/idempotence.
Do not silently retract/orthogonalize or claim it remains the prior quadratic
projector optimum. If an idempotent constraint is imposed, derive the tangent
gradient/retraction and test it as a distinct variable. G=V^T V and affine norm
remain the correct decoded-state norm. New chart transitions still require full
W_next^T V_previous matrices/center offsets, with actual sublayer placement.

An encoder reading actual full source h is teacher-injected, not causal deployed
state acquisition. Any success must subsequently transfer compact dynamics,
generative own-history behavior and native export/precision/routing/DRAM cost.
Internal normalized SSM input distortions still~40%-50% and are not corrected
by a final-head-only step. Multiple source families/scales and useful n/50 remain
open. All existing failures/qualified primitives remain valid and reusable.

Estimate actual CPU cost from one bound matrix calculation before a multi-step
run. Announce exact resource budget/stops; if hours are needed, launch frozen job
and follow user's pause instruction. No delegation/Luna/T4 selected here.
