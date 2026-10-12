# Exact categorical residual correction: frozen finite experiment

12 October2026. Actual trained-source head and raw source states, no source model
or native learner. New variable: exact finite categorical objective/read-write
factor gradient, following failed global/conditional quadratic projections.

## Data, reference and parameterization

FIT24 cases, first/middle/last LABEL indices from bound corpus (72 states), each
weight1/72 (three labels per equally weighted case). Source raw h24 BF16 decoded
to F64. Source reference q computed directly from actual source final RMS/gamma,
trained source head and lm_head multiplier in F64, all65537 vocabulary entries.
No cached BF16 logits used as FIT reference. DEV uses the SAME previously audited
DEV72 raw positions/F64 source head reference. No DEV-dependent parameter fitting.

Initial V/W from the audited global source-function basis24, mu=raw zero vector.
For row inputs X=H-mu,Z=X W,Y=mu+Z V^T,R=sqrt(mean(Y^2)+1e-5), logits=(Y/R) A^T.
A folds actual source final gamma and head multiplier. Objective exact mean
KL(q||softmax(logits)). Full vocab, no initial-Fisher or top-token replacement.

With weighted logit gradient(p-q)/72, normalized-feature gradient g=((p-q)/72)A:
`gy=g/R - Y*(sum(Y*g))/(2048*R^3)`;
`gradV=gy^T Z`; `gradW=X^T(gy V)`;
`gradmu=sum(gy - (gy V)W^T)`.
Both paths through mu and entire RMS Jacobian included. Full arrays persisted.

## Derivative controls, trust direction and selection

Group scales norm(initial V),norm(initial W),sqrt(mean_FIT sum(h^2)) for mu.
For each group V/W/mu, two deterministic directions:sin(flat_index*.017+phase)
at phase0,.7, reshaped and scaled to that group's norm scale; other groups0.
One joint normalized-gradient direction,7 directions total. Central finite
differences at eps1e-5 AND2e-6,28 point evaluations, all full scores/decoded
states/norms/coordinates/per-row loss persisted. Compare to full analytic gradient
inner product:abs(finite-analytic)/max(1,abs(analytic))<=2e-6 at BOTH scales.
No post-observation change to eps, tolerances or accepted derivative subset.

One trust direction per group is -gradient*scale/norm_gradient; absolute group
norm<=1e-12 gives zero direction. Report all norms/scales. This is a declared
normalized-group probe, not an invariant/optimal natural gradient or per-expert
noise-amplifying update. Directional tests include the actual joint direction.
Fixed steps alpha=.0001,.001,.01,.1; calculate and persist ALL four even a
derivative/scientific failure. Choose smallest exact FIT mean KL (fixed order
breaks ties),accept only all14 derivative controls PASS and relative FIT
improvement>=1e-4. Otherwise select unchanged initial factors. Persist a selection
seal with exact parameter extents BEFORE semantic DEV evaluation. DEV does not
select steps or thresholds. No second gradient, epoch or chart-count expansion.

## Qualification and cost

Generic V/W update preserves factorized rank<=256; does not necessarily preserve
W^T V=I/idempotence or earlier covariance optimum. No silent retraction. Affine
norm G/cross/constant remains exact. Full-source encoder is teacher-injected;
internal SSM/core dynamics, conversation, engine/ternary/export/DRAM/speed and
additional families/scales are still missing. A better final-state predictor is
not a useful composed chatbot. Actual finite quality proxy:meanKL<=.01,worst<=.1,
disagreement<=.01 on selected DEV72, report even failure. FIT descent is a separate
component decision and cannot confer full-goal admission.

CPU6/affinity0..5,900s/OS8GiB/output2GiB,no children. Record duration of first
full source FIT72 head multiply, exact inputs/source/protocol freeze/binding/argv/
PID/creation/terminal and all artifacts. Relevant hashes before AND after.
Source head read directly from actual weights; no model loader/backward framework,
GPU/native/T4/RESERVED/longalign call. Foreign publishers retained; no clean timing
admission. Independently audit every score,probability,gradient,difference point,
candidate bit update,FIT selection and selected DEV even a scientific failure.
First execution fault preserved, any repair separate source/version/namespace.
