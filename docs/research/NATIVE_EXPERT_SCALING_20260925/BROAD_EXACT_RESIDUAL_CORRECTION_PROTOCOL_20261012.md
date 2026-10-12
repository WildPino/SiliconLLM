# All-label exact residual correction: frozen protocol

12 October2026. New variable:supervised coverage (all4422 FIT/all4386 DEV) and
reproducible score factors. Initial ORIGINAL global source-function256 V/W/raw
mu0, not previous72-label corrected factors. No epoch or chart-count expansion.

## Data, arithmetic and custody

All existing48 cases,actual raw source h24 and corpus LABEL positions. Equal
case weight1/24,label weight1/T_case. FIT and DEV remain the established split;
these are repeatedly used diagnostic data,not fresh RESERVED or final admission.
Actual source final gamma/head/multiplier F64 reference on normalized raw h,
not cached BF16 teacher logits. No source model generation or framework backward.

X=H-mu,Z=X W,Y=mu+Z V^T,R=sqrt(mean(Y^2)+1e-5),A=scaled source head*gamma.
Exact-real factored scores `(Z (A V)^T + A mu)/R`. Persist parameters,V/W/mu,
projected head/bias,Z/R,per-label KL/ID disagreement/source+predicted entropy,
original selected source states/weights and case bounds. Every full-vocabulary
score is reproducible without storing a dense label*vocabulary array per point.
Case/label chunks64;no vocabulary sampling,truncation or quantization.

Baseline ALL FIT labels computed both factored and direct2048-dimensional head:
max score sup-relative<=1e-10,per-label KL delta<=1e-9,all argmaxes identical.
Record each batch. Factor prerequisite is part of acceptance,not a reason to
skip other computations/audits on failure. All other points independently audited
via DIRECT source-width scores/probabilities,including original source reference.
Different F64 operation orders are approximate bit-level,exact in real arithmetic.

## Broad gradient and finite controls

Exact full-V residual(p-q)*case/label weight and g=residual A;full RMS Jacobian
gy=g/R-Y sum(Y*g)/(2048 R^3). gradV=gy^T Z,gradW=X^T(gy V),
gradmu=sum(gy-(gy V)W^T). Store normalized/raw gradient fields and complete three
parameter gradients;full logit residuals reproducible from saved inputs/factors.
No dense master/gradient over a100B expert bank implied by these small factors.

Whole-group trust direction -grad*scale/norm;floor norm<=1e-12 gives0. Scales
initial V/W norms and source sqrt(sum weighted ||H||^2) for mu. These are fixed
normalized-group probes,not invariant/natural-gradient optimization. Use the
actual joint direction for central finite differences at2e-6 AND2e-7,unchanged
abs(fd-analytic)/max(1,abs(analytic))<=2e-6. Source per-group Jacobian identity
already independently verified;new full weighted gradient must be audited again.
All4 finite point factor packets retained even a numerical/scientific failure.

Fixed trust candidates alpha.0001/.001/.01/.1. Persist all parameters/factors/
metrics. Accept FIT-minimum case-mean KL (fixed order ties) only factor controls
AND both derivative scales pass AND relative FIT decrease>=1e-4. Otherwise choose
unchanged initializer. Seal choice/parameter+factor extents BEFORE new DEV metrics.
After seal evaluate BOTH initial and selected on all4386 DEV labels;DEV cannot
choose candidate or thresholds. No second gradient or unbounded iteration.

## Quality, cost and scope

Report weighted case and unweighted label KL/IDs,all24 cases and12 domains.
Quality requires case meanKL<=.01/disagreement<=.01,each label worstKL<=.1,
everycase KL<=.05/disagreement<=.05,eachdomain caseKL<=.05/disagreement<=.05.
Separate component decision verified FIT descent;quality failure cannot admit
chatbot,causal state acquisition,internal operators,native precision/useful n/50.
Generic corrected factors keep rank<=256,not W^T V=I or prior projector optimum.
Norm/skip/core transfer must be explicitly qualified before composed use.

CPU6/affinity0..5;900s/OS6GiB/output3GiB/no children. Full relevant hashes before
AND after;source/protocol freeze,binding/argv/PID/create/exit/terminal. Record
baseline gradient/preflight wall;no clean/native timing admission. No source
model/GPU/native/T4/RESERVED/longalign calls;external publishers and foreign work
retained. No hours-long job selected. First execution fault retained;repair only
new source/version/namespace. Complete direct probability/gradient/parameter
update/selection audit regardless of scientific failure.
