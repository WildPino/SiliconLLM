# Frozen all-label entropy-mixture dual witness

12 October2026. All4422 FIT/4386 DEV,2 frozen recorded distributions,8 lambdas
[0,.25,.5,.75,.9,.99,.999,1]. Require complete oracle audit PASS first.
New dual witness family,not more primal optimization/head fitting. CPU only;
no source model/GPU/T4/RESERVED/longalign. Actual full source/operator q/current
actual F32 head promoted F64,original weights/quality budgets unchanged.

Candidate0=softmax(A_aug final_relaxed_v),candidate1=softmax(final_actual_native
F32 score words). Any simplex p is a valid dual witness;candidate1 moments are
computed from its own full probabilities,not copied from a real-code gradient.
Teacher q recomputed from actual BF16 source head/gamma/multiplier/raw H/sourceeps.
Stable full softmax,explicit normalization,q moments/entropy must match original
qualified source moments relative1e-9/entropy absolute1e-8. Probability mass1e-11.

r=(1-lambda)p+lambda q,normalized explicitly. Store mixture entropy,all257 moments,
mass,support_K(m_q-m_r),rawD=H(r)-H(q)-support,displaymax(0,D-1e-8) for every
lambda/family/label. Mixture moments use exact linearity with explicit normalization
mass;independently direct-multiply selected mixture moments and require max row
delta<=1e-9. Lambda1 endpoint absolute<=1e-8. Secant-slope concavity slack1e-8.
Tie selection:first maximum in family0..1/lambda0..7 order. No floor admission
before full independent entropy/moment/support/source probability audit.

Entropy0log0=0,no probability clipping. Count all q/p zeros;source stable-logit
reverseKL and derivative-at1 are diagnostics. Formula admission only when BOTH
represented distributions have no zeros. For source exp underflow retain log
upper bounds on omitted mass/entropy from zero entries;do not silently claim
exact untruncated real q. Numerical normalization/tail differences are part of
scope;allowance1e-8 is not an interval certificate. Nonzero mathematical λ1 floor
must not be fabricated by forcing its numerical endpoint to0 before checking.

All8-lambda moment grids stored. Select maximum displayed lower per label;
original equal-case/domain weighting for numerical real-domain budget violations.
Real half-ball lower applies to real native hemisphere subset,not a proved global
F32-machine floor or a model/causal capacity bound. Zero lower means this witness
family uninformative,not optimumzero or head adequate. No trajectory convergence.

CPU6threads/affinity0..5,batch64;900s/OS6GiB/output512MiB,no child processes.
Fresh output/result/terminal PID/creation even on failure;full consumed/output
before/after hashes. Independent CPU full audit900s/OS6GiB required;different
source feature/log normalization/moment/entropy/support reductions,all labels/
families/lambdas/selection/case-domain values/underflow handling,not sampled.
No runtime/clean timing/native useful tokens/s inference. After witness decision,
return to decoder/core/conditional transfer with causal information requirements.
