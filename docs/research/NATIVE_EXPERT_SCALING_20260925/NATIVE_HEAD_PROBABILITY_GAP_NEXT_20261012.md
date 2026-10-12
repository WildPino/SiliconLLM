# Current native head probability gap: proposed next

12 October2026. PROPOSED/UNIMPLEMENTED/UNBOUND. No optimizer/GPU preflight run.
Prior [complete affine-ray result](NATIVE_HEAD_AFFINE_RAYS_RESULT_20261012.md):
2342/2361 actual native target codes,DEV case coverage98.6616%,but available KL
contribution9.6007 versus current predictor2.93425. Winner availability does not
settle distribution capacity or causal learning. Do not repeat winner-only rays.

## Information question

All8808 actual source probabilities have qualified sufficient moments m=A^T q,
entropy H(q) for current actual F32 head/bias/epsilon promoted F64. Fixed-head
full loss L(v)=logsumexp(A v)-m^T v-H(q),gradient g=A^T p(v)-m. Holding the head
fixed permits code/encoder training without storing full teacher distributions
or replaying the source model. If the head changes,moments must be recompiled;
do not treat them as sufficient for arbitrary changed readers.

For arbitrary reference v*,the exact identity is
`L_q(v)=L_q(v*)+KL(p(v*)||p(v))+(A^T p(v*)-m)^T(v-v*)`.
If an unconstrained information projection exactly matches moments,its last term
is zero:fixed-head irreducible loss and code loss separate. Approximate stationarity
or constrained optimum does not justify dropping the residual term. This identity
is an analytical diagnostic,not an observed floor or proof of a finite optimizer.

Native feasible v lies on positive hemisphere radius16;convex relaxation is
K={||v||<=16,v_last>=0}. For relaxed feasible v,the existing support lower
L-g^T v-16sqrt(||g_first256||²+min(g_last,0)²) bounds native optimum in exact-real
arithmetic. Retain gaps/negative bounds;zero lower does not prove attainable zero.
Numerical positive bounds need separately frozen error margins/independent checks;
they are not interval certificates. Relaxed loss is not a native feasible upper.

## Candidate algorithm and actual native uppers

Reuse historical projected-descent/backtracking skeleton only after explicitly
changing current257-dimensional head/domain/objective/reference. Projection onto
K:zero a negative last coordinate,then radial clip to16 if needed. This is a
convex relaxed problem. Frozen per-label majorants/monotonic losses/support lowers
and exact algorithm replay must be independently audited. Iteration/cohort/budget
not selected yet. Existing DEV per-label oracle fitting is not held-out learned-
model validation;head weights remain unchanged.

Construct an actual positive-sphere candidate from relaxed first256 coordinates u:
after an explicitly frozen finite-code boundary clamp,choose last=sqrt(256-||u||²)>0.
Reevaluate its full objective;do not reuse relaxed loss. Recover
`t=sqrt(256*epsilon)*u/last`,castF32,then call actual C and compute the resulting
real v from that actual tF32. Selected candidate must beat/retain the original
warm native code under explicitly frozen full categorical criteria,not top1 alone.
Bound real/native loss differences from full logits or independently compute full
source q/native probabilities;report which. All candidate native output bits need
complete independent scalar-reference verification before admission.

## Immediate next executable step

Implement/freeze/bind **GPUF64 initial loss/gradient qualification and cost only**,
all8808 preflight v/m/H and stored CPU losses/gradients,full current65537 head.
No fitting/optimization or new source acquisition. Deterministic local Torch2.6,
TF32 off,float64,batched memory/calls. Full CPU/GPU loss/gradient agreement,
fixed warmup/repeats and GPU/OS peaks/driver/runtime/source/binary extents required.
CPU prior median.203s per64/28.217s nominal pass. Existing paired255 GPU run cost
is contextual,not evidence for the new sufficient-moment objective or shape.
Freeze numeric gates/caps before observing;choose backend/cohort/steps after actual
cost. No T4 selected. If estimated run/audit hours long,announce frozen bound/stops,
launch exact job,then user-requested goal PAUSE;no Luna or delegation.

One bounded probability-image diagnostic must change a pipeline choice:if useful
native upper exists,use it as code-transfer supervision;otherwise change reader/
conditional output geometry before expensive compact-core training. A numerical
floor would concern only this fixed head/domain/cohort. Return to causal reusable
SSM/SWA/ternary experts afterward. No generic donor port,small-student/raw speed/
oracle lookup substitutes for useful RAM-driven n/multiple models/same artifact50.
