# PQT-012: natural original encoder experts with counted low-rank compensation

6 October2026. Prospective scope; no selected weight extraction, quantization,
calibration or candidate evaluation has executed. Dispatch only after SRC007003
actual raw/source/Git retention and complete numerical audit are documented.
Keep SRC007's final-decoder four-expert eligibility failure unchanged.

## Question and scope

Can behavior-aware progressive ternary weights plus a small, explicitly stored
linear residual preserve original pretrained FFN outputs on naturally routed
contexts more closely than direct/progressive ternary weights alone, within
35% of the original FP16 coefficient payload? This tests a mixed artifact,
not pure ternary weights or whole-model language capability. First qualify
the deployment artifact and local fidelity; native cost and whole-model
predictive/generative quality require later distinct experiments.

SRC007003's complete original float source/trace gate passes, but only final
decoder expert37 meets its fixed64/32/32-row and8/4/4-document eligibility.
The final encoder layer has118 eligible experts. Choose the final encoder
architectural counterpart prospectively; this changes the expert scope rather
than lowering the decoder threshold. In `encoder.block.11.layer.1.mlp`, take
the four eligible experts with largest calibration dispatch count, then ID
ascending. No candidate error, target spectrum or held-out error influences
selection. Freeze the exact coverage SHA63769518517d9368d2399b5534b879186b809620e0bf298b07f8f3bd5f67f5c3.

| Expert | Calibration rows/docs | Development rows/docs | Held-out rows/docs |
| --- | --- | --- | --- |
|81|321/119|295/119|299/119|
|87|238/70|243/72|250/77|
|18|236/107|274/120|267/117|
|91|229/38|180/31|124/26|

Exactly1,024/992/940 naturally dispatched rows, in actual batch/document/token
order. Include all qualifying rows, including sentinel/EOS positions. Do not
force routing, change capacity, select favorable rows or change splits. These
are the original384 frozen C4 masked-span documents and consumed source
evidence. No novel-domain, pretraining-exclusion or semantic novelty claim.
The smallest held-out document count is26; cluster metrics by document.

## Inputs and artifact identity

Two original F32 matrices per expert, WI3072x768 and WO768x3072, all eight raw
payload hashes from immutable SRC003 binding/google switch-base-128 revision
86c815ec05361a33a8b49fc717277da9c0a4e711. Use the qualified restricted ZIP reader
for selective, bounded read-only extraction after local process admission.
Do not modify source archives or the other checkout/environment. Count all
physical reads and exact selected NPY witnesses; this is a four-pair diagnostic
subset, not a checkpoint clone. Alternatively defer acquisition when admission
is unavailable; no unsafe source access or silent replacement.

Bind every X/Y row to its exact original capture archive/array hashes and its
router-selected document/token position. Recheck expert X against router X
and the exact capacity mask, full finite arrays, source/Git/coverage identities.
Retain all source and consumed input bytes in a private ZIP_STORED .bin capsule
<=128MiB, each member<=64MiB. Expected coefficient/context payload93,659,136B
before headers/manifests. Stream full SHA, exactly-one recursive mount, safe
paths/no executable pickle. Freeze source/input/controls before any T4 job.

## Fixed candidates and calibration

Use unchanged group64 F32 scales/direct/packing and progressive routines from
`ternary_math.py`: 2-bit codes00=zero,01=+1,10=-1,11 invalid. Direct scales are
the fixed per-group least-square ternary solution. Progressive column block128,
curvature damping0.01; WI uses original X, WO uses the candidate's post-ReLU
hidden calibration values. No gradient optimizer or learned scales. Both
complete original matrices become ternary codes; count actual committed
columns and every phase. PR is a baseline; it is not a new discovery.

Candidate bases D and PR, plus each base with residual ranks32/128/256:
D,PR,D-LR32,D-LR128,D-LR256,PR-LR32,PR-LR128,PR-LR256. Add group64 symmetric
I4 and rowwise symmetric I8 baselines. I4 uses unchanged `scale_math.int4`:
per-group clipping factors0.70/0.80/0.90/0.95/1.0, smallest original weight SSE,
first factor wins ties, symmetric codes-7..7. I8 uses row maximum/127 and
nearest-even rounding/clipping to-127..127. Zero groups/rows encode zero.
Exactly ten arms per expert,40 artifacts.
No post hoc rank/damping search or arm removal. Report every outcome.

Fit one residual regression per base/expert on calibration only. Let R be
the original actual Y minus the base FFN output. Center X and R on calibration;
lambda=0.01*mean(diag(Xc.T Xc/n)). Require positive finite calibration energy.
Compute F64 ridge C=Xc.T*(Xc Xc.T+n*lambda*I)^(-1)*Rc using a solve, then SVD
of Xc*C. For requested rank r use k=min(r,n-1,768), the first k right singular
vectors V, A=C*V and B=V.T. Store F32 A,B and b=mean(R)-mean(X)*A*B. Candidate
output is ReLU(X*QWI.T)*QWO.T+(X*A)*B+b. This is rank-truncated ridge output
regression, not a claimed optimal nonlinear reconstruction or reproduction
of another published algorithm. Retain the actual solve/SVD input/output,
means, singular values, factors and all serialized artifacts. Eight ridge
solves/eight residual SVDs/24 projected exports expected; zero optimizer steps.

Low-rank compression-error compensation is motivated by
[EoRA v6](https://arxiv.org/abs/2410.21271v6); activation-sensitive decomposition
is discussed in [ASVD v5](https://arxiv.org/abs/2312.05821v5). Those primary
sources motivate a method family, not evidence that ternary Switch experts
or this output-regression variant meet our quality/storage gates. Implement
the equations above directly; do not import unpinned external implementations.

Before fitting, verify original F64 WI/ReLU/WO on calibration rows against
actual Y at relative RMS<=1e-5. Seal all40 deployment artifacts before decoding
development/held-out numerical values for evaluation. Then perform the same
source-function check on those splits. Metadata/hashes may be checked earlier;
development/held-out values never enter scales, curvature or residual fits.
All worker evaluations use only reloaded artifacts, F32/no-grad/no AMP/TF32 off;
independent F64 reaggregation must agree at RMS<=1e-5. Retain all predictions.

## Metrics, gates and interpretation

For each artifact/split, relative RMS=sqrt(sum((candidate-Y)^2)/sum(Y^2)),
with denominator floor1e-12 on RMS(Y); retain actual sums, row/document counts,
per-document energies and2000 fixed-seed document bootstrap resamples.
Seed20261006+expert ID+split offset (calibration0/development1/held-out2),
independent NumPy PCG64; same within-expert/split resamples for all arms.
Intervals describe this document sample, not population or capability guarantees.

A candidate may advance only if every selected expert has development and
held-out aggregate RMS<=1% and bootstrap97.5th percentile<=1%; residual arms
also require RMS ratio to D<=0.9 and its bootstrap97.5th percentile<=0.9 on
both splits. Complete stored deployment artifact (packed codes, all scales,
factors/biases, metadata/headers) must be<=35% of9,437,184B original FP16
coefficients per expert. Preserve actual dimensions/effective ranks/storage
and full reload equality. Nominal D payload1,474,560B; residual FP32 factors
cost6144*k+3072B, before headers. No hidden floating original matrix, correction
or runtime cache is permitted in the deployment artifact. Source witnesses
are separate diagnostics and never accessible to its forward function.

Report F32/I4/I8 control behavior separately; an I8 storage failure does not
invalidate source identity. If no candidate passes, report a scoped failure
for these four natural encoder experts, method and data. If one passes, it
only justifies a subsequent whole-model/native experiment. Count all fitting,
export, reload/evaluation, installation, independent audit and failed costs.
No native speed claim follows from packed size, MAC counts or GPU timings.

## Execution admission and limits

Pinned image/runtime and both-T4 identity remain those of SRC007003. Private
dataset/new kernel namespace, fresh actual owner/quota/reservation/all-owned-
terminal admission. No overlapping remote job on the same account. Max server
5,400s, fitting/export worker3,600s, independent audit1,200s, install600s; clamp
each to remaining time. RSS<=8GiB, each GPU allocator<=4GiB, outputs<=1GiB.
Retain actual complete child clocks/exit codes and first failures. Actual
GPU expert fitting uses deterministic routines; CUDA math is distinguished
from the independent NumPy/F64 audit. Local preparation/controls require live
process admission, one CPU thread, complete child<=600s/RSS<=2GiB, no native
timing or local GPU. Preserve numbered first stops and freeze meaningful codec/
regression/row-order/leakage/nonfinite controls before dispatch. Delegate long
T4 status monitoring; coordinator dormant until terminal or meaningful failure.
No private write precedes exact source/input/Git qualification.
