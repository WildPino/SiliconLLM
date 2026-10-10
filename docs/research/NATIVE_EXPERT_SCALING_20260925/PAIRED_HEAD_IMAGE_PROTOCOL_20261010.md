# Paired head/domain information probe: frozen prospective protocol

10 October 2026. Goal incomplete. One new uncertainty, one fixed-head arm.
Prior: paired output codec numeric PASS, FIT quality FAIL; full stored audit complete.
[Algebra and decision rationale](PAIRED_CODEC_HEAD_IMAGE_NEXT_20261010.md).

## Question and custody

Can arbitrary feasible codes recover the cached donor distributions through the
new paired head, or does this fixed head/domain already have a quality floor?
Reuse K65537x255, gain a=4.328941201436105, rho1, original epsilon1e-5,
D256 carrier construction, sealed original F32 native norm/head and coherent FIT
teacher. No new head, source generation, history, model weight, DEV or T4 call.
The old Adam1 head/scales and old oracle are not rerun. Changed head/domain and
warm phi/a are the new variable. Numeric head fidelity is separate from quality.

Exactly first/middle/last output-label indices of each of24 FIT cases, metadata
selection independent of scores:72 labels/12 domains. Initial x=phi/a, norm<=8;
optimize ||x||<=16. A=aK; center columns over vocabulary (probability invariant).
F64 categorical KL, gradient A^T(p-q). Projected descent with row backtracking,
max512 updates,30 backtracks/update, initial L1, halve L between updates with
floor1e-12, double rejected row L, majorant allowance1e-12. No random starts,
restarts, additional scale or fit arm. Stop only all72 bound gaps<=1e-5,512 updates,
or mechanical/resource limit. No quality-triggered audit omission.

For each code preserve feasible upper KL and the actual point supporting largest
max(0,KL-g.x-16||g||) lower. These are convex tangent-ball numerical F64 bounds,
not interval certificates. Preserve checkpoints every16 updates and final point,
current gradients/losses, best upper/lower codes and line-search L. Save selected
teacher, initial/upper/lower full scores F64, gradients and original native72
readouts. No claim that a causal compact core can produce these optimistic codes.

## Predetermined decisions and numerical gates

Encoder room: sum72 feasible upper <=.25 sum72 initial KL. Whole-case lower is
sum of the three selected lowers / actual full case label count (omitted labels
have lower0). Whole cohort = average24 case lowers; domains = average two case
lowers. Head/domain floor present if cohort>.01 OR any case>.05 OR domain>.05.
Mixed/room/floor/unresolved branches preserve both observables. The mean72 lower
is NEVER interpreted as a mean4422 bound. Sample feasibility requires mean upper
<=.01, max upper<=.05 and disagreement<=1%; this does not admit full FIT or DEV.

F64 stored losses/scores absolute1e-9, gradients1e-8, bound calculations1e-8,
initial KL versus sealed prior1e-10, radius roundoff allowance1e-10. Carrier
radicand may be rounded to zero only inside this declared allowance; preserve
minimum and count. Native raw score discrepancy<=1e-4, mean real/native KL<=1e-6,
argmax discrepancy<=1%, norm relative<=1e-5. Complete audit even on quality or
native precision failure. Existing original four function bodies sealed exactly.

## Costs, controls and audit

Expected one arm1-3min on local RTX3060, reused old two-arm190.64s forecast;
max300s held launch including pre/post hashes, worker reserve30s, OS4GiB,
GPU allocated2GiB/reserved3GiB, output256MiB/log8MiB. Audit held300s/OS2GiB,
output2MiB/log8MiB and no GPU. Stop and preserve first mechanical fault;
no consumed code mutation or completed optimizer replay. No owned timing overlap;
existing launcher enforces process inventory. Foreign publisher is untouched.

Count actual objective/gradient calls including every backtracking evaluation;
latent row updates=72*iterations. Beyond fg calls, six72-row score products are
performed (three saved scores, two argmax/reference scores, one uncentered native
reference). Exactly72 original final-readout rows in one DLL call; full engine0,
head fits0, source/history/head forwards0, model parameter updates0, DEV0.
This is latent convex optimization, not pretrained weight training.

Independent CPU F64 stored audit checks all bound inputs/output hashes, exact
selection/teacher/phi custody, all initial/upper/lower full scores and final loss/
gradient/certificate points, every saved current objective/gradient, checkpoint
feasibility/monotone envelopes, original native state/norm/F64 head reference,
all metrics, grouped lower semantics, decisions/call counts/resources. No descent
update, eigensolve, native binary call or donor replay in audit. Checkpoint upper/
lower point shape/feasibility and monotonicity checked; final best certificate
points independently recomputed for every label. Audits are counted separately.

Binding seals this code/protocol, prior numeric result/audit/receipts, paired and
source metadata,24 teacher/phi files, current head/decoder/norm/native DLL/C and
engine, qualified helpers/runtime. Reused3.5GB old score streams and full donor
weights are not consumed by this probe. Code/binding freeze commit before launch.

No useful chatbot, >=50 tokens/s, ternary experts, LUT/routing mass, useful n,
physical DRAM or cross-family admission. After audit choose next correction from
measured head floor and encoder room; do not tune unchanged head indefinitely.
