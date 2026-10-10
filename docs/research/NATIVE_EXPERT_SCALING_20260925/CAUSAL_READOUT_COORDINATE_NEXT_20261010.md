# Next: qualify causal coordinates, then categorical head adaptation

10 October2026. Prospective, no new feature reconstruction/fit/native replay yet.
[Completed head-image evidence](PAIRED_HEAD_IMAGE_RESULT_20261010.md) exposes
encoder room; it does not require that the next encoder be nonlinear. The
current quadratic linear encoder fails, but another KL-trained linear encoder
has not been ruled out. Choose a control that connects directly to original C
causal states and can produce an unchanged-operator artifact.

## Reuse an actual causal path without a history replay

Actual27 [native baseline](ORIGINAL_ENGINE_BASELINE_RESULT_20261009.md) retains
all48 FIT/DEV full-V prefix logits, strict packed-only original C computations,
exact prefix/split/reuse bridge, and complete stored adjudication. Quality fails
(DEV caseKL8.76437/tasks0/16); numerical/GPU bridge limits remain inherited.
Use actual27 because it already has all24 FIT histories in C, not because it was
selected after a new quality sweep. The matched actual51 runs have DEV logits,
not an equivalent complete stored FIT feature cohort.

Let H be the exact F32 old65537x256 head decoded to F64, f the original F32
post-final-norm causal state, and z the retained full native logits. Ideally z=Hf;
in reality native dot rounding contributes delta. If H has full column rank,
QR/SVD can recover fhat=H^+z, with uncertainty governed by sigma_min(H).
This is a new stored observable; no original native history/optimizer inference
is rerun. It is NOT bit-exact recovery of f and cannot hide floating-point error.
Prove/measure rank, singular values/condition, QR identities, every full-V score
residual and norm, and a conservative forward-dot/feature error bound before
using fhat in learning. Existing original dotf/rmsnorm bodies and packed final
norm constrain this calculation. Handle adopted row_offset/positions exactly.
If rank/conditioning fails, report it and obtain fresh native final-state
witnesses with an explicit qualified observer; do not silently regularize and
claim native state identity.

Old head C prefix files metadata sum7,668,091,148B across48 records (some shared
paths); packed candidate_seeded27.packed520,029,440B/SHA7d419a9ecb8fb5a9f3687388df5f286554166954ccc0c81a31fe866a5bdf2652.
Verify unique consumed files, not the unrelated87GB historical audit collection.
No source model weights, source inference, old masters/Adam/RNG, or full old
training replay are needed. Select existing output-label positions only,8808
rows=4422FIT+4386DEV; DEV feature qualification must be independent of teacher
scores and fit decisions. Full context histories remain the already generated C
histories; this is teacher-forced evaluation, not free-running chatbot success.

Before this substantial calculation: implement/freeze a new CPU tool, parent
custody links, unique input extents, numerical gates and resource/stop policy.
Forecast tens of seconds to a few minutes including8GB input hash traversals;
proposed600s held worker with60s reserve,OS3GiB/output256MiB/noGPU, then separate
600s/OS3GiB full stored audit. Refine caps before launch from actual input sizes.
Save factors/features and per-label residuals so audit needs no QR replay or
native/source call. No quality-triggered audit omission.

## An operator-closed categorical adaptation

If coordinates qualify, keep original causal core/embedding/ternary functions/
router/old final norm fixed. Let A=aK from the newly paired codec, and learn
J255x256 with x=J fhat. Target p_J=softmax(AJ fhat).
For fixed features and A, mean teacher KL is convex in J:

objective_j = logsumexp(AJ fhat_j) - (A^T q_j)^T J fhat_j + c_j,
gradient_J = sum_j w_j [A^T(p_j-q_j)] fhat_j^T,
c_j=sum_v q_jv log q_jv.

Only65,280 trainable coefficients; teacher moments/entropy can be cached once.
Use FIT only, equal-case weights. A FIT-only covariance/least-squares initialization
from already stored phi/a may improve conditioning; choose its support cutoff,
initialization and solver before observing a new fit. Do not interpolate72 oracle
codes and claim full4422 knowledge transfer. No DEV head/gain/optimizer selection.
One fixed method, bounded updates with independent complete stored audit.

Export H_new=(A J) as an ordinary F32 Vx256 head. The product is fused OFFLINE;
original final norm and original matvec consume it. Runtime gets no extra
matrix, square-root carrier, iterative inversion, donor or new readout kernel.
The final head has a new effective domain induced by J and the old norm, so the
radius16 diagnostic bound cannot be reapplied silently. Head coefficients really
change here, unlike the completed fixed-head latent probe. It is a new converter
branch, not resuming a frozen failed dose. Do not pretend the core's coordinates
already equal the new whitened source coordinates.

Measure real F64 fit and F32/native reconstruction sensitivity to the recovered
feature uncertainty. Then run the unchanged packed C consumer on the newly
exported artifact for exact FIT/DEV numerical and quality gates; cached feature
prediction alone cannot qualify it. Original arithmetic/GPU surrogate mismatch
from earlier work remains open; native evaluation is authoritative. If useful
held-out readout recovery appears, proceed with compact causal joint adaptation,
generation/tasks and same-artifact throughput. If the constrained readout fails,
separate finite optimization, insufficient current causal features and remaining
head image before selecting the next representation/history correction.

## Scope

No useful-n/DRAM/structured CPU mass/family/speed or useful-chatbot admission.
Native core is a useful starting scaffold, not a proven information-preserving
history encoder. Feature reconstruction is the exact next implementation point;
no unconditional long training campaign or unchanged latent dose is authorized
by these numerical results. Goal unchanged; source-adaptation branch stays frozen.
