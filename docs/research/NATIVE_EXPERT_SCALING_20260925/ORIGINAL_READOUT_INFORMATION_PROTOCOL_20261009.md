# Prospective current-readout information probe

9 October 2026. Goal ACTIVE/INCOMPLETE. This is a FIT-only mathematical diagnostic,
not a useful model, new whole-model dose, DEV measurement or quality admission.

## Question, reuse and exact selection

Which part of the observed KL11.60 ->10.58 is imposed by the **current fixed**
original D256 RMS/head, even with freely chosen internal features? This decides
whether to correct readout initialization before pricing broader whole recovery.
The head is trainable; this cannot prove a universal D256 capacity ceiling.

Reuse the actual Adam1 candidate.packed and complete native.f32 of the
[source-informed transfer](ORIGINAL_FALCON_TRANSFER_RESULT_20261009.md), and cached
broad FIT teacher distributions. No model/native/teacher forward, parameter
update, donor-state capture, RESERVED or T4. All original numerical FAILs retained.

Sort all24 FIT case IDs lexicographically. For each select saved label indices
0, floor(labels/2), labels-1: exactly72 labels/12 domains. Selection is metadata
only; all full-vocabulary65537 distributions included. Actual student comparison
is only the three byte-matched native rows of broad_fit_smol_magpie_ultra_022;
no student distribution exists for the other23 cases at this actual state.

## Algebra, controls and optimization

Original RMS gives u=x/sqrt(mean(x^2)+1e-5), ||u||<sqrt256=16. Its closure is the
radius16 Euclidean ball. Current head/final_norm give A=head*diag(gamma), no output
bias. Common vocabulary-row mean is subtracted from A, probability-invariant
in real arithmetic. Probe two fixed controls A and sqrt8*A, without changing
the native operation count. Scaling is an initializer candidate, not preservation.

For each cached source q independently minimize convex KL(q || softmax(A*u))
over ||u||<=16. Evaluate full probabilities/objective/gradient in Torch CUDA F64,
no TF32/AMP or autograd; real saved BF16 source logits convert exactly to F32 then
F64. Feasible features do not construct history-dependent functions or a chatbot.

Projected gradient descent starts u=0, individual L=1. Before each update halve L
(floor1e-12); backtrack each failing row by doubling L until the convex quadratic
majorant holds, with absolute F64 slack1e-12; at most30 trials/update. Projection
is u*min(1,16/||u||). At most512 updates/arm; no restart or fallback dose. All
reported objective/gradients and full distributions are F64. Check finite values,
probability normalization<=1e-12, descent<=1e-9 and feasibility<=16+1e-10.

At every feasible iterate record upper f(u) and tangent lower
f(u)-grad(u).u-16*||grad(u)||. Running best feasible upper and maximum lower
(including KL>=0) bracket the real-arithmetic optimum. This is F64 evaluation
of an analytical certificate, NOT an interval-arithmetic proof. Reject numerical
bound inconsistency>1e-8. Stop when all72 gaps<=1e-5 or512 updates. An unresolved
gap remains unresolved; a stalled solver is not a rank theorem. Save all72 best
feature vectors/control, per-label bounds, norm/argmax diagnostics and traces.

## Frozen decisions

Report independently, without exhaustive model-capacity claims:

- Current feasible mean KL<=1: readout can express a sampled optimistic mean
  distribution within that loss; prioritize history/conditional recovery.
- Current lower mean KL>1: the CURRENT fixed image imposes material sampled
  loss; correct its initialization/subspace before an expensive fresh-core dose.
- Scaled feasible mean KL<=.75*current lower mean KL: certified substantial
  improvement between these fixed amplitude controls; price amplitude correction.

Uniform baseline is log65537-H(q); report teacher entropy and matched native loss.
If both fixed readouts have large lower bounds, a scale-only fix is insufficient;
reconsider projected head/subspace before inferring missing whole-model capacity.
Flags can coexist; compare numerical bounds, not optimizer convergence alone.
No native quality/parity/rate/generalization admission follows any flag.

## Cost and stopping

One local family<=600s including pre/post input hashes, worker reserve45s;
OS worker+launcher<=4GiB, GPU PyTorch allocated<=3GiB/reserved<=4GiB,
new result namespace<=8MiB and worker log<=4MiB. No worker descendants. All
worker/launcher live memory observations use the held Windows process handle
through exit. GPU scope is allocator, not total driver memory. Preflight hashes
are preparation cost, separately observable. No other Python/compiler/native
benchmark overlap; exact publisher exception retained. Save first failure;
observation timeout alone is not exit and never warrants a restart.

Script: [original_readout_information.py](../../../benchmarks/native_expert_scaling/original_readout_information.py).
Binding/launch commands and actual revision/hashes/results will be recorded after
the prospective code/protocol/binding freeze. Reuse source Adam1 unchanged.
