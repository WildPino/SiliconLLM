# PQT-008: global adaptation during complete progressive conversion

Prospective, 6 October 2026. Previous goal turn made empirical progress:
PQT-007 was completed, independently replayed, retained and adjudicated.
All 169 PR local objectives improved, yet complete-model preservation failed.
PQT-008 tests a distinct global adaptation hypothesis; no outcomes exist yet.

## Question, fixed reference and scope

Can calibration-only source-distribution adaptation during staged hard ternary
conversion preserve complete-model behavior better than the same adaptation
after one-shot conversion, at the same applied optimizer-update budget?

Original Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, original safetensors 988,097,824
bytes, SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
Qualified PQT-007 T4x2 software image/runtime, F32 SDPA, deterministic operations,
TF32/AMP/dropout disabled. One frozen teacher on cuda:1 and a student on cuda:0;
independent audit subsequently uses cuda:1 after fitting processes terminate.
No new pretraining. Every unique 2D matrix is finally ternary: 169 matrices /
493,961,216 coefficients, shared embedding/readout counted once. All 71,552
original norm/bias coefficients remain F32 and frozen. Full unique parameter
count 494,032,768. No silent matrix exemption or detached original readout.

[LLM-QAT](https://arxiv.org/abs/2305.17888) motivates source-distribution
distillation at low precision; its 4-bit LLaMA findings do not establish this
ternary procedure. The present experiment uses fixed real calibration contexts,
not that paper's generated-data procedure, and quantizes weights only. The
[straight-through estimator](https://arxiv.org/abs/1308.3432) motivates a
surrogate gradient; it is not the derivative of the discontinuous hard function.
No implementation copied or paper-level performance reproduction claimed.

## Three final arms and exactly matched update schedule

D: unadapted original-weight group-64 symmetric ternary baseline. ONE:
convert all 169 matrices before the first update. STAGED: convert the tied
embedding/readout first, then transformer blocks 0..23 in source order, all
seven matrices of a block together. Eight updates at each of 25 stages gives
200 updates; then 56 additional updates with all matrices ternary. Thus STAGED
has all matrices ternary for updates 193..256 (64 updates), while ONE has 256
fully ternary updates. This exposure difference is part of the tested schedule,
not a claim of matched fully-ternary exposure.

ONE and STAGED each start from the same original latent F32 matrix weights;
each has exactly 256 applied updates on all 169 unique matrix parameters.
Unconverted STAGED matrices remain trainable float until their scheduled stage.
No vector updates. Reset weights and create a fresh optimizer between arms;
no warm start from another fitted arm. Every parameter's Adam counter must
equal 256; check actual counters after every update, not just loop iterations.

Scales fixed throughout: original-weight group-64 L2-optimal scales, same
source-grid algorithm as PQT-007 D, ties to zero at +/-scale/2. Hard forward
uses exactly `code*scale`. For converted matrices a custom autograd operation
passes the weight gradient unchanged to the latent F32 weights; scales get no
gradient. Unconverted matrices use ordinary F32 weights. Shared embedding and
readout refer to the same latent parameter and fixed scale tensor, and both
become hard together. No clamp, margin loss, bias adaptation or weight decay.

Adam: lr .0002, betas (.9,.999), epsilon 1e-8, weight decay 0, foreach/fused
disabled. Global gradient norm clipped to 1.0, error on nonfinite norm.
All latent weights, gradients, logits, losses and exported values must be
finite; invalid or missing gradients/counters stop the run. Identity-surrogate
and true softmax-loss derivatives are separately qualified on synthetic data.
Record hard code changes for every matrix relative to D; a training process
with no hard code changes is not evidence of effective discrete adaptation.

Calibration: same first sixteen WikiText train windows of 129 tokens; first
128 positions each, 2,048 unique positions. Immutable train revision/file
identities are the PQT-007 bindings. Seed 20261019 NumPy default_rng: concatenate
sixteen independently generated permutations of the sixteen window IDs,
one complete window per update. Both arms use the identical 256-window schedule.
Each arm sees 32,768 position exposures, not 32,768 independent samples.

Teacher source F32 logits are calculated once per complete 128-position window
with its original head and retained exactly. Each training update computes
student F32 logits at those same positions and its own current head. Cast
teacher and student logits to F64 for log_softmax and mean KL(P_teacher ||
P_student), temperature 1, averaged across 128 positions; gradients propagate
to F32 student matrices. No candidate/source evaluation or generation data in
the fitting objective. Retain every applied-step loss, global norm, stage,
active matrix count, parameter counters, costs and final code identities.

Only final checkpoints are exported/evaluated. No best-online-loss checkpoint
selection, learning-rate search or extra repair updates. Final payload contains
ternary codes, fixed scales, counted original vectors and metadata only; latent
F32 weights, gradients and Adam buffers are fitting memory/costs, not deployed
float correction paths. All three payloads freeze before new news selection.

## New evaluation identities and fixed decisions

Same immutable AG News test source as PQT-007, not an unseen provider split.
Explicitly exclude consumed row IDs 301,6594,2327,4122,4351,2183,3136,6864,
3109,1754,5,2586,6999,1230,3104,117. Bind prior source/selection identities in
PQT_008_EVALUATION_EXCLUSIONS.json. After all final payloads freeze, permute
7600 IDs with NumPy default_rng seed 20261019; take first sixteen eligible
rows with >=129 pinned-tokenizer tokens, unique first-129 arrays, and first-32
prefixes different from every prior consumed row prefix. No label/loss-based
selection. First eight are 129-token prediction windows; next eight yield
32-token prompts. Verify role disjointness and no selected prediction first-128
or prompt first-32 sequence occurs in calibration. Insufficiency/collision
stops without relaxing rules. Audit independently reproduces every identity.

All earlier strict absolute gates remain: >=99% source argmax agreement over
1024 positions, mean KL <=.01, NLL delta <=.01 nats/token; own-prefix greedy
generation <=32 new tokens, >=95% position agreement with unmatched tails
counted as mismatches, >=7/8 exact traces; complete payload <=35% of full
988,065,536 FP16 bytes. Retain source/candidate states, choices, pre-choice
generation states, raw points and per-window results. Perplexity is diagnostic,
not semantic/classification accuracy. No changed retrospective PQT-007 gates.

Prospective relative schedule signal, separate from promotion: STAGED has at
least 10% lower held-out mean KL than ONE with no worse NLL delta, source
argmax agreement or generation position agreement, and identical raw code/
scale/vector byte budgets. Headers may differ; their actual bytes are counted.
Every absolute gate still required for preservation/storage promotion. If
relative signal fails, report that frozen staged schedule, not all schedules.

Retain final frozen calibration states and metrics separately from online
losses, including D. Check exported complete hard-model states exactly against
the fitting wrapper at final checkpoint before news selection. Never use
that consistency check as a source-quality gate or choose a checkpoint by it.

## Qualification, audit, resource limits and retention

Synthetic-only controls precede all original numerical access: exact hard
forward and ties, identity weight/no-scale derivative, independent scalar KL
gradient, tied lookup/readout gradient sum, genuine optimizer counter/code
change, stage boundaries and same schedule, codec/finite/malformed controls.
Check a tiny causal SDPA network backward on the actual remote hardware.
No local full-model loading/training or GPU fitting. Shared native CPU timing
still requires an explicit owner reservation, independent of this GPU job.

Independent process imports no fitting module. Rehash source/every final file,
independently decode complete payloads, verify fixed scales and frozen vectors
against original archive, parameter/tied coverage and hard changes against D.
Reconstruct teacher calibration logits exactly, both 256-window schedules,
all 169 optimizer counters and stage-active counts. It does not claim to
replay the whole training optimization trajectory; counters/code changes and
loss histories are separately checked. Replay original and each exported
complete model on all final calibration/evaluation states and every own-prefix
generation choice exactly. Recompute all final metrics/gates independently;
F64/F32 readout tolerance <=1e-5 and near-tie sensitivity recorded separately.

Total two-arm adaptation plus export <=7200s, experiment process <=7800s,
audit <=1200s, install <=900s, server <=10800s. Cap each arm at 256 updates;
no silent truncation on time or memory. OOM/nonfinite/resource failure preserves
first evidence and invalidates this run, not a numerical failure diagnosis.
Report both physical GPU peaks, process memory and all phase/arm costs;
PyTorch hard weights execute reconstructed F32, not packed native kernels.
Uncompressed whole-model archives count all NPY/ZIP headers and descriptors.

Fresh private acct3 `sirwildpino/pqt-008-global-staged-20261006-001`, pinned
qualified T4x2 image, original inputs hash verified. Exact committed source/
bundle and fresh owned-kernel/quota admission before single dispatch. Monitor
long T4 job through the existing delegated monitor and wait for terminal or
meaningful failure. Retrieve once, retain exact small raw evidence in the
dossier, large immutable arrays outside Git, independently adjudicate, update
TERNARY_INDEX.md. Goal remains active; success still needs native expert
transfer/usefulness and broader validation, failure is scoped to this budget.
