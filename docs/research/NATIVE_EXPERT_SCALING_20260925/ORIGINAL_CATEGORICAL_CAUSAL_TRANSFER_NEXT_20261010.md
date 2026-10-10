# Next converter component: fixed-head categorical supervision for original C

10 October 2026. Compiler/FIT consumer and original causal preflight COMPLETE/full audits PASS.
[Actual causal result](ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_RESULT_20261010.md): numerical bridge qualified, case quality FAIL; all owned CLOSED.
[Grouped parameter steps](ORIGINAL_CATEGORICAL_TRUST_STEP_RESULT_20261010.md) COMPLETE/full independent audit PASS, actual C local descent qualified, absolute quality FAIL.
Next [changed-state GPU/backward+full route ID/mass comparison](ORIGINAL_CATEGORICAL_CHANGED_HISTORY_NEXT_20261010.md) PROPOSED/UNIMPLEMENTED; multi-case training UNEXECUTED. Full goal incomplete.
This replaces the completed first-state/global-head diagnostic as the next
operational step. It does not resume an old optimizer or donor-adaptation run.

## What the new evidence changes

The [analytic correction](ANALYTIC_ONSET_HEAD_RESULT_20261010.md) establishes
that fitting 24 selected first codes exactly with one global linear map can
destroy continuation predictions and fail to generalize to DEV first states.
It does not establish an information ceiling or the impossibility of a learned
nonlinear causal map. The current core already fails before candidate feedback.
Another unchanged head fit or more steps of the old optimizer would not test a
new transfer mechanism.

Return to learning the function from token history to a categorical state,
using original SSM/SWA, activation quantization, selected ternary banks and
normalized routing mass. Preserve the original computation and cost envelope;
do not put the donor's full history computation into inference.

## Exact sufficient statistics for a fixed executable head

Let H be a FIXED actual F32 head interpreted as real coefficients, q the
qualified source distribution at a canonical teacher boundary, and f the
target's post-RMSNorm state. Define, in nats,

    m = H^T q
    c = sum_v q_v log(q_v)
    z = H f
    L(f) = log(sum_v exp(z_v)) - m^T f + c.

Then L(f) = KL(q || softmax(H f)), and

    gradient_f L = H^T softmax(H f) - m.

These identities hold for EVERY f, not just a teacher projection, a stationary
code or the old native states. A teacher boundary can be compiled once into
256 moments and one entropy scalar. This preserves its loss and state gradient
for this fixed head in real arithmetic. Full-V target normalization and head
work remain in training and inference; this is a supervision/storage reduction.

The tuple is not enough to train a changing H, reconstruct arbitrary q, or
evaluate another head. Retain source scores and their custody. Changing H
requires recompilation or the original q. Do not differentiate through m while
claiming fixed-head equivalence. Finite precision in q normalization, moments,
GPU head products and gradients must be measured separately. In particular,
rounded F32 logits are not an exact linear real map merely because the weights
are F32; an exact real identity is not an automatic bitwise software identity.

## Head and norm convention

Use the existing paired native head, not the failed global analytic correction:
`paired_output_codec_20261010/head.f32`, 65537 by 256 F32,
SHA256 `768aaf9e17873707624e4489a8f14f345afb2722e256de871113b7d6e64baa66`.
Its last column is zero and it is paired with original epsilon 1e-5, final norm
weights equal to one, and the 255-coordinate constant-norm carrier convention.
The [72-code probe](PAIRED_HEAD_IMAGE_RESULT_20261010.md) already exercised
actual original C norm/head bodies; do not repeat that oracle optimization.

For a reference code x with norm at most 16, the ideal raw state is

    u = [x ; sqrt(256 - ||x||^2)].

The real head before its F32 cast is [A sqrt(1 + epsilon), 0], so original
RMSNorm followed by that head yields A x. This is a feasible reference state,
not a new square-root operator added to inference. The original core must
learn to produce useful raw states; full categorical loss uses its actual norm,
and does not require it to reproduce each oracle coordinate exactly.

The paired quadratic encoder failed FIT quality (KL .8041); selected optimized
codes have mean KL .00653 but maximum .23083. Neither establishes that this
fixed head can pass all whole-corpus gates. Keep that limitation visible.
If a measured fixed-head obstruction appears, revisit the decoder or conditional
output representation rather than hide it by redefining success.

## First bounded executable stage

Implement a loss compiler with a NEW code/protocol/binding/output namespace.
Read the existing qualified 48 FIT/DEV teacher cohorts (8808 labels) and the
fixed native head. Store moments, entropy, source argmax, canonical positions,
split/domain and complete source/head extents. Keep FIT and DEV arrays separate;
the future training loader must accept only the frozen 24 FIT identifiers.
Repeated DEV evidence is development evidence, not untouched final admission.

Use row-max-shifted probabilities. Verify all compiled statistics independently
with another normalization and vocabulary-block accumulation. Add scalar
`math.fsum` witnesses. On the already stored 72 native-normalized probe states,
compare the compiled loss and state gradient with the direct full-q definition
using this SAME fixed F32-as-real H. Those are new supervision-identity checks,
not another 512-step code search or a new C history rollout. Account for every
head probability/gradient row used by these checks.

Freeze tolerances, actual runtime modules and all extents before observation.
Forecast one compiler plus full independent audit, roughly minutes, with proposed
300 s per stage, OS 2 GiB and output 128 MiB; review actual shapes and resource
feasibility before binding. No source forward, new teacher generation, SVD,
optimizer, native history, RESERVED query, GPU or T4 allocation in this stage.
Do not stop a full audit on scientific failure. Persist numeric maxima before
assertions so a failed comparison retains the decisive observable.

Success makes a reusable converter/training input real. It does not admit a
useful chatbot, family transfer or engine speed.

## Subsequent causal experiment, after compiler qualification

The compiler and FIT-only consumer now pass complete qualification; see
[compiler result](ORIGINAL_CATEGORICAL_LOSS_COMPILE_RESULT_20261010.md).
`original_categorical_history_learner.CategoricalHistoryLearner` is implemented
and AST checked, not GPU/native qualified. It preserves the existing widened
SourceLearner original operators and streamed bank adjoint, exposes postnorm
features at canonical positions, and installs the paired F32 head/norm coherently
after loading the actual27 core/banks. Head/final norm remain fixed; validate
their values/dtypes against the constant supervision buffer before each loss.

Training readout uses F64 products of the EXACT F32 head coefficients and F64
moments/entropy. This avoids claiming an exact compiled identity with separately
rounded F32 logits. Core/ternary surrogates remain F32; conversion of upstream
gradients into those operators still needs the preflight. Packed C retains the
ordinary F32 head/norm; the training-only F64 buffer adds134219776 B to GPU
storage and16777472 multiply-adds per supervised readout label. Actual costs and
GPU/C rounding differences are unmeasured for this adapter, not admitted by the
CPU compiler proof. Changing H, enabling AMP on its buffer or training norm
weights silently would invalidate the stated geometry.

Deterministic candidate for first preflight: shortest existing FIT history,
ties by identifier, `broad_fit_explore_instruct_rewriting_008`,58 input IDs/18
teacher labels. This selection uses lengths, not new loss outcomes. Implement
and bind the runner before observing its model outputs. Reuse immutable actual27
model/packed/executable and source/FIT compiled extents; declare measured prior
training costs, GPU/OS/output budget, exact original native prefix comparison
and dense/compiled loss/state-gradient gates. No truncation or new source
generation. One NEW changed-head/norm case forward/backward/native bridge, no
optimizer step or replay of old head/model histories. Fully audit retained
outputs even if quality/numerical gates fail, then decide whether training can
begin or the bridge/initialization must be corrected.

Create a NEW initial model and optimizer namespace. Start from the retained
original actual27 core/bank tensors; replace head and final norm coherently.
Do not attach the new head to an old checkpoint and call it qualified, or resume
Adam moments that belonged to another objective without a documented choice.
First qualify one frozen FIT forward/backward and native packed bridge under the
changed head/norm, comparing dense-q and compiled supervision and accounting for
ternary surrogate derivatives versus actual native forward arithmetic.

Then consider one bounded causal/core/bank training campaign with an explicit
first-response term, for example the prospective equal-case objective

    J = (1/24) sum_i [0.5 L_i,0
                     + 0.5/(n_i - 1) sum_(j > 0) L_i,j].

The previous onset mass was only 1.17535%. This new objective gives the start of
each answer explicit influence while retaining continuation supervision. It is
a proposed objective, not a claim that balancing solves generalization.
Freeze initialization, trainable groups, step/data budget and DEV decisions
before execution. Existing training costs must determine the local budget;
do not infer a long T4 budget from this algebra or allocate T4 silently.

Quality gates remain donor-relative case/domain KL and disagreements, original
chat tasks, candidate answer follow-up and own-history behavior. Assess actual
C on the SAME exported artifact, then isolated full end-to-end accepted tokens/s.
The 104 fresh plus 64 older RESERVED cases remain untouched. Useful RAM-driven
n, structured CPU expert IDs AND normalized mass, physical DRAM and another
family/scale are still required. This first campaign only addresses the missing
compact causal transfer; expert scaling cannot substitute for that success.
