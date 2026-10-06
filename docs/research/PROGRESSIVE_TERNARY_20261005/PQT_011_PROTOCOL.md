# PQT-011: matched global adaptation with learned group scales

6 October 2026, prospective before apparatus execution or new model outputs.
Consumed-development method comparison; no promotion. This record freezes the
intended method and first apparatus qualification. Remote execution remains
conditional on a complete, independently auditable implementation and exact
input/source bindings. Do not dispatch the apparatus alone as a model experiment.

## Question and matched conditions

Four arms FIXED_ONE, FIXED_STAGED, LEARNED_ONE, LEARNED_STAGED. All start from
the same immutable original Qwen/Qwen2.5-0.5B source, revision
060db6499f32faf8b98477b0a26969ef7d8b9987. FIXED must reproduce PQT-009 B_ONE/
B_STAGED code, scale and vector bytes; any mismatch stops interpretation.
Learned variants differ only in scale adaptation. Both schedules are retained,
with no outcome-based winner selection or hyperparameter search in this run.
An additional direct D archive supplies original fixed scales, vectors and
code-change references; it receives no updates and is not an extra fitted arm
or a new evaluated/promotion candidate.

Reuse the exact qualified PQT-009-R1 broad calibration: 256 windows of129 tokens,
32,768 positions, one deterministic permutation and256 applied updates per arm.
The calibration arrays, all teacher-logit identities and original parameters
must be rebound exactly before execution. Original teacher frozen, F32 SDPA,
seed20261005, TF32false, same pinned runtime as PQT-009/010. All169 unique
matrices trainable; original F32 vectors frozen. Shared embedding/head weight
and group scales remain single objects with two functional uses.

ONE hardens all169 matrices from step1. STAGED hardens the shared embedding/head
at step1, then one original-order transformer block per eight updates; all169
hard from step193 through256. Original latent weights reset between arms.
Teacher-target generation, final fitted-set readouts, hard-export equivalence
and every actual per-parameter optimizer counter are recorded and checked.

## Exact representation and surrogate

Group64, two-bit packed ternary codes {-1,0,+1}, one F32 scale per original
row/group, all original F32 vectors and full archive overhead counted. Original
group scales alpha0 use the existing original-weight fit rule. Learned scales
are alpha = alpha0 * exp(r), r initialized to zero, one F32 latent r per scale.
After every update project r into [-ln(4), ln(4)]. Thus positive scale ratios
are within [1/4,4], subject to F32 rounding. Original zero-scale groups stay zero.
No additive correction or deployable r: export only final alpha, packed codes
and original vectors. Raw parameter budget remains154,649,088 bytes; aliases
count once. Training scale/Adam memory is additional and must be reported.

Hard forward: expand alpha to its64 coefficients, round latent weight/alpha
nearest-even, clamp to[-1,1], zero-scale code0, then code*alpha. No softened
deployed weights. The latent-weight backward is the existing biased identity
surrogate; it is not the mathematical hard-rounding derivative. Holding codes
fixed, backward for each scale is sum(gradient*code) over its64 coefficients.
Ordinary chain rule through exp gives gradient with respect to r. Code-boundary
derivatives are ignored. This declared rule is not presented as an unbiased
gradient estimator. Inactive stages use original latent float weights; add a
zero-valued dependency on r so scale gradients are explicit zero and counters
advance, with no pre-activation scale updates.

Adam weight group lr0.0002, scale-log group lr0.002; betas(0.9,0.999), eps1e-8,
weight_decay0, foreach=false, fused=false. FIXED uses only the weight group.
Learned variants clip the combined latent-weight/log-scale gradient norm to1;
FIXED clips its original weight norm to1. Changed effective weight update sizes
are part of this declared joint method; this is not a unique derivative-only
causal attribution. No learning-rate sweep or hidden normalization. Record
gradient norms, scale ratio extrema, saturation counts, zero-group count,
code changes and actual counters; stop on nonfinite values or changed tying.

## Qualification before source/data execution

First CPU stage uses only small synthetic tensors, one thread, no original
model/corpus/GPU or native timing. Freeze source commit before controls. Check
hard-forward byte equality against fixed-scale rules at r0; independent scalar
group-scale gradients and piecewise finite differences away from code boundaries;
explicit identity weight surrogate; zero groups, nearest-even ties, nonfinite/
negative/wrong-geometry stops; real Adam scale change and projection; shared
lookup/readout gradient sum; inactive zero gradients and actual counters.
Retain exclusive report and actual source identities, first failures unchanged.

Later qualification must additionally verify all169 bindings, complete toy
causal-network gradients/stage boundaries, packed final scales/codes through an
independent decoder, all scale counters and source-reset equality. First-stage
checks alone do not qualify the full experiment. The independent complete-model
auditor must not import the fitting driver or learned-scale gradient module.
It reconstructs every decoded parameter, checks original vectors/ties, all
target/history/final records and exact wrapper/export states; it does not claim
to replay the optimizer trajectory.

## Evaluation and decisions

Exact consumed PQT-009-R1 Wiki/news prediction8x129 and prompts8x32, label1024
positions/domain. Bind raw arrays and source/FIXED baseline output identities
before numerical execution. Own-head inference, at most32 own-prefix greedy
tokens, original EOS rule, unmatched tails count as disagreement. Save all
prediction/pre-choice states, choices, per-position and per-window metrics.
Separate independent nonlinear/choice replay and complete F32/F64 readout
diagnostic <=1e-5; separate standard-library reaggregation before interpretation.
For the final fitted set, independently replay every F32 position and a fixed
128-position F64 diagnostic at floor(i*(32768-1)/127), i=0..127; do not claim
an all-position F64 fitted-set bound. Evaluation-domain F64 covers every position.

Descriptive absolute gates separately both domains: argmax>=99%, meanKL<=0.01,
meanNLLdelta<=0.01, generation-position agreement>=95%, exact traces>=7/8,
complete archive<=35% of988,065,536 FP16 bytes and total four-arm fitting/export
<=7200s. Every counted byte includes final learned scales. Learned vs same-
schedule FIXED signal: >=10% lower KL, no worse NLL/argmax/generation-position
agreement, identical raw capacity, separately both domains; report joint and
per-domain signals. Also report staging vs ONE descriptively for each grid.
Even absolute passes on these consumed roles cannot promote a method.

## Resources, evidence and stops

No local donor/GPU/native benchmark during apparatus preparation. Full trial
would use a fresh private acct3 reference, two T4 devices, original teacher on
device1/student on device0. Before dispatch: exact corpus/checkpoint/runtime/
input and source bindings, completed CPU controls, private dataset
readiness if needed, fresh owner/quota/all-addressable-terminal admission and
single frozen bundle. New physical-device hard-scale and causal-gradient controls
run inside that admitted job before any original model/data function. Use the
existing strict historical inventory marker only.
Install<=900s, four-arm fit/export<=7200s, evaluation<=1800s, audit<=2400s,
server<=12600s; stop on budget/OOM/hash/selection/finite/replay failure.
The bootstrap experiment process is bounded to9300s, including setup, fitting
and evaluation; installation plus experiment plus audit caps sum to12600s.
These outer stops do not relax the separately enforced fitting/evaluation caps.

The exact consumed binding exceeds the self-imposed900,000-byte inline script
transport cap even after compression (945,436 base64 bytes for the binding
alone). This is a local precaution, not a confirmed platform source-size limit.
Transport that single frozen file as a private binary dataset, preserving its
4,112,180 bytes and SHA256 unchanged. Resolve exactly one mounted binary and
verify raw size/hash before copying it to the source directory. Retain its
exact raw source copy with terminal outputs and reverify against frozen Git.
Other executable sources and certificates remain hashed inline. No scientific
input, procedure or acceptance threshold changes with this transport.

Capture each arm/process CUDA allocated/reserved peaks and process RSS peaks,
setup/teacher/export/evaluation/audit times; no nested time double counting.
First failures and any numbered repairs are charged. No unknown peaks filled in
retroactively. Complete raw outputs and actual server source bytes retained once
on terminal; no silent rerun or criteria relaxation. Delegate long-job status
monitoring and wait dormant. Native CPU reservation still awaits explicit owner
availability and is not inferred from machine idleness or this protocol.
