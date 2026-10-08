# Complete original-engine target learner: pre-observation protocol

8 October 2026. NEW pilot, not a repeat of Qwen fitting, Tiny scan packets or the
completed Falcon16 selection cases. Freeze this protocol and code before any
new weight-basis, target forward, source calibration generation or learning.

## Decision, reused evidence and scope

Uncertainty: can the complete compact recurrent/conditional/ternary target be
initialized from the qualified Falcon1.5B source, differentiated and actually
updated in local12GB GPU memory? Does one small whole-output pilot recover
output information sufficiently to justify a separately budgeted calibration?

Reuse pinned acquired source80ebc50d7799a440b96c93bb6686a3924a09b0cb and its
producer-SHA/header/interaction contract;14/16 selection evidence is complete.
Reuse original engine AQ63/ternary-pair matrix algebra, F32 sensitive organs,
qualified source recurrence bridge and the48-head target budget. No previous
generation, native packet, fit, audit or source acquisition is replayed.

This is a prerequisite/feasibility pilot, NOT final chatbot quality, large-n
capacity, native parity or>=50 evidence. A loss reduction on six short cases
cannot demonstrate retained donor intelligence. No T4 here.

## Target and transformation, explicit approximations

F32 targetD512/L12/V65537, untied embedding/readout,10 SSM768 with48 heads*16,
N256/group1/conv4/gate-before-RMS, two4-head*128 SWA sites5/11/window128,
72 distinct source-derived banks per layer, selected8, h128. Full signed
SwiGLU SiLU(gate)*up. All organs, master expert weights, row scales and router
are learnable. Source fallback SSD chunk16 and whole-block nonreentrant
checkpointing. TF32 disabled; no optional SSM/hub kernels.

ONE shared orthogonal2048->512 basis from equal-trace embedding/readout Grams,
top512 eigenvectors descending, max-coordinate-positive sign convention.
Record trace retention and orthogonality<=1e-4, NOT a capacity gate. Bake source
embedding5.65685/head.01953125 and projection MuP factors into F32 weights.
Projected norm gamma is diag(P^T diag(gamma) P)*sqrt(2048/512): an explicit
energy assumption and trainable approximation, not exact RMS equivalence.

Target blockt takes core/norms from source2t+1; its banks contain contiguous
128-feature groups from source2t and2t+1,36 each. Retaining aggregate feature
slot count loses width/composition and is NOT preserved knowledge. SSM retains
all48 A/delta-bias/D entries and every fourth channel within each64-channel head;
all256 B/C state coordinates and selected convolution rows are preserved.
SWA retains source query heads0/2/4/6, corresponding KV0/0/1/1, source RoPE
dimension128/theta1e11 and key scaling. Depth reduction, missing parallel branches,
SWA truncation, projection and output composition need joint learning.

Each gate/up/down row gets eight deterministic ternary least-squares scale
iterations. Gate multiplier applies ONLY to gate; down multiplier to down.
Forward uses AQ63 absmax scale, reciprocal multiply, round-to-even/clip,
integer codes -1/0/1 and F32 exact integer-dot range. Signed intermediate is
quantized anew for down. STE gives master-weight and scale gradients, activation
gradient through dequantized proxy. Clamp learned row scales>=1e-8 after updates.
This algebra is compatible with byte-pair LUT products but native packed/C
rounding equivalence still requires actual verification.

Temporary flat router computes72 full scores, stable descending ties prefer
lower IDs, top8 softmax renormalizes selected mass. Not the proposed tree or
large-n routing solution. Actual parameters254,932,736; F32+gradient+Adam array
floor4,078,923,776 bytes, moment arrays2,039,461,888 bytes. Active matrix products
69,632,512 including flat routing; scalar scan/activation/RMS and context costs
excluded. Distinct source rows do not establish distinct useful functions.

## New data and fixed update/evaluation recipe

`chatbot_hybrid_pilot_cases_v1.json`: four FIT/two DEV cases. Separate from16
selection cases and excluded from future final quality. Original template/IDs,
both producer EOS, max8 new IDs/context<=128, greedy BF16 donor. Save EACH
complete generated prefix and full-vocabulary F32 generation scores immediately.
Student sees donor history: prompt+generated[:-1], evaluates positions at which
donor chose the generated IDs. Readout computed only at supervised positions.
Full-vocabulary mean forward KL(source||target), temperature1. No gold task
accuracy threshold or student own-history claim. DEV is never optimized.

Initial evaluation ALL6; AdamW lr5e-5/weight_decay0/foreachFalse, eight updates,
fixed FIT order repeated twice, batch1, global gradient clip1. Each update checks
finite output/loss/ALL gradients/parameters, positive per-layer core/bank/norm
gradients, actual readout change and actual full Adam moment bytes. Save immutable
step metrics and keep a CPU snapshot at each completed boundary for fault recovery.
Final evaluation ALL6; retain initial component checkpoints and final full learner
plus optimizer. Initial projection checkpoints and supervision are reusable.

Numerical gate: complete finite8 updates, ALL12 active gradient groups and actual
readout change. Recovery gates: final FIT position-weighted KL strictly lower than
initial; final DEV KL<=1.02*initial. Report all case KL/ID disagreement without
posthoc threshold changes. `PILOT_RECOVERY_PASS` requires all three; failure closes
this eight-update recipe only. Even a pass needs broader calibration and whole
fresh quality before deployment, not a longer unbudgeted run.

## Bound cost, provenance and stop/recovery

Local expected seconds to tens of minutes; hard1800s FAMILY (input SHA/serialization
included), family OS16GiB, GPU allocated11GiB/reserved11.5GiB, output5GiB,
worker log4MiB. Expected initial1.02GB+final optimizer learner3.06GB+labels/basis.
CPU recovery snapshots add real RAM/copy time and are charged. Actual resource
peaks measured, not inferred from array counts. No simultaneous experiments,
no worker subprocess; preserve exact unrelated publisher exception.

Shared launcher holds the Windows worker handle through actual exit, samples
peak working set including completed final serialization, verifies bound bytes
before/after, returns terminal output extents/hashes. Binding covers source,
code/cases/protocol, Python and selected Transformers/Torch Python files plus
foreign tracked preservation hashes. Exact isolated versions/paths checked;
NOT a full DLL-tree certificate. No global environment/package modifications.

Stop on finite/shape/gradient/resource/interaction failure, retain first failure,
all packets/shards/metrics and completed-boundary recovery snapshot. Any numbered
repair reuses completed packets/basis/initial shards and completed update checkpoint;
no completed source calls/initial observations/updates are repeated. No automatic
repair with a changed scientific loss/geometry/threshold. Incomplete output is not
a complete result. Independent saved-only numerical/state/provenance audit should
check initial/final metrics and optimizer/parameter/state consistency without
replaying the target experiment; native C parity remains a subsequent gate.

## Subsequent quality gates, required before a long adaptation launch

Prepare a substantially broader FIT/DEV calibration with balanced domains,
longer contexts and donor supervision on both common and target own prefixes.
Keep all six pilot and16 selection cases excluded from fresh final gates.
Before observing fresh data, freeze donor-relative own-history dialogue/task,
per-domain quality floors and an explicit tolerance; evaluate SAME final packed
C artifact and>=50 accepted batch1 IDs/s, physical DRAM, LUT IDs/mass and
useful-n/RAM. These gates are required, not results or thresholds claimed here.
A T4 launch needs measured FP16-compatible training feasibility/throughput,
explicit reason/token-time budget/stops communicated before allocation.
