# Prospective Falcon -> original geometry: initialization and one full-history step

9 October 2026. Goal ACTIVE/INCOMPLETE. First actual source-informed D256/L6,
n1152/full V65537 learner. This measures construction, a complete-history
learning step, residency and native representation. It does not qualify useful
chatbot quality or speed. Prior E32 direct bridge is retained without another
whole E32 update/output replay.

## Source and initialization, all approximations explicit

Pinned Falcon-H1-1.5B-Instruct revision80ebc50d7799a440b96c93bb6686a3924a09b0cb,
actual3.109GB BF16 source safetensors/config/tokenizer/chat template. Use cached
canonical embedding+head Gram basis P512, first256 columns only; do not rerun
eigendecomposition or source inference. Full source embed/head projected by P256,
baking embedding_multiplier5.656854249492381 and lm_head_multiplier.01953125.
Projected diagonal source final RMS gamma; each target site's input/FF gamma
uses arithmetic average of corresponding four source gamma vectors projected
by squared P coefficients. Norm factor1; this isotropic-coordinate approximation
does not equal source RMS in general. Preserve F32 organs/full tokenizer ID range.

Site l draws ALL4,608 FFN rows from source blocks4l..4l+3. Contiguous36 groups
of128 and strided36 groups of128 give two distinct partitions; four sign choices
(+,+),(+,-),(-,+),(-,-) per group. Ordering source block, partition, group, signs;
n=1152 at each site. Retain all6,912 actual maps/128 source row IDs/signs. Gate
projection bakes mlp_multiplier.4419417382415922; down bakes.13020833333333331.
Down masters additionally multiply n/16=72. The unweighted real-arithmetic sum
over sign quadrants gives each projected bilinear gu; summing both partitions
duplicates it twice. Uniform ALL-n normalized mass with scale72 yields1/8 the
sum over four source blocks, i.e. their averaged g/2 gate proxy. Actual source
SiLU(g)*u is different; |SiLU(g)-g/2|<=g^2/4. Actual learned deterministic top8
is not a uniform sample or unbiased estimator of the full sum. No source-function
equality is claimed after projection/ternary/AQ/normalization/selection/depth change.

Router rows are normalized signed gate-row-mean plus signed up-row-mean fingerprints;
bias0. Their usefulness is unproven, but maps are explicit and nonidentical grouped
coverage. Coefficient count679,477,248 matches source FFN count only. Native
effective bundle hashes detect exact duplicates; diversity is not useful capacity.

Core is explicitly FRESH original Mamba1/SWA, not transplanted Falcon Mamba2 state.
CPU generator seed1710, no old E32 checkpoint/Adam/RNG adoption. In_proj/qkv normal
std1/sqrt(input_width); causal conv identity at last tap/bias0; x_proj std.1/sqrt512;
dt_proj std1/sqrt16; dt inverse-softplus of loguniform[.001,.05]; A_log=log(1..96)
per512 channels; Dskip1; out_proj/o std.01/sqrt(input_width). Learn history/state
from complete source outputs. This initial loss must not be interpreted as a
successful source core transfer. Output and FFN initialization use actual donor
parameters. Source24-to6 composition is a learned substitution, not average equality.

## Streamed first-order adjoint and qualification

Source learner inherits qualified original scan/SWA/dReLU/AQ63/flat full-softmax/
stable lower-ID top8/mass-before-hidden-AQ63. CPU masters and SAME CPU quantizer
decisions/scales used in GPU forward and packed export; no blanket model.to(cuda).
Custom bank forward groups all selected ranks by expert once, computes exact
F32 integer sums/scales one expert at a time, scatters to unique pair slots and
adds ranks0..7 in order. Masters/gradients/Adam stay CPU; organs/router/head stay GPU.

Backward recomputes one expert with independent differentiable local leaves and
the same identity STE as original tensor learner, calls its first-order VJP,
writes each unique expert into ONE dense CPU gradient per bank. Input pair
gradients reduce in fixed rank order; normalized selected-mass gradient flows to
router. No atomics/index_add/E*max_count padding/full-bank GPU copy. This is a
first-order optimizer adjoint, not higher derivatives of rounding. Whole blocks
use nonreentrant checkpointing; complete SSM/SWA history is preserved.

Before source construction, compare NEW dispatch CPU and GPU versus the retained
original tensor bank at E32 on17 seed1711 Gaussian vectors and seed1712 output
cotangents. Compare all outputs and x/gate/up/down/routerW/routerBias VJPs;
normalized L2 difference<=1e-4 for each tensor (floor1e-8). Exact actual selected
IDs and mass differences<=1e-6. This is a necessary changed-adjoint check,
not another small-model update or native-loader sweep. Stop on failure.

## One new actual longest-FIT step and native comparison

Select longest FIT by (complete student_input_ids length,id) from immutable
adopted broad48 corpus: broad_fit_smol_magpie_ultra_022, entire1,507 IDs,
256 supervised positions1251..1506, V65537 teacher BF16 packet. No truncation/
history reset/teacher replay/RESERVED. Validate packet/input/position/label extents.
Freeze ledger and actual initial model/RNG before backward. Compute source-relative
whole-label KL using all65,537 logits per label, not greedy agreement alone.

One NEW update only: seed1713 CPU/CUDA RNG, F32, no AMP/TF32/dither/auxiliary loss,
Adam lr5e-5/betas(.9,.999)/eps1e-8/wd0/foreachFalse. Inspect every parameter gradient
and moment finite and every site's aggregate core/norm/router/bank gradient>0.
Global clipping uses aggregate F64 L2 norm across CPU/GPU tensors, coefficient
min(1,1/(norm+1e-6)). Preserve actual Adam1/model/RNG/whole input/routes/gradient
receipts in immediate durable checkpoint. Full dense CPU moments include dormant
rows; this is not a block-resident optimizer for10B/100B.

Export actual state E4BPv001 and run original packed executable on ALL1,507 IDs,
retaining ALL1,507*65,537 native logits and inside-forward6-site routes. A NEW
GPU no-grad complete-head output uses same entire history. Require every row
normalized RMS<=1e-4 (reference RMS floor1e-8), exact actual selected IDs, max
mass difference<=1e-6, finite outputs/unique IDs/normalized positive mass.
Report greedy differences separately; none can replace full-output comparison.
Native witnesses IDs0/31/32/1023/1024/1151/all6 sites/all gate/up/down rows must
equal CPU int64 dots of newly quantized masters:18,432 exact coordinates.
Compare before/after full-label KL and disagreement; one FIT update is not a
quality recovery gate. No fresh source inference or generated-chat/speed admission.

## Budget and durable failure boundary

Exact master count717,877,248/F32 bytes2,871,508,992; banks679,477,248/2,717,908,992B.
Dense bank masters/gradients/Adam10,871,635,968B CPU; total four-array bytes
11,486,035,968 across devices before workspace. Packed payload507,505,920B;
full native logits395,057,036B; initial/final snapshots about2.872/8.615GB.
Complete sequence touched union is observed, not assumed k8; one-expert device
residency excludes still-charged scan/checkpoint/head/router/gradient workspaces.

Family900s/final reserve120s/OS24GiB/GPU allocated10GiB/reserved11GiB/
namespace16GiB/log4MiB, one worker/no timing overlap. Host80GiB permits dense
prototype state; no T4 or10B/100B memory feasibility follows. Progress/guards
between expert batches, phases and durable writes. Hold worker/direct native
child through exit; reuse original packed executable, no new compiler.
Before launch freeze code/protocol, source/cached basis/corpus/selected teacher/
reference checkpoint/native artifacts/principal Torch CPU+CUDA runtime files/
foreign/original bytes. Pre-post checks. No T4 allocation or throughput forecast.

Stop on resource/integrity/nonfinite/adjoint gate failure. Retain first fault,
completed metadata, actual initialized/updated state and RNG if available; do not
repeat completed transformations/updates/native outputs solely for observation
timeout or metadata failure. A numeric native gate failure retains updated state
and complete outputs as evidence, with decision FAIL, not a relaxed threshold.
Passing proves this source-informed construction/update/native numerical scope.
Useful whole recovery/DEV+retention/fresh own-history chatbot and SAME artifact
>=50, useful large n/structured CPU IDs+mass/physical DRAM/family variants remain.
