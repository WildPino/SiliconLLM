# Current resumption after445: matrix-level common/private feasibility

**4 October 2026. Goal ACTIVE/INCOMPLETE.** The user resumed experimental work
following the [strategic review](STRATEGIC_REVIEW_20261004.md) and requested
regular reassessment of the whole problem in algebra, geometry and information.
The review remains a historical document-only snapshot. This file is the current
operational plan, superseding its earlier441/442-only instructions.

## Goal and invariant scientific boundaries

Convert pretrained knowledge into compact reusable core plus useful selectively
consulted functions in benchmarks/phase60/engine.c, preserving original-relative
prediction/generation/task quality AND>=50 acceptedbatch1IDs/s on SAME artifact
(100 stretch). Multiple families/scales, actual~10B/~100B as resources allow.
User priority: useful n grows with RAM; CPU LUT/routing/physicalDRAM/quality stay
viable as choices grow. Stored labels/functions, consulted IDs, distinct useful
functions and bytes actually read are different quantities.

Original Switch128/256 artifacts remain qualified in their bounded original
contexts.128 ordinary96.56/prose55.98IDs/s;256 ordinary63.53/prose34.78. Warm
S29/T14 infilling, source-specific worker setups, no cold/long-context/global
rate promise. No new complete artifact or useful384/10x/~100B model exists.
Routine Graphify disabled; frozen donor-adaptation Giga evidence remains reusable.

## New evidence from442..445

- **442**: first inclusive150sec admission failure retained before mapping/forward,
not a causal outcome.443 repairs only timing phases; all336 original complete
heads byte-exact. Source128 ID+1 mean self-KL.176457653/41argmax changes, fixed0
.078196180/29, removal.032704093/21. Original identity matters in this local pilot.

- **441**: fixed expert0 retaining99.66%of440's small foreign benefit does not show
source functions are interchangeable. The foreign chain input/core/probability/
readout/mask changed jointly; no identified readout-only culprit or attenuation
factor between unlike estimands.440 remains CLOSED.

- **444**: source-unique functional contrasts: Fbar(x)=mean_e F_e(x), d_e=F_e-Fbar.
One development-fit output basis32/64/128/256 with oracle mean/coefficients fails
validation energy/native KL. Rank256 val79.10%/KL.04539. Two development position0
anchors contribute88.44%of private norm, making rank32 weighteddev96.52% misleading
as a statement about every input. No retrospective gate change.

- **445**: equal-relative-anchor weighting also fails, rank256 macrodev90.50%/val78.52%.
Separate local oracle95%rank75..87 at validation; rank96 local energy96.86..98.25%.
No new head or posterior check there.128 centered functions have rank<=127 at
one fixed input, an algebraic population bound, not an inference algorithm.

All respective apparatus gates pass, first failures retained, all sessions
terminal. Records: [443](METH_443_SWITCH_NATIVE_CAUSALITY_RESULT_20261004.md),
[444](METH_444_SWITCH_FUNCTION_CONTRAST_RESULT_20261004.md),
[445](METH_445_SWITCH_RELATIVE_CONTRAST_RESULT_20261004.md).

## Reassessment: which mathematical problem remains?

The observed function differences cannot be represented adequately by these
small fixed shared OUTPUT bases estimated from18inputs. Weighting alone is not
a repair. This does not say each expert needs75latent dimensions: many private
rank1 matrices can collectively span many output directions. Each expert may
have its OWN low-rank factors; their union need not be low rank. Input-dependent
ReLU gates also make functional differences more complex than weight differences.

An input-specific basis fitted offline to all128 functions is unavailable at
runtime without paying enumeration cost. Do not propose it as an n-independent
LUT or cheap router. Development sampling, private matrix structure and actual
cost must be tested separately. Avoid another global-output/readout fit now.

## Immediate next NEW pilot, prepare/freeze before calculations

Use ONE donor/source128, ONE bank11 first, retaining the actual source core.
Investigate MATRIX-level common plus PRIVATE factors, not another foreign-core
map or shared output basis. New rank/exposure/precision/cost choices must have a
frozen controller/protocol before any matching, SVD, fitting or model forward.
446 controller/math/[protocol](METH_446_SWITCH_PRIVATE_DELTA_PROTOCOL_20261004.md)
are frozen63a5f04 before first import; no numerical outcome yet. Fixed source128
reference0/targets1,64,127/all1344source inputs; ranks16/32/64/96/128, no rescaling,
identity versus coupled unit-WI assignment, private versus full-matrix factors.
First original native1344FFN replays and all matched permutations byte-exact,
then F64-shadow qualification/compression geometry. Admission300s/numerical600s/
total900s,3GiB/384MiB,CPU0/Torch1/BLAS1. First failure retained, no rerun.
Exact one-time command is in the protocol. No new native export/head/rate claim.

1. Bind original payload/tensor/native-capture provenance. Predeclare a small
   fixed expert subset and full development/validation input populations; do not
   choose IDs from utility. The18-anchor study is inadequate evidence for a full
   source population; existing1008dev/336val captured source inputs are available.
2. Treat ReLU neuron symmetries explicitly: positive diagonal D/permutation P
   gives WI'=DP WI, WO'=WO P^T D^-1 in real algebra. Coupled transformations
   preserve the real function, not automatically native I8/A16/F32 bytes. Compare
   identity alignment to ONE prospectively specified affordable matching recipe;
   do not average arbitrary internal neuron coordinates or claim an optimal
   alignment from a heuristic. Gate/down biases and zero rows need explicit handling.
3. For an admissible aligned gauge, define WI,e=WI,0+DeltaWI,e and
   WO,e=WO,0+DeltaWO,e. Evaluate compressibility of PRIVATE deltas against full
   matrices at the same coefficient/precision budget and on actual inputs.
   Preserve a_e=ReLU(WI,0 x+DeltaWI,e x); do not replace this by a sum of separately
   activated common/private networks. Private bases may differ by expert.
4. Measure native original-relative feature/posterior errors and identity effects
   with matched full/base-only/private-removal/permutation controls. An optimistic
   rank bound must specify its metric and sampled population. Local input-aware
   SVD is not a global KL/nonlinear-compression bound. Do not spend long training
   if representation or active cost is already inadequate.
5. Charge ALL shared/private operators, lookup/index construction, normalization,
   routing, active bytes and precision conversion. A nominal parameter saving
   does not certify native latency or actual DRAM. Screen the exact proposed
   format before a broad fit; budget/stop/disk/memory limits fixed in the protocol.

### Exact common down reuse and the top1 cost trap

For aligned shared bases and private nonlinear activations a_e, real algebra:

sum_e p_e WO,e a_e = WO,0(sum_e p_e a_e) + sum_e p_e DeltaWO,e a_e.

The common down matrix is applied ONCE even though activations differ. A common
WI,0 x can likewise be reused before each private correction/nonlinearity.
Quantization and changed sum order still need fresh quality checks.

Let input width D, private hidden width M, active count k, equal factor rank r
on each of two projections. Dense arithmetic coefficient counts are:

C_original=2kDM;
C_base_delta=2DM+2kr(D+M).

For Switch k=1, a FULL shared base plus private factors ADDS arithmetic; possible
storage savings do not make it faster. A cheaper base, acceptable measured
margin or a different representation is necessary. For larger k the common
term can be amortized, but all private nonlinearities and weighted sums remain.
These are coefficient counts, not measured time bounds.

Bank storage before scales/index overhead is approximately:

S_base_delta=2DM*b_base+n*2r(D+M)*b_private.

Against source I8 storage n*2DM, D768/M3072 and F32 private factors require
r<153.6 for asymptotic private-storage saving, before the shared base/metadata.
Different factor precision changes this threshold and requires its OWN numerical
qualification. Output rank75..87 does not determine this matrix-delta rank.

## Routing/n and whole-artifact gates remain independent

Switch selection uses both argmax score and full normalization. Adding candidates
changes Z=sum exp(s), thus the old selected amplitude, even without a new winner.
Flat hierarchy rewriting alone saves no dot products. Certified adaptive bounds
must charge candidate search/refinement/fallback/normalization; worst case can
still be O(n). Existing393 scores may support a separate bounded diagnosis.

After a promising useful-function representation, require causal usefulness of
more REAL functions at two actual n increments, routing/LUT scaling and hardware
DRAM evidence, full original-relative prediction/generation/task quality and
SAMEartifact>=50 acceptedIDs/s. Another family/~100B needs real values/reference
and measured geometry. RAM-only storage accounting or repeated IDs never suffice.

## Controls and closed routes

Freeze scientific source/controller/protocol BEFORE observations. CPU-only bounded
pilot; no concurrent modeljob/native timing, no newGPU/T4/network/corpus unless a
new authorized plan justifies it. T4 requires prior reason/budget/stop communication.
Keep first failures, never overwrite or rerun old experiments. Preserve unrelated
work, original374/389 binaries, current engine and exact publisher daemons.

403/404/406/414/416 bridges;426/431/435/440 fits;444 absolute-output PCA and445
weighting-only correction; specified Granite/Ling/Giga full-width/LUT formats stay
CLOSED. Matrix-level private bases/nonlinearities are a different NEW hypothesis,
not a reinterpretation of those failures. Reassess representation, information,
active cost and useful-n evidence before expanding tests or training.
