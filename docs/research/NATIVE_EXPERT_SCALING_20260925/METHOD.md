# Pretrained-to-native conditional-capacity method (work in progress)

**Status: research procedure, not a validated conversion recipe.** This file
describes what can be reproduced now, what has failed, and what still needs an
experiment. The target is one identifiable artifact derived from a pretrained
LLM, retaining useful held-out, generative and task quality against that donor,
running in the project C engine at ≥50 accepted batch-1 tokens/s on a declared
CPU and context. The same artifact must pass quality and rate. A second donor
family or scale must test which steps transfer. More stored parameters alone
are not evidence of transferred capability.

## Current reproducible complete candidate

The saved candidate is [276 diagnostic I16/private128](METH_276_DIAGNOSTIC_COMPLETE_CORE_RESULT_20261002.md),
1,329,447,260bytes/same725 fields. Its own versioned loader and all6144
composition controls pass;277/280 prediction,281 generation health and282
anonymous semantics pass.283 full PIQA now passes. [284 actual phase60 archive/operators](METH_284_NATIVE_ARCHIVE_RESULT_20261002.md)
now pass all6144 controls. [285 whole native prefix smoke](METH_285_WHOLE_PREFIX_RESULT_20261002.md)
executes complete prefixes/cache but FAILS the unchanged5% ablation logit
guard (5.4224%). All96 top1 choices and12 cache controls pass; the original numeric5% guard remains failed.286/287 identify tiny local
rounding errors with nonlinear propagated amplification.288 norm64 and289
FP32-residual variants fail and are not adopted.290 reference is stable;
291 actual-input hybrid is byte exact to original CPU but no one-group GPU
restoration meets its fixed rescue indicator.

[292](METH_292_NATIVE_PRIMARY_EVALUATION_POLICY_20261002.md) explicitly
revises evaluation order/estimand BEFORE selecting new sources: measure
actual ORIGINAL-profile native donor-relative quality directly, preserving
all numerical failures and unchanged final quality/rate criteria. This is
not a numerical repair.292 selects24 untouched sources with3224 exclusions;
293 source-only answerability passes before inference.294 real phase60
scoring entry preserves original forward arithmetic with relative cache/
absolute RoPE window positions; consumed-data bridge passes actual byte
closure/top1/NLL oracle. [295 actual native prediction](METH_295_NATIVE_PRIMARY_PREDICTION_RESULT_20261002.md)
now passes all pooled/category quality and apparatus gates on24 untouched
sources: BPB1.23886670 versus donor1.24367193/E1280 1.23842220 and
donor-top1 agreement96.0600%. The process-only repair reuses exact saved
GPU controls after cleanup guard failure; native CPU29.422minutes is assay
cost, not accepted rate. [296 actual native generation](METH_296_NATIVE_GENERATION_RESULT_20261002.md)
passes all health/cache/state-capture gates: all prompt first heads/choices
exactly295, nine real generated-state cache/prefix byte checks, three fault
controls, native EOS23/early0/repeat0. All normalized native prompt/generated
states are retained for diagnostic CPU work. [297 strict anonymous semantics](METH_297_NATIVE_SEMANTIC_RESULT_20261002.md)
FAILS unsupported claims versus E1280 (41>40), while other five gates pass.
All72 findings committed before map access; no post-count adjustment.
The fixed ORIGINAL native execution profile is closed for promotion:
skip its PIQA/K64/rate promotion work. Archive stays diagnostic/unpromoted.
GPU quality passes and native295/296 scoped passes remain valid but cannot
override this semantic stop. A changed candidate needs a prospective new
quality cohort. Useful large-n and cross-family compact transfer remain open.

The case-specific [METH-259 saved candidate](METH_259_UNIQUE_BANK_CORE_RESULT_20261002.md)
is available:1,321,032,556bytes,725 tensors, full24-layer Qwen0.5B source
core plus30,556 actually unique effective conditional parameter functions,
1243..1280/layer, original1280 route labels with explicit alias maps.
Input source FFNs use rowQ8/LUT,32 BF16 outliers,rank32 source residual and
32 private source BF16 input rows; exact tied BF16 embedding/head and
original non-FFN organs are retained. Existing learned centered E1280
functions/routing are conserved, not newly invented donor capacity.

| Available step | Tool | Evidence and limit |
| --- | --- | --- |
| Build actual core/dictionaries/maps | `meth259_unique_bank_core_export.py` | All725 fields exact,6144 original component/conditional vectors bitwise equal |
| Load complete archived configuration/weights | `meth259_unique_bank_core_export.load_stored(path,device,expected_sha)` | All executed weights from archive, no donor/checkpoint fallback;9 archived code hashes Git-checkout stable |
| Full-model development | [METH-260](METH_260_COMPLETE_CORE_DEVELOPMENT_RESULT_20261002.md) | All frozen BPB/top1/K64 gates pass on consumed24 sources; no new independent source proof |
| Native fixed-source FFN | METH-253 | 9.318ms/24-layer component; full native model/conditional alias lookup unqualified |

Exporter requires original pinned211 core/non-FFN/head proposal,252 source
fixture, parent/child training artifacts and125 source-state vectors for
control. Full saved loader needs the archive and implementation dependencies;
export and quality commands/resource limits are in the linked protocols.
The [METH-261 manifest](METH_261_COMPLETE_CORE_FRESH_MANIFEST_RESULT_20261002.md)
now freezes24 independent sources with3176 old-source exclusions and the
original fragment guards, without any new model scoring. [METH-262](METH_262_SOURCE_ANSWERABILITY_RESULT_20261002.md)
passes all24 source-only answerability decisions before inference.
[METH-263](METH_263_COMPLETE_CORE_FRESH_PREDICTION_PROTOCOL_20261002.md)
passes [independent full-head prediction](METH_263_COMPLETE_CORE_FRESH_PREDICTION_RESULT_20261002.md)
on the same artifact: BPB-.000110010 versus BF16 E1280,-.005313449 versus
donor,top1+.446429 points versus BF16 E1280; all frozen gates pass.
[METH-264](METH_264_CACHED_REFERENCE_STOP_20261002.md) stops on original
BF16 donor cached/full choice differences before candidate generation.
[METH-265](METH_265_FULL_PREFIX_GENERATION_RESULT_20261002.md) passes
all full-prefix generation-health/K64 gates. Frozen266 PIQA regression
and267 anonymous semantic review follow. [METH-267](METH_267_ANONYMOUS_SEMANTIC_RESULT_20261002.md)
now fails severe-error/missing-detail gates; fixed259 native promotion is
closed. [METH-266](METH_266_COMPLETE_CORE_PIQA_RESULT_20261002.md) passes
PIQA regression,1290/1838 versus1291 for both controls, but cannot override
the semantic failure. A changed core/route mechanism needs new checks.
Frozen268 traces consumed prefixes and injects reference routes only as
an oracle diagnosis. This is nondeployable and cannot reopen259 promotion.
[Its result](METH_268_GENERATION_ROUTE_REPLAY_RESULT_20261002.md) recovers
15/40 choice differences but leaves25 and increases mean logit error4.3858%.
Actual source-BF16 intermediate/LUT/core error and induced routing both
need qualification; same-input route/maps remain exact.
[METH-269](METH_269_FFN_BF16_NODE_RESULT_20261002.md) separates these on
6144 consumed states: BF16 nodes alone worsen mean error2.22%,oracle exact
source features improve it80.40%. [METH-270](METH_270_PRIVATE_SOURCE_ROWS_RESULT_20261002.md)
fixed128 coefficient-ranked input rows improve mean error14.67%,missing the
prospective15% gate. The recipe stops before native timing/full assembly.
All old32 rows and nonprivate fields remain exact. [METH-271](METH_271_OUTPUT_AWARE_ROWS_RESULT_20261002.md)
activation/output-aware selection at fixed128 improves mean/reserve22.42%/
21.87% and passes384 native numerical vectors, but11.398ms fails the10ms
component budget. No full archive is licensed. Any changed execution kernel
requires its own prospective numeric/cost qualification at unchanged fields.
[METH-272](METH_272_FUSED_INPUT_RESULT_20261002.md) shared input-dot fusion
retains bitwise outputs but fails both cost and paired speedup gates;
10.991ms versus10.865ms unchanged control. [METH-273](METH_273_PRIVATE128_PHASE_RESULT_20261002.md)
actual private128 phase diagnosis retains bitwise values and identifies
shared/concurrent down-right92.14% versus private/team6.02%,including added
instrumentation overhead.[METH-274](METH_274_I16_SHARED_INPUT_RESULT_20261002.md)
I16 input/shared Q8 integer dots pass all6144 actual CPU source/reserve and
GPU/native controls (median4.67e-7/max1.04e-6) and59,768,832 scalar-I64/AVX
dot checks,but10.551ms fails10ms.[METH-275](METH_275_PAIRED_I16_LAYOUT_RESULT_20261002.md)
paired/aligned layout preserves all6144 CPU outputs yet10.645ms/3.54% paired
gain miss both10ms/5% gates. Both fixed kernels stay stopped.276 now explicitly
revises assembly order for an unpromoted diagnostic complete I16/private128
archive: composition and actual whole budget are unknown,and the component
allocation is not the user's50tok/s requirement. No old gate/result changes;
full fresh quality/actual same-artifact rate remain mandatory for promotion.
[METH-276](METH_276_DIAGNOSTIC_COMPLETE_CORE_RESULT_20261002.md) now saves
a1,329,447,260byte complete diagnostic archive and owns a distinct versioned
loader. Only72 private fields change,all653 other fields byte exact259;
all725 readback/loader and6144 FP32 equation/BF16 return/route/alias/isolated
conditional controls pass. Extra payload8,262,144bytes,no new capacity.
134.422s/endRSS5.87GB/peakGPU2.86GB;original zero-score CUDA launch failure
preserved,environment-only repair. `meth276_diagnostic_complete_core.load_stored`
needs the saved artifact and pinned project implementations,not donor or
checkpoint weights. This establishes archive composition,not whole quality
or rate. [277 consumed development](METH_277_COMPLETE_I16_DEVELOPMENT_RESULT_20261002.md)
passes all existing BPB/top1/K64 gates with exact194/260 controls,118.125s.
Versus259,BPB improves.000119123 but top1 agreement drops.244771points;
do not infer semantic repair from source-error reduction or consumed BPB.
278 now freezes24 new sources with3200 exclusions,204.359s CPU;source pool
unchanged,no donor-pretraining exclusion claim.279 source-only answerability
now passes all24 and annotations are committed before outputs.280 new
complete prediction passes all pooled/category/BPB/top1/K64 gates,112.156s;
BPB1.17331320 versus donor1.17828037/E1280 1.17323483,donor-top1 96.5006%.
The E1280 BPB bootstrap interval includes0,no significant gain claim.
New source IDs are now consumed;281 new-source full-prefix generation now
passes all72 health/generatedK64 controls,602.594s,candidateEOS22/repeat1,
same as donor (E1280EOS23/repeat1). [282 anonymous semantics](METH_282_COMPLETE_I16_SEMANTIC_RESULT_20261002.md)
now passes unchanged strict gates:unsupported37/severe17/missing0 versus
42/21/0 donor and42/20/1 E1280. All72 findings committed before unblinding;
finite same-agent review,not independent human/broad quality proof.
[283 full PIQA](METH_283_COMPLETE_I16_PIQA_RESULT_20261002.md) now passes,
1289/1838 versus1291 for both controls,-.108814points;paired lower
-.598477/-.544070points,all four frozen gates pass. All newly recomputed
control choices/option NLLs are exactly266.1101.906s/endRSS3.43GB/GPU3.14GB.
Repeated regression,not independent task proof. Full native integration/rate
remains open;diagnostic status and all previous fixed stops remain unchanged.
This reference is not production decoding. Actual native
whole-model and route/LUT/real-DRAM checks and same-artifact>=50 accepted
tok/s remain mandatory. This is specific to the small dense donor; the
active source feature count still grows with dense donor width/depth.
Sparse GigaChat/10B/100B/second-family variants are not automatically qualified.

## 1. Source-family variants and route selection

A sparse-source variant is **GigaChat 3.1 Lightning 10B-A1.8B**, source revision
`189fff27a1dee68473960c3d5bca53e0e07a3191`. Its full HF archive is about11.480B
including MTP;the actual414-tensor base GGUF has10,672,535,616 elements
(BF16 matrices/F32 controls),26 layers,64 routed experts/top-4,one shared expert,
MLA, and a dense first FFN. Source shards, BF16 GGUF and Q4_K_M GGUF are
locally bound in the [source-binding record](../donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md).
Ordinary180 and diagonal-weighted181 rank192 representation screens fail.
[298 full routed-input covariance](METH_298_GIGACHAT_FULL_COVARIANCE_PROTOCOL_20261002.md)
was prospectively frozen: collect actual source-BF16 routed inputs on the
same distinct calibration domains; fit uncentered FP64 rank192 output
directions and compare real test-output reconstruction to ordinary and
diagonal factors. [Its result](METH_298_GIGACHAT_FULL_COVARIANCE_RESULT_20261002.md)
FAILS all four gates:test median66.430%/min37.820%;even the diagnostic
test-domain optimum has median87.454%,so calibration shift alone cannot
rescue rank192. Verified real routed input captures are now reusable;
no direct factor export or rank retune is licensed. This is not model
quality,useful-n or native-rate evidence. Reused source graph/library capture
does not resume the generic donor-port line. Even hypothetical int8 routed
factors leave822.29MB/token if other Q4 organs remain; MLA/head/shared
treatment and complete cost gates are required before full export.
[299 nonlinear block omission](METH_299_GIGACHAT_NONLINEAR_BLOCK_RESULT_20261002.md)
also fails all24 gates:half-width original32-channel blocks leave51.787%
median relative output L2 on Cyrillic,despite all full-output controls.
The 298 captures remain usable for separately frozen mechanism/calibration
work;they are consumed domains,not fresh full-model quality. A
[full-feature shared-vector-LUT hypothesis](FULL_FEATURE_VECTOR_LUT_PROPOSAL_20261002.md)
would preserve original channels and encode pairs of source coefficients
with a shared small palette. Its table construction can be independent of
macro-expert count,but row gathers remain large and down inputs differ.
[300 complete-organ cost preflight](METH_300_GIGACHAT_VECTOR_LUT_COST_RESULT_20261003.md)
rejects this fixed two-coefficient U8 format BEFORE palette training:
all414 actual source/Q4 descriptors reconcile;complete addressed weight
829.378MB/token,optimistic common-palette829.150MB,versus560MB allotment.
Full construction writes494.669MB/token and812,810,240 lookups address
3.251GB of logical FP32 table values. These are logical counts,not physical
DRAM or native latency. Even zero-head/zero-palette leaves729.854MB.
The source-derived tool explicitly prices32 different MLA K-B/V-B head
inputs and four different routed down inputs;no implicit table reuse.
Useful large-n still needs selective routing:flat F32 router reads rise
9.8304->98.304MB/token for the analytical64->640 topology,though selected
codes/scales and query construction stay fixed. Only64 source experts are
real;no extra capacity was trained/instantiated.
[301 four-coefficient U8](METH_301_VECTOR4_LUT_PREFLIGHT_RESULT_20261003.md)
fits descriptor423.625MB/token,but real source-shaped synthetic phase60
operators FAIL14ms:stable83.608/82.278/81.626ms medians;all309 matrices,
16,968 exact scalar rows/18,568 decoded FP64 rows/negative controls pass.
No donor coefficients or quality are encoded;n640/training skipped.
[302 same-math attribution](METH_302_VECTOR4_COST_PROFILE_RESULT_20261003.md)
has all30 output/route hashes exact301;matvec71.137ms(80.139%),tables16.650ms,
MLA combined41.439ms>routed26.541ms. Table-only or expert-only tuning does
not license a complete rescue at unchanged other measured costs.
The [joint compact-core proposal](GIGACHAT_JOINT_COMPACT_CORE_PROPOSAL_20261003.md)
is UNTRAINED:smaller attention,ten nonlinear128-wide children per source
parent(select one),640 labels/layer retaining original9.437B routed coefficient
capacity. [303 actual I8/full-Q6 native cost](METH_303_COMPACT_I8_PREFLIGHT_RESULT_20261003.md)
passes7,176 exact integer/scaled rows,97,194 stored bank edges,real Q6 decoding
and25-layer macro/child controls but FAILS14ms:31.268–31.936ms medians,
repeatability1.0214/9.953GB peakRSS. No teacher collection/training for this
unchanged implementation. Attribute unchanged math first, then select/freeze
a changed cost path. Child exposure/utility and independent composed quality
still require their own audit. No student/artifact/rate or useful new n yet.
[304 unchanged-math attribution](METH_304_COMPACT_I8_PROFILE_RESULT_20261003.md)
retains all30 exact303 outputs/routes and complete selftests. Profile-build
projection20.308ms/90.478%,MLA8.531ms/head4.658ms are diagnostic only;its
21.846–22.477ms medians still fail14ms and do not replace303's31ms. Different
compiled-run timing causes remain unisolated. [305 exact biased-byte/two-part I8](METH_305_BIASED_I8_PREFLIGHT_RESULT_20261003.md)
now reproduces every303 output/control but FAILS14ms at22ms. [306 packedI4/
tile4/serial tiny-query](METH_306_PACKED_I4_PREFLIGHT_RESULT_20261003.md) retains
all coefficient capacity, changes precision/compute scheduling, and passes
packed format/row/head/router controls;310.572MB scenario/5.127GB RSS but
PASSIVE20–22ms FAIL14ms. Those controls prove own arithmetic, not fidelity of
a learned donor/student conversion atI4. No teacher weights exist yet.
[307 unchanged-binary wait profile](METH_307_OPENMP_WAIT_POLICY_RESULT_20261003.md)
retains all90 original306 outputs/selftests, but ACTIVE14.780/11.764/11.320ms
fails firstmedian14ms/repeatability;PASSIVE A/C drift also fails. No post-count
subsets/gate changes or quality training. [308/309 binding investigation](METH_309_WINDOWS_AFFINITY_RESULT_20261003.md)
preserves308 startup failure;309 proves singleton physical-core masks but
fails variation/all14ms gates. Stop affinity tuning; no trained candidate
is licensed. A separate regional-output subspace diagnostic can reuse298
captures to test the nonlinear ten-child representation before any student
investment. It is a mathematical output-projection bound, not a native
model or learned child realization. Cost and source quality both remain
open; a passing local bound cannot qualify the failed CPU profile.
[310 real nonlinear regional subspace](METH_310_GIGACHAT_REGIONAL_SUBSPACE_RESULT_20261003.md)
now closes the fixed PCA16/spherical10/stand-alone rank128 parent recipe:
all four gates FAIL, pooled test error median81.607%, poor child exposure.
Its command reconstructs full source outputs from existing actual captured
post-SwiGLU states, checks coordinate/source closure and fits only input
regions/output spaces on the fit split. These optimistic coefficients are
not learned executable children. A [shared256 plus regional128 residual
bound](GIGACHAT_SHARED_REGIONAL_BOUND_PROPOSAL_20261003.md) is still proposed:
test the shared core's additional output space before rejecting that full
geometry. Do not waive known exposure/cost failures or infer composed quality.
[311 shared output-space/residual bound](METH_311_GIGACHAT_SHARED_REGIONAL_BOUND_RESULT_20261003.md)
also fails all gates despite energy improvements; exact independent-parent
recipe closed. Next reconstruct the COMPLETE routed-plus-shared FFN target
on existing actual captured inputs, with source routing and nonlinear
arithmetic controls. Individual-parent loss is not automatically mixture
loss; this change requires its own local and composed quality checks.
[314 complete source FFN targets](METH_314_GIGACHAT_MIXTURE_TARGETS_RESULT_20261003.md)
now provides a reproducible missing input/output step: deduplicate actual
captured normalized source inputs, replay original GGML routing, emulate
original BF16 activation packing, construct four parent+shared full targets
and export six bit-exact/hash-bound NPZs (1.609GB/37,381 states,69.5sCPU).
312/313 apparatus failures/partial datasets are preserved. Source post-SwiGLU
closure<=3.153e-7 and every original router SLOT pass. Complete mathematical
FFN targets do not assert captured native full-MoE bit parity or compact quality.
Rare/anchor-conditioned coverage remains insufficient for ten useful children
per all64 parents. [315 partial wall stop](METH_315_GIGACHAT_COMPLETE_MIXTURE_BOUND_FAILURE_20261003.md)
is preserved; [316 equivalent conditioned projection](METH_316_GIGACHAT_COMPLETE_MIXTURE_BOUND_RESULT_20261003.md)
completes all37,381 states with exact315 fitted-field hashes and192 fixed-index
SVD comparisons(max2.063e-15 relative difference). All four scientific gates
FAIL: complete-mixture TEST medians53.790%/66.767%/21.560%; even joint-span
oracle medians46.364%/59.084%/18.114%. Fixed shared256/regional128 fields are
closed before coefficient training/export. Poor exposure remains failed.
This local true-coefficient bound does not reject jointly learned fields or
all nonlinear conditional architectures.

[317 full-width additive-weight accounting](METH_317_GIGACHAT_ADDITIVE_DESCRIPTOR_RESULT_20261003.md)
provides a NEW representation prerequisite only: preserve original widths,
two U8 indices/eight coefficients into two256x8 I8 books, F32 row scale,
S7 activation proposal, originalQ6 head/F32 router/norm/BF16 embedding.
Fresh source/header geometry controls and542.987MB addressed/token accounting
pass560MB yardstick. No books are trained/exported.318 now implements a source-sized native kernel;319 closes its unchanged cost profile.
Original active encoded1.429B coefficient products still require measurement.
640/6400-bank scenarios are analytical, not additional useful knowledge:
flat routing increases active cost; hierarchical costs require a learned and
validated new router. Proposed640 storage24.84GB versus6400 storage241.31GB.
[318/319 native full-geometry decoder](METH_319_FULL_WIDTH_ADDITIVE_CPU_RESULT_20261003.md)
now passes all source/numeric/format/route/capacity controls on fully allocated
3.187GB with actual source head/router/norm/embedding. All nine repetition
medians38.124–55.526ms exceed14ms; pooled ratio1.456 fails1.10. Unchanged
full-width decoder is CLOSED before training/640-bank allocation.318 startup
failure remains preserved, exact binary319 omits OMP_PROC_BIND to initialize.
No unmeasured cache/DRAM explanation or accepted-rate claim from these fixtures.
The bounded-I8/S7 format is inspired by AQLM but does not inherit its results.
No low-rank output cap, no activation-LUT table builder, no return to unchanged
IQ2/Q2 maps. Full source-aware fitting, useful learned hierarchy, causal whole
quality and same-artifact accepted rate remain missing steps.
No old low-rank factor-LUT result or source-bank size establishes its quality.
The Q4 base-only artifact passed [fresh paired BPB](../donor_adaptation/probes/STRAT_01_GIGACHAT31_FRESH_BPB_PROTOCOL_20260919.md),
[PIQA](../donor_adaptation/probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md),
and [document rollout](../donor_adaptation/probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260921.md).
Those are source/format quality baselines, **not** proof of a low-cost native
target. A uniform ideal W4 would use 814 MB/token for its large matrices, but
the [actual mixed Q4_K_M descriptor ledger](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md)
charges **1,016 MB of active compressed payload/token** including Q6_K,
Q5_0, F32 router, head and controls. At 50 tok/s that is 50.8 GB/s for
weight payload alone. These are calculations, not a measured C rate or DRAM
trace. The old C donor-port line is paused; its 54/64 passing
layer checkpoints and first normalization residual are reusable fidelity
evidence, not an automatic instruction to continue that port.


### Smaller-active pretrained sparse donor branch: provisional Switch

[321 official Switch applicability screen](METH_321_SWITCH_METADATA_SCREEN_RESULT_20261003.md)
selects a lower-active source route after GigaChat representation/cost failures.
Base128/256 have the same123.765M decode matrix coefficients, original top1
routing and source ReLU banks. F32 descriptors497.537/499.896MB include complete
decoder matrix/head/router; BF16 and W8 remain unqualified precision scenarios.
Encoder/cross-KV work occurs per prompt; growing cache/attention must be charged.
Source is pretrained span reconstruction, not instruction/chat evidence.

[324 isolated official reference](METH_324_SWITCH_REFERENCE_RESULT_20261003.md)
qualifies official4.57.6 with unchanged project Torch2.6; installed5.13.1 fails
323 capacity/cache semantics. Source code byte-equal to pinned official release.
Capacity resets per call; saturated full-prefix versus tokenwise equivalence
is not assumed. No installed package is patched.

[325 actual source headers](METH_325_SWITCH_SOURCE_HEADERS_RESULT_20261003.md)
and [326 full acquisition](METH_326_SWITCH_ACQUISITION_RESULT_20261003.md)
bind original base256 six archives and seven side files,58,860,061,113B, all
expected whole LFS/config/tokenizer hashes,1687.563s within frozen budget.
[327 actual tensor binding](METH_327_SWITCH_TENSOR_BINDING_RESULT_20261003.md)
verifies all6392 finite F32 tensors and four byte-equal embedding/head aliases,
confirming14,664,154,368 unique architectural parameters. All12 banks contain
256 distinct WI/WO parameter-tuple hashes each. Tuple uniqueness does not prove
effective function diversity, exposure or useful additional capacity. Source128
headers exist; its full original weight payload is not acquired.

[328 native Tiny contract](METH_328_SWITCH_NATIVE_CONTRACT_RESULT_20261003.md)
implements exact original source mapping/loader, bidirectional encoder, relative
position buckets, self/cross attention and cache, per-call capacity, original
selected probability multiplier and scaled tied full vocabulary head. Both
capacity controls and nine semantic faults PASS. Prototype bounds are batch1,
CTX256/E<=256/D<=1024/L<=24/FF<=4096, not generic RAM-scale n.

[329 actual full source](METH_329_SWITCH_FULL_SOURCE_RESULT_20261003.md),
[330 stable RMS](METH_330_SWITCH_STABLE_RMS_RESULT_20261003.md) and
[332 F64 matrix dots](METH_332_SWITCH_F64_MV_RESULT_20261003.md) each verify
all6392 actual C mapped tensor bytes against327. Pooled/per-state1e-4 logits,
exact source route choices/capacity/greedy controls pass; selected probability
1e-6 fails case0 at1.2815/1.6391/1.4901e-6. Original frozen qualification FAILS.
Do not promote any unchanged candidate. Inputs are two consumed engineering
controls, not untouched whole donor-relative quality or accepted rate.
[331 actual trace](METH_331_SWITCH_ROUTER_DIAGNOSTIC_RESULT_20261003.md) preserves
330 outputs byte-exact and decomposes the worst error into both upstream input
and classifier contributions; local router F64 alone remains insufficient.
This does not prove a universal rounding floor.

The [specified numerical policy](SWITCH_NATIVE_ARITHMETIC_POLICY_NEXT_20261003.md)
is now implemented333/334, explicitly separating target correctness from
original pretrained quality. [333](METH_333_SWITCH_SPECIFIED_ARITHMETIC_RESULT_20261003.md)
reference norm primitive FAIL retained before native/full source execution.
[334](METH_334_SWITCH_ROUNDED_REFERENCE_RESULT_20261003.md) explicitly rounds
sqrt/reciprocal boundaries and passes SAME NumPy oracle, both Tiny/nine faults,
all6392 source native bytes and complete source matched arithmetic. Probability
max1.6391e-7, all discrete routes/capacity/greedy exact, unchanged1e-4 every-state
and1e-6 probability gates against independent recipe. Original329/330/332
original-backend qualification remains FAIL; no original-donor quality promotion.
Original unmodified donor remains PRIMARY for a NEW untouched quality cohort.
Full precision apparatus is not a final compact fast artifact.

The [compact integer path](SWITCH_COMPACT_INTEGER_NEXT_20261003.md) is now
partly reproducible.335 writes ALL6392 tensors but fails final20min readback;
337 verifies1722 readonly segments and fails its time guard. Preserve both.
[338](METH_338_SWITCH_TENSOR_RECOVERY_RESULT_20261003.md) changes verification
policy explicitly: freshly bind ALL original coefficient bytes to327 plus
config/tokenizer, compare entire immutable target from RAM with reconstructed
original335 codes/controls/scales/padding. All gates PASS; no fresh ZIP-envelope
claim. Output14,818,015,744B,6313 row-I8 and79 F32 names, one F32 embedding
physical copy plus separate head I8 view from SAME tied original. All12x256
target tuples remain distinct. Conversion verification1032.625s/25.246GB RSS;
previous interrupted1200.015+1200.093s are additional cost.

[336 native contract](METH_336_SWITCH_W8A8_CONTRACT_RESULT_20261003.md) now
PASS: AVX2 I8->I16/VPMADDWD/I32, F32 absmax/nearest-even I8 activation,
F64 scaling/F32 return; full serialized codes/controls/scales identity and
complete matched independent NumPy I64 reference outputs EXACT. Tiny/extreme/
nine faults pass.422.859s/max2.018GB combined RSS. Official META architecture
loads actual F32 controls/lookup; all I8 Linear shape descriptors replaced by
actual serialized integer forward. Dynamic n loader within RAM/shape bounds,
only real256/Tiny2 tested. Original331 cached consumed diagnostic case1 has
8 changed routes/1 of5 greedy choices, no untouched quality claim.

[339/340 actual CPU](METH_340_SWITCH_W8A8_COST_RESUME_RESULT_20261003.md) retains
premeasurement engine-serialization stop and full cost FAIL on64-token prefill/
repeatability. [341 batched prefill](METH_341_SWITCH_BATCHED_PREFILL_RESULT_20261003.md)
changes only encoder Q/K/V and cross-K/V execution schedule; all complete
outputs exact340/336. All unchanged6-thread cost gates PASS: actual full
encoder/crossKV/32 forced decoder positions median7.643/17.510ms per position
for source9/64; decoder6.652/8.442ms, repeat ratios1.039/1.048. Source64thread1
comparison still fails20ms/stability. Actual logical decode matrix addressed
bytes129.017MB/position; physical DRAM not inferred.39.891s340/51.329s341
apparatus cost, maxcombined~0.928GB. Forced positions not accepted generation.

[342 new source manifest](METH_342_SWITCH_FRESH_SPAN_MANIFEST_RESULT_20261003.md)
now makes24 new PG19book rows/96 known8-token masks reproducible;36 previous
book-ID exclusions, four distinct63-token windows/book, source57/target11.
Model-free selection46.329s/1.504GB endRSS, original tokenizer/config bound;
no model-score selection or unknown pretraining-exclusion guarantee.
[343 prediction](METH_343_SWITCH_FRESH_PREDICTION_RESULT_20261003.md) completes
ORIGINAL unmodified official CPU1 PRIMARY versus native341CPU6 SAME artifact.
Original canonical ALL coefficients and cached331/native336 bridges exact.
Seven gates PASS: book-bootstrap NLL upper+.013893 all tokens/+.034667 mask,
accuracy limits pass; top1 agreement94.791667% FAIL95%. Unchanged W8A8 is
closed before free generation/task/accepted-rate promotion. All342 books now
consumed; strict teacher-forced8-token exact0/96 in both arms is insensitive.
[344 attribution](METH_344_SWITCH_HEAD_ATTRIBUTION_RESULT_20261003.md) has
independent source-head/native-head/primitive oracles PASS. At fixed consumed
native states, HEAD A16 with identical I8 weights yields35 choice differences
versus55, equal to more costly source-F32-head count. This supports a NEW
head-only activation16 execution variable, not new quality or changed343 gates.
[345 initial apparatus](METH_345_SWITCH_HEAD_A16_CONTRACT_RESULT_20261003.md)
input-filename binding stop preserved before observations. [347 numeric](METH_347_SWITCH_HEAD_A16_CONTRACT_RESUME_RESULT_20261003.md)
qualifies actual HEAD A16 execution: I32 lanes/I64 global reduction, thirteen
extreme/tie/scalar cases, Tiny/nine faults, full independent engineering states
and ALL96 consumed343 forward states/routes EXACT with344 head counterfactual.
152.266s/max3.170GB. [346 actual SAME binary CPU cost](METH_346_SWITCH_HEAD_A16_COST_RESULT_20261003.md)
PASS unchanged6-thread gates: full7.452/15.372ms per32 forced positions for9/64
source, decode6.426/7.207ms, repeats1.0174/1.0037.37.297s/max0.928GB; unchanged
129.017MB logical matrix weight reads/position, no physical DRAM inference.
The64-source prefill~261ms is not hidden:32-position amortization does not
establish >=50 accepted end-to-end on shorter responses. Need actual generation.
[348 NEW sources](METH_348_SWITCH_HEAD_A16_FRESH_MANIFEST_RESULT_20261003.md)
select24 new books/96 known8-token masks without scores,62 old exclusions.
[349 original-primary quality](METH_349_SWITCH_HEAD_A16_FRESH_PREDICTION_RESULT_20261003.md)
ALL eight unchanged thresholds PASS on96 NEW masks: original top1 96.7803%,
mask NLL upper+.008128, all-token upper-.007338. Known accuracy-.78125pp with
lower-1.43229pp within-2pp; do not call identical quality. All350 predecessor
348 books now consumed, old343 FAIL remains. Strict8-token exact still0/96
BOTH arms, so a NEW more informative task is required.1835.985s/max9.337GB.
[352 selected real banks](METH_352_SWITCH_CONSUMED_ROUTE_COVERAGE_RESULT_20261003.md)
visits original238-256/encoderbank and208-231/decoderbank in consumed343 paths;
parameters distinct and broadly consulted, no causal useful-capacity scaling.
[350 NEW source-only task](METH_350_SWITCH_MULTI_SPAN_MANIFEST_RESULT_20261003.md)
24 books/96 four-short-span cases,87 old exclusions, source29/labels14.
[353 wrapper numeric](METH_353_SWITCH_GENERATION_CONTRACT_RESULT_20261003.md)
PASS full exact Tiny/official-cache and actual engineering greedy trajectories
through unchanged345 mathematical forward, new explicitly identified binary;
73.235s/max1.896GB. No original free-generation quality from engineering data.
[351 full task](METH_351_SWITCH_MULTI_SPAN_QUALITY_RESULT_20261004.md) FAIL:
16/18 prospective gates PASS; masked-only original top1 agreement94.791667%
versus95% and prose-edit upper.122823041 versus.10 fail. Original informative
known-field exact19.53125%/healthy88.541667%, native89.583333%; known-answer
noninferiority gates PASS. ALL96 teacher and natural own-cache trajectories,
fresh coefficients/official/native bridges retained30ca546. All350 sources
consumed.349 prediction and353 numerical PASS stay scoped;354 rate draft
ineligible/unexecuted.355 separately frozen consumed-state diagnosis restricts
natural comparisons to identical decoder-input prefixes through first changed
choice. Source/head-weight/upstream precision attribution precedes a new
variable, whole numerical/cost and NEW full prediction/generation/task gates.
355 full independent head oracles PASS; original-state target head leaves8
versus40 masked changes, native original F32 head leaves40.356 ALL quantized
inputs A16, SAME I8 weights/F32 controls/router/lookup, full independent
numerical/cache/natural/long controls ALL7 PASS.357 apparatus path stop and
358/360 CPU6 cost-repeat failures retained.361 new PRIMARYCPU1/one actual
physical-core logical processor[0] profile ALL6 cost gates PASS, SAME356 binary,
full8.1876/17.9727ms per32 forced position, repeat1.0123/1.0285. Not accepted
natural rate.362 NEW source-only24books/96 four2-token-mask cases,111 prior
exclusions/source29/target14, ALL guards PASS.363 NEW full original-primary
quality ALL18 PASS retained3396ee2, masked original-top196.4844%/prose-edit
upper.091533, original/native healthy82/81 of96. ALL sources now consumed.
364 SAME356 executable/CPU1 affinity[0] full accepted rate FAIL retained93e7cac:
45.1114 ordinary IDs/s (includes markers)/lower42.5362<50; prose-only24.6979,
ALL rejected-case full times included. Repeat1.005005 PASS.365 new exact encoder
O/denseFF/F32router batching/four-token weight-conversion reuse requires fresh
independent numerical gates and ALL96 complete forced/own natural bytes EXACT
363 before inheriting its scoped quality and measuring366cost/367full rate.
365 ALL9 numerical/ALL96 byteequivalence PASS;366 CPU1 repeatFAIL1.101982
retainedc0611e7, no367 rate promotion. Next368/369 use SAME stable361/356
binary/quality for learnedbank identity interventions through smallserialized
manifest remapping, unchanged payload/math/router, fixedtwo nonidentity
bijections. This is diagnostic usefulness evidence; frozen paired task/NLL
gates follow independent complete control oracles.368 ALL7 numeric/serialized
bankcontrols PASS;369 ALL7 pairedconsumedbankharm PASS, BOTHfixedcontrols
maskedNLL+2.8278/+2.6429nats andgeneratedfieldexact-18.75/-19.79pp. Supports
matched learnedbank usefulness, noteach-expert/extra-n gain/NEWquality/rate.
370 exact physically compact64/128 exports PASS;371 both dynamic-n full
independent native contracts PASS (same356 binary).372 paired consumed
quality claim FAIL5/7:64 loss supports additional-function contribution;
128 improves masked NLL but has mixed generation outcomes. Neither subset
preserves original teacher choices (masked agreement38.28%/49.22%); neither
inherits363 quality. More n does not establish monotonic quality benefit.
[372 result](METH_372_SWITCH_NESTED_BANK_USEFULNESS_RESULT_20261004.md) retains
all primary and descriptive outcomes. Separately frozen373 will compare equal
source29/forced14 CPU phase/routing/consultation cost across actual64/128/256.
Logical bytes, mapped file bytes and sampled RSS do not measure physical DRAM.
No relaxed acceptance/50/1.10 or optional repeats. Whole quality remains bounded
English short four-span infilling; useful RAM-scale n/LUT/physical DRAM/causal
bank identity/real128/cross-family/~100B goal remains active.

An alternative sparse/recurrent family is **Granite 4.0 H Tiny base**. The
[metadata screen](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md) binds an
official revision with 6.939B total and 1.465B analyzer-active weights/token,
36 Mamba2 and four attention layers, 64 routed experts/top-6 and a shared
path. Its ideal W4 active payload already exceeds the 14 ms design
allotment at 40 GB/s. The [official Q4 GGUF header
ledger](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md) charges 913 MB/token;
only the header prefix was fetched. Full local weights, a paired quality
baseline and a Mamba2 operator bridge are missing. It is a candidate, not
evidence that the method transfers.

Another locally bound sparse donor is **allenai/StdMoE_1b14b_1T_Preanneal**,
revision `d2a4949c9d4ad6cf47fbac131f7e020077332b21`: 16 layers,
128 experts/top-8 and one shared expert, with all 11 fp32 shards present
([acquisition](../donor_adaptation/probes/STRAT_02_STAGE0_ACQUISITION.md)).
Its existing [W4 paired gate](../donor_adaptation/probes/STRAT_02_W4_BF16_QUALITY_RESULT.md)
failed: +0.026025 BPB, upper one-sided CI95 +0.027178 against the
+0.02 bar. Restoring only the fp32 router improved the point to
+0.019532, but the [heldout upper CI](../donor_adaptation/probes/STRAT_02F_W4_ROUTER_F32_HELDOUT_RESULT.md)
was +0.020395 and still failed. That arm prices 676,397,056 active
bytes/token before native overhead: a 16.91 ms/token payload floor at
40 GB/s, leaving only 3.09 ms of a 20 ms total budget. Thus this real
E128 source is a **different-family transfer candidate**, not a
demonstrated quality-and-rate path; repeating its
already failed nominal W4 gate would not advance the method.

The dense-source transfer challenge is **Qwen2.5-1.5B** at revision
`8faed761d45a263340a0528343f099c05c9a4323`, whose local snapshot
contains `config.json`, tokenizer and safetensors, now hash-verified with
the H1 checkpoint in the [local readiness record](METH_10_QWEN_LOCAL_READINESS_20260926.md).
It is dense and has a
different mixer, so it tests a different branch of the procedure. Existing
training on 8/28 FFN layers improved a hard carve from 1.096636 to 0.962593
BPB at H1 S2 and **0.923490** at the later H1 S3 checkpoint, while the
intact donor was 0.767595; that is an adaptation signal, not
quality retention ([H1](../donor_adaptation/probes/H1_THE_CARVE_TRAINED.md)).
The frozen H4+H2I composition worsened BPB by 0.245430 and free generation;
do not reuse it as a final model or re-run the same cell
([H5](../donor_adaptation/probes/H5_CROSS_COMPOSITION_RESULT.md)).

Candidate selection currently uses these reproducible checks:

| Check | Input / tool | Output and decision |
|---|---|---|
| Identity and rights | Exact source revision, license, tokenizer, shard hashes; existing source-binding manifests | Reject an unknown or mismatched payload before scoring; record any terms that restrict use |
| Operator inventory | `benchmarks/donor_adaptation/donor_inventory.py analyze` for covered configs, plus the donor-specific tensor-header/organ ledger | Classify dense versus routed/shared experts; list mixer, router, head, tokenizer and unsupported C operators |
| Conditional cost | Count distinct stored weights, active weights per token, router and head separately; price bytes with measured hardware rates | If the active path cannot plausibly fit a 20 ms total budget, require a stated transformation before porting |
| Baseline quality | Paired donor/format held-out scorer, task and rollout instruments with pinned IDs and document split | Keep the source donor as the reference; reject a broken input or tokenizer mapping |

The initial dense-adaptation selection was **provisional**: same-geometry
Qwen2.5-0.5B-Instruct passed the [METH-42](METH_42_INSTRUCT_DONOR_PILOT_RESULT_20260927.md)
chat-format pilot's repetition screen, but broad task quality and
packed/native conversion remain unverified. GigaChat is the locally bound
sparse ~10B capacity case and now has a measured pretrained-expert rank
screen. Retain StdMoE E128 as a
second sparse family only if a transformation addresses both its
measured W4 fidelity and active-byte failures. Do not claim one
procedure handles these donors until their full pipelines pass.
For another family, rerun the inventory and choose a variant from its actual
operator/traffic map; do not infer compatibility from model names or parameter
counts. The metadata-only Granite/LFM candidates are not local quality or
rate results.

## 2. Current dense-donor procedure

1. Bind source revision/weights/tokenizer and all parent/child artifacts.
   Preserve pretrained behavior with zero-output residual experts before
   actual route/function training. Existing accepted centered E1280 uses
   product-key128parents/top4,10 child choices and nonlinear rank8 BF16
   residuals; parent/child projections rank64/rank32. This is a small-rung
   result; useful tenfold new capacity remains unproven.
2. Qualify a compact source FFN: rowQ8 gate/up,513-entry SiLU LUT,32 BF16
   private source input rows,rowQ8 down with32 BF16 escapes/output row,
   separate BF16 rank32 residual and biases. All4864 features remain active.
   Non-FFN organs/tied head remain exact BF16. Local error/cost does not
   establish complete-model quality or constant-cost larger dense transfer.
3. Audit effective BF16 functions. Canonical A/B signatures and finite
   real-state distances detect aliases. Save unique childB/shared parentA
   and int32 leaf maps, preserving original route labels and exact selected
   BF16 values. Distinct hashes alone do not establish useful capacity.
4. Export all725 tensors/config and load every executed weight from the
   archive, with no source/checkpoint fallback. Preserve tied embedding/head.
   K64 int8/FP16 proposal is only for exact-head reranking; full probabilities
   remain required for NLL/BPB/tasks. Finite K64 passes are not universal.
5. Compare consumed development, then select excluded sources and freeze
   source-only answerability before independent model scoring. Evaluate
   whole-model BPB/category/ranking plus generation/task/semantic behavior
   against original donor and corresponding BF16 learned-bank reference.
6. Qualify native whole-model arithmetic/decoding on the same archive before
   its accepted rate. Do not add separate component quality/speed passes.

| Tool/step | Reproducible inputs/output | Evidence and cost |
| --- | --- | --- |
| `meth259_unique_bank_core_export.py` | Pinned211 non-FFN archive,252 source fixture,parent/child checkpoints,125vectors -> full259 archive | 82.750s,3.95GB RSS/3.28GB GPU; all725 fields and6144 component controls exact |
| `meth259_unique_bank_core_export.load_stored(path,device,expected_sha)` | Archive plus implementation dependencies -> full24-layer model | No donor/checkpoint fallback; all executed tensors archived, tied head;9 helper byte hashes stable across Git checkout |
| `meth260_complete_core_development.py` |259 and consumed24-source controls | All gates pass,105.172s; consumed data |
| `meth261_complete_core_fresh_manifest.py` and262 annotations | Fixed pool/exclusions ->24 new sources and source-only answers | 3176 exclusions;220.360s CPU selection; different PG19 rows, not new corpus |
| `meth263_complete_core_fresh_prediction.py` |259 and261 sources -> independent prediction | All gates pass,104.250s |
| `meth265_full_prefix_generation.py` | Same sources/weights ->72 greedy128-cap continuations | Health/K64 pass,531.656s; full-prefix reference, no production-cache/rate proof |
| `meth266_complete_core_piqa.py` | Same259,all1838 cached PIQA items -> option NLL/paired accuracy | Frozen regression; repeatedly used task, not independent sample |
| `meth267_complete_core_blind_semantic.py` |265 outputs,261 excerpts -> anonymous panel/map/committed verdict/counts | Frozen strict123 rubric; single-agent arm-anonymous review, not independent human study |

Commands, frozen hashes/budgets/gates and scoped results are linked from
[INDEX](INDEX.md). Preserve apparatus failures before narrowly specified
repairs. Candidate/data/precision changes require new prospective checks.
The264 original BF16 donor differs on2/48 cached/full choices. Keep that
failed cached recipe and zero-mismatch guard;265 full-prefix recomputation
is a semantic reference, not efficient production decoding.

## 3. Native integration and useful n scaling: required work

The253 fixed-source FFN C operator measures9.318ms over24layers and passes
finite component controls. The older BF16 E1280 native whole-model reference
measures16.818tok/s,20.791 with selected head, below goal. Those paths differ
from259; combining their times cannot prove259 rate. Its actual native
complete model/unique-bank alias lookup remains unqualified.

Load exact archived parameters; qualify actual CPU arithmetic/cache/generation
and price core,router/selection,selected experts,head,context/KV,dispatch and
glue. Record CPU/compiler/thread/context/batch/decoding configuration, actual
cache residency and DRAM reads. Measure prefill/TTFT separately; final lower
CI95 must reach50 accepted batch1decode tokens/s,100 as stretch. Preserve
held-out/generation/task quality on that same timed native artifact.

For useful n, keep top-k/core/precision/data recipe explicit; learn genuinely
different functions and evaluate untouched sources with displaced/rotated
route controls. Report per-expert exposure,storage,route distribution,
selected-set unions,real DRAM traffic and full routing/LUT cost as n grows.
Hierarchical selection still has parent/key costs. Synthetic/copied pools
can measure component cost but establish no transferred capability.
The current C geometry fixes128 parents/10 children. The
[capacity accounting and prior stops](NATIVE_CAPACITY_SCALING_STATUS_20261002.md)
give actual source-derived operand counts and distinguish RAM capacity,
linear child-search work, stored distinct functions and useful learned
content. Its hypothetical larger rows are not implemented model results.

## 4. Resources, applicability and evidence retention

Record source revision/hash,tokenizer,data IDs/splits,output artifacts,
conversion/training time,GPU-hours,tokens,peak RAM/optimizer state,distinct
versus tied/copied parameters,resident/active/read bytes,stage latency and
whole accepted rate. Local RTX3060/six threads with frozen wall/RSS/GPU
stops; T4 requires prior reason/budget/stop communication. No CPU rate test
overlaps a model job; ideal bandwidth is arithmetic, not a measured rate.

Sparse/recurrent/larger families require source fidelity, full active-path
pricing including MLA/shared/head and their own compact transformation and
complete quality checks. Dense0.5B results do not qualify10B/100B or another
family. Useful RAM-scale n, same-artifact whole quality/rate and demonstrated
cross-family/scale applicability all remain part of the final goal.

Detailed former decisions and variant evidence are preserved in
[METHOD_HISTORY_THROUGH264](METHOD_HISTORY_THROUGH264_20261002.md); historical
next steps there are superseded. This document records the procedure;
operational process/result/resumption state lives in [INDEX](INDEX.md).
