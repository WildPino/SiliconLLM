# Pretrained-to-native conditional-capacity method (work in progress)

**Status: research procedure, not a validated conversion recipe.** This file
describes what can be reproduced now, what has failed, and what still needs an
experiment. The target is one identifiable artifact derived from a pretrained
LLM, retaining useful held-out, generative and task quality against that donor,
running in the project C engine at â‰¥50 accepted batch-1 tokens/s on a declared
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
compiled-run timing causes remain unisolated. A new exact-I8 biased-byte/
two-part activation kernel is proposed, not implemented/measured; preserve
all source precision/fixtures/geometry/head and exact outputs, independently
prove pair saturation/I32 limits, freeze its full native gate before data fit.
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
