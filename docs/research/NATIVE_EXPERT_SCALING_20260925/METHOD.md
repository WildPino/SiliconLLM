# Pretrained-to-native conditional-capacity method (work in progress)

**Status: two bounded same-family pretrained source scales have qualified quality/rate;
generality and useful larger-n scaling remain research.** This file
describes what can be reproduced now, what has failed, and what still needs an
experiment. The target is one identifiable artifact derived from a pretrained
LLM, retaining useful held-out, generative and task quality against that donor,
running in the project C engine at ≥50 accepted batch-1 tokens/s on a declared
CPU and context. The same artifact must pass quality and rate. A second donor
family must still test which steps transfer beyond this same-family result. More stored parameters alone
are not evidence of transferred capability.

## Latest capacity-transfer question

401 real128+256 bank384 union fits RAM but all236 core/control names differ.
403 identity,404 rank32 and406 full affine ridge input interfaces fail local
validation;405 unregularized full map is ineligible by conditioning. No384 model.
407 actual immutable Ling-mini headers enable another-family COST screen:
19 banks256/top8, source16.2556B, SAME317 hypothetical503.905MB active descriptor.
No original Ling values/reference/native quality/rate or physical DRAM yet.
408/410 full-width/joint formats cost FAIL;409 attribution PASS. Ling fitting stopped.
411 actual Granite1.335B full row-I8 byte inventory433.232MB; Granite3.299B
unchanged I8 geometry closed.412 exact I8/A16 full fixture math/archive PASS,
both3/6 cost/repeatability FAIL; no original values/reference/export/quality.
413 four-row I8/A16 exact412 full outputs/ALL11 controls PASS but both14ms CPU
gatesFAIL; stop Granite I8 kernel route before values/reference/quality.
414 source-lexical full basis failed conditioning before validation;415 spectral
diagnosisPASS, both source ranks768/cross767 at1e-6.416 identified rank767 partial
map mathPASS/ALL12 input gatesFAIL, error1.458..4.293 times mean-only.
Lexical weight bridge closed before384 selector/model, no further map tuning.
417 capture controller schema failure retained before compile/native.418 repairs
only acquisition status handling, ALL9 gatesPASS: all192 paired teacher+192
natural full outputs/routes/greedy unchanged;4895 final-bank states with complete
selected WI/WO integer replay and normerror0.372.984s/3.096GB RSS/1.955GB output.
[418 observation contract](METH_418_SWITCH_FUNCTION_CAPTURE_RESULT_20261004.md)
is reproducible, without changed model arithmetic or new accepted-rate claim.
419 apparatus disk-guard failure retained;420 ALL384/4895 zero-forward states
and ALL157266560 vocabulary rows byte-exact, but first real directional FD fails.
421 diagnoses loss-subtraction conditioning; stable equivalent loss fixes that
numerical check.422 actual256 all gradient/FD controlsPASS but128 A mixed-primal
STE vs global-smooth gradient error.004939>same1e-3, contract not eligible.
[423 local backward diagnostic](METH_423_SWITCH_SAVED_PRIMAL_GRADIENT_RESULT_20261004.md):
independent same128 native-primal A/C chain vs autograd errors0.0; no ReLU mask
changes. Using different loss logits accounts for global gradient discrepancy;
full native/smooth logiterror5.618e-5. No derivative through native rounding claim.
[424 matched-primal local qualification](METH_424_SWITCH_MATCHED_PRIMAL_RESULT_20261004.md),
frozen188a9ef: ALL8 PASS, F64 independent NumPy FD and ordinary autograd at SAME
native values, both real ALL6 gradient fields max9.86120e-8. No ReLU crossings;
constant offsets fixed throughout FD.29.547s/max1.805GB, two real rank8 forwards
exact418; ALL384 proof inherited420 through unchanged419/422 prefix and fresh
archives. A local approximate derivative, not through rounding/hard selection.
[425 first pilot failure](METH_425_SWITCH_FUNCTION_PILOT_RESULT_20261004.md):
actual cross composition/affine prerequisitesPASS, tiny adapter score FD fails
before updates. [426 complete pilot](METH_426_SWITCH_FUNCTION_PILOT_RESULT_20261004.md),
frozena824d2a/retaineda3a540e: stable equivalent relative-CE FD reproduces old
error and passes SAME bounds; ALL6336 fixed updates/complete matched control.
Capacity2/9PASS/7/9FAIL: real hard mixture CE2.325199 vs original2.293060,
adapter2.295684;47 added positions/30IDs,3 locally useful vsrequired8, removal
helps. Original prediction preservation also fails.422.953s/max3.146GB.
Close THIS learned replacement recipe, no rank/seed/epoch/gate/lr sweep.
[431 completed additive pilot](METH_431_SWITCH_ADDITIVE_PILOT_RESULT_20261004.md),
frozenb287b91/retained55e0cb5: ALL1344 zero-added predictions exact, ALL6240
fixed updates/two matched arms/frozen426 classifier.427/429 numeric failures
retained;428 precise tiny checker and430 fixed real numeric probe diagnose and
qualify SAME original bounds. Training teacher-mixture objective unchanged.
Capacity2/9PASS/7/9FAIL. Original256 preserved within mean/book limits; no useful
added IDs or required benefit. Real mixture2.298618 vs original2.293060 and
adapter2.292198,21added positions/18IDs/ZERO useful, function removal improves.
Proposed original+extra when ON is TWO final-bank functions, other11 unchanged;
no combined C model or inherited quality/rate/actualDRAM.463.281s/max3.185GB.
Close THIS additive-feature/rank8/fixed-selector recipe before further fitting.
[433 output-level feasibility](METH_433_SWITCH_OUTPUT_BOUND_RESULT_20261004.md):
432 first F32 comparator failure retained; sole F64 caller repair as431,
ALL6 apparatusPASS. Arbitrary posterior under mean256KL.02/book<=.05 achieves
17.63%mixture/26.64%128gain, numerical duality gap5.60e-14. Oracle, not a model.
55IDs have>=2val observations before frozen gate, only2after.434
[frozen function/selection](METH_434_SWITCH_FROZEN_SELECTION_RESULT_20261004.md),
47f3f43:zero updates, original431 hard full matrices/losses and masks exact.
Fixed oracle selection yields12/10locally useful IDs but about.5%mixture gain
and insufficient permutation harm. Both routes fail ALL9 potential requirements.
Close THIS checkpoint before fitting another selector; actual431 staysFAIL.
125.313s/max2.492GB/259.081MB, six full matrices/actual source hashes retained.
Next [shared nonlinear geometry hypothesis](SWITCH_FUNCTION_RETARGETING_PILOT_NEXT_20261004.md):
NEW435 one shared768->128->768 input transport supervised on all1008dev
captures; independent new numeric prerequisites before any fit. No435 source/
protocol/outcome, no model export from local regression. Existing404/406 input
eligibility and ALL9 capacity/whole-quality/SAMEartifact50 demands unchanged.
Original routes intact; usefulRAM-n/LUT/DRAM/another-family/~100B remain open.

## Independently pretrained source128: complete bounded transfer qualified

The independently pretrained original `google/switch-base-128` (revision
86c815ec05361a33a8b49fc717277da9c0a4e711) is locally acquired and actually
bound, not the370 pruned base256 subset.378 verified all29,862,948,392 archive/
side-file bytes in895.625s;377 pre-network protocol-path failure remains
retained.379 verified all3320 F32 source tensors, four tied aliases,7,415,217,408
architecture-unique parameters and12 banks with128 distinct WI/WO parameter
pairs each (55.047s/max10.197GB RSS). Different parameter pairs do not establish
useful or mathematically different effective functions.

380 converted ALL source banks/core using the unchanged335/337 row-I8/F32
recipe, all segment readback and six export gates PASS (263.172s/max10.422GB
RSS). Complete new target7,541,946,880B, payload SHA
6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe;
spec SHA3a4892ad2bc846937a3ad7898048df64b9a26dc391a3d02c110fe75cd614511a.
This is serialization/conversion evidence only.381 source-specific complete
independent target arithmetic/cache/route/own-greedy ALL8 PASS (132.610s/2.657GB).
382 NEW source-only24-book/96-case cohort PASS;136 previously used rows excluded.
383 SAMEartifact CPU cost first FAIL6/7: source64 decode repeat1.259571>1.10,
all full output/worker/profile/latency-median gates pass. Keep383 authoritative;
draft384/385 dependent chain ineligible. NEW386 separately tests quality only
using unchanged363 ALL18 rubric, with performance explicitly unqualified.
386 cached/uncached original teacher bridge failed before any NEW scores
(1.12696e-6>1e-6); raw preserved. NEW387 uses matched official cached teacher
mode via generate raw-logit capture, same1e-6 bridge and unchanged ALL18 rubric.
No rate promotion follows quality without a justified new performance variable.
387 ALL18 NEW original128 quality PASS (662.265s/max7.010GB), independent
original PRIMARY and ALL96 official cached teacher logits exact0 difference.
Overall original teacher agreement97.47%, masked97.14%; generated-field counts
92 original/89 native of384, both96/96 healthy; prose-edit upper95 .089451.
389 ALL192 complete teacher/natural bytes exact363/387 at NEW three actual
physical workers[0,2,4].390 independently qualifies original128 costs ALL6;
full256 three-worker repeat1.105770>1.10 FAIL remains closed.391 SAME original128
artifact/389 binary/390 profile ALL5 accepted FULL-rate gates PASS:
96.56385 ordinary IDs/s/lower95 94.58827;55.97659 prose IDs/s/lower54.59729.
96/96 healthy,1142 ordinary IDs INCLUDING480 sentinels,662 prose IDs;
ALL96 median FULL time11.826372s, aggregate repeat1.012403. Warm short fixed-ID
full model computation, same exclusions as376; no cold text service claim.
391 cost166.313s/max1.335GB. The second independently trained source scale now
qualifies joint whole quality/rate with its source-specific worker profile.
383 six-worker failure remains authoritative, and draft384/385 ineligible.

The question is repeatable preservation across two actual trained sources at
same-family7.4B/14.7B scales and128/256 candidate counts. Different pretrained
core/weights/exposure prevent a matched-training causal n comparison. The positive
source-specific outcome leaves useful>256/10x n, hierarchical CPU LUT/routing,
physical DRAM, additional families, broad contexts and~100B open.

## Current qualified route: original Switch base256 to native C

[376 complete rate](METH_376_SWITCH_PHYSICAL_WORKERS_ACCEPTED_RATE_RESULT_20261004.md)
qualifies the SAME model whose [363 original-primary quality](METH_363_SWITCH_ALL_A16_MULTI_SPAN_QUALITY_RESULT_20261004.md)
passes all18 frozen gates. Original14,664,154,368 unique pretrained parameters,
12 banks each256 distinct real WI/WO functions, compact14,818,015,744B target.
[374 execution contract](METH_374_SWITCH_PHYSICAL_WORKERS_CONTRACT_RESULT_20261004.md)
passes complete source reversal/independent primitive/Tiny/fault/cache controls
and ALL96 teacher/own-natural full output SHA EXACT363. Same374 binary with
[375 CPU cost](METH_375_SWITCH_PHYSICAL_WORKERS_COST_RESULT_20261004.md) PASS is
measured in376:63.5254 accepted ordinary IDs/s, one-sided lower95 59.5118>=50;
81/96 accepted cases/895 IDs, ALL rejected-case full time charged. These IDs
include405 structural sentinels; prose-only34.7793/lower32.6497. Preserve that
counting boundary: this does not establish prose-only50 or instruction chat.

Declared hardware/runtime: Ryzen53600X/80GiB/Windows, primary6 workers on
physical cores0,2,4,6,8,10, each live mask/group/thread ID read back, ACTIVE /
infinite blocktime. Original-primary reference uses qualified isolated324
Transformers4.57.6 with official original weights and forward/cache semantics.
Scope:24 initially NEW PG19 books/96 short four-span cases, source29/cap64,
observed natural length10-13. All sources now consumed. Model FULL encoder/
crossKV/cached decoder/head/argmax/stop included; warm/hash-primed fixed IDs,
startup/initial worker setup/load/tokenization/serialization/cleanup excluded.
No cold text-to-text service wall-time or broader-context inference.

| Available step | Classification | Evidence / cost |
| --- | --- | --- |
|321/325 source metadata,326 acquisition,327 all original tensors/ties | Exact source identity and applicability screen |58.860GB source,1,687.563s acquisition, all6392 finite canonical tensors, tied aliases exact |
|335 row I8 representation,338 recovery/readback | Lossy representation, exactly verified implementation |14.818GB actual codes/scales/F32 controls; interrupted335/337 retained,338 final readback1,032.625s |
|334 specified arithmetic,356 all quantized inputs A16 | Changed rounding plus activation approximation, independently verified |Explicit norm/softmax/scaling, exact I64 projections/Tiny/fault/cache; original-backend329/330/332 failures remain |
|363 complete comparison to ORIGINAL donor | Joint statistical acceptance of the complete transformed model |ALL18 held-out prediction/generation/known-task gates;1,721.672s/max7.336GB |
|374 physical worker execution | Exact execution transform on qualified target |ALL96 complete teacher/natural bytes exact363;307.844s/max1.404GB |
|375/376 cost and SAMEartifact accepted FULL rate | Measured CPU resource/performance acceptance |37527.078s;376200.453s/max1.337GB, lower59.5118 ordinary IDs/s |
|368/369 real function identity interventions | Diagnostic causal usefulness under fixed intervention |Both fixed mismatches harm prediction and known answers; not individual-expert or additional-n proof |
|370/371/372/373 actual smaller banks | Exact subset export, changed model, diagnostic quality and cost |64/128 physically compact;372 mixed quality;373 fourfold CPU full growth2.91%/upper3.62%, not monotonic n gain |

The compact reusable source core and original real conditional functions are
kept distinct. All12 bank expert pairs total14,495,514,624 original coefficients;
original source uniqueness/ties are counted without treating reencoded head
views or lookup aliases as new capacity. Each decoded position addresses
123,764,736 I8 matrix code bytes +534,016 row-scale bytes +4,718,592 F32 router
bytes, independent of stored expert pool size except the flat router. Actual
full timing includes selected weight access costs; logical descriptors and
RSS do not measure physical DRAM traffic or establish cache locality.

Applicability currently validated at real independently pretrained original
base128/base256 and exact nested64/128 numerical controls. Loader shape bounds d<=1024, ff<=4096,
encoder/decoder<=24, vocab<=65536, heads*dk=d, buckets32/distance128; projection
columns<=4096 and exact I32-lane/I64-sum bounds. These are implementation
preconditions, not proof that every matching model passes quality. Source
names/ties/activation/router/capacity/cache semantics must match the qualified
Switch/ReLU/top1 branch. Generalizing across shapes/checkpoints requires new
source/export/independent numeric and NEW original-primary quality/rate gates.
Larger source C2048 has different widths/FF/core depth and does not fit the
current source/target resource envelope; do not infer~100B compatibility.

[Reproduction and exact artifacts](SWITCH_BASE256_REPRODUCTION_20261004.md)
provides commands and immutable inputs. Old364/358/360/366 failures stay closed;
no optional polishing of the unchanged qualified376 rate. Original
base128 complete transfer is now qualified through391; it is not370's subset
of base256. See SWITCH_BASE128_REPRODUCTION_20261004.md. Cross-family/greater-than256/useful large-n/
physical DRAM and broader context/task proof remain open. The full goal remains incomplete.

## Earlier dense-family complete candidate (rate/quality promotion closed)

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
all primary and descriptive outcomes. 373 actual equal
source29/forced14 CPU comparison ALL7 PASS retained537f01b. Fourfold64-to256
bank increases full cost2.91% (upper95 3.62%), decode0.51% (upper1.27%).
Router matrix profile grows3.83x and is3.63% of OWN full profile time at256.
Every warm/repeated/profile full teacher output exact363/372. Sampled child
RSS1.114/1.269/1.355GB differs from whole mapped3.904/7.542/14.818GB files.
This does not replace failed364 accepted natural rate or establish >256 scaling.
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


## Current routing uncertainty after the two qualified source scales

392 read-only ALL384 retained complete output traces freshly match363/387/389.
Actual selected-probability medians are .098/.106 for256 teacher/natural and
.248/.258 for128. All256 routes would change selected-expert scaling by>1%
under singleton normalization; no changed-model harm was measured. Per-bank
coverage spans173-252 of256 and107-126 of128 teacher choices. Different
sources/cohorts prevent causal n interpretation; visits are not individual
expert usefulness. Existing top1 traces cannot reconstruct ranking/tail mass.

393 captured all full scores/actual inputs, complete384 output hashes/IDs exact
and independent selected probabilities EXACT0. Minimum oracle m for1% scaling
median/95th140/191 at256,71/98 at128 teacher; top64-only normalization rejected.
394 fixed source-only low-rank classifier/refinement grid and395 full-dimensional
I8/A16/refinement grid both rejected. All apparatus controls pass; no changed
router is eligible for whole quality/rate.395 source128 m16 passes22/24 bank/modes
but misses212 selected IDs in the remaining bank; no pooled override. Exact best-expert lookup
alone does not preserve the softmax-scaled original function. Larger actual
useful n, physical DRAM and another donor family remain separate open gates.
Frozen GigaChat314-319 assets are reusable;319 already uses full-width additive
I8 palette decoding and synthetic operator fixtures, not an unquantized dense
baseline or complete learned model. Its38-56ms failed costs cannot be promoted
by inheriting Switch execution rates; unchanged generic port remains paused.


## Current source applicability and exact LUT question

[397 result](METH_397_QWEN_NEXT_SOURCE_HEADERS_RESULT_20261004.md), retained1c8d978,
now bindsALL41 original Qwen3-Next80B shard headers/75,944 tensors. Main count
79.674391296B+MTP1.650471424B reconciles81.324862720B. Actual48 banks have512
original tensor slots, not proven distinct/useful values. All5 metadata gates
PASS;133.468s/117MB/16.655MB downloaded metadata and zero tensor values.
Original BF16/F32 full-resident sizes do not fit80GiB; a streamed reference
remains unqualified. SAME317 hypothetical full-width active descriptor
1,278,965,760B exceeds560MB: unchanged geometry closed before acquisition.
Core/head/router costs all included. Formula omissions/gates/full ledger are
in the report; this is not a general rejection of transformations or80B.
396 initial physical manifest line-ending failure retained460db6c before397
procedural repair. Source-specific selected-topk normalization still requires
an original numerical contract; Switch tail results do not transfer.

[398 result](METH_398_ADDITIVE_INTEGER_LUT_RESULT_20261004.md), frozen6127962 /
retainedf33015c: SAME318 additive representation with NEW integer activation
LUT builder/AVX2 gathers. Source-sized GigaChat64-parent/top4 geometry/precision
fixed; synthetic books/codes plus actual head/router/norm/embedding. All90
64-bit output/route fingerprints EXACT319 and all independent scalar/numeric
controls PASS; medians137.689-164.653ms>14ms/poolrepeat1.195832 FAIL. Builder
included; workspace73.400MB/logical table writes340.001MB/token. MAIN44.281s/
max3.232GB. No actual DRAM, causal model, donor book fidelity or useful larger n.
This exact kernel closed before training/640-bank work. See
[METH-398 clarification](METH_398_INTEGER_LUT_SCOPE_CLARIFICATION_20261004.md)
for fingerprint versus exhaustive byte-comparison boundary.

[399 attribution](METH_399_INTEGER_LUT_COST_PROFILE_RESULT_20261004.md),32802d2:
SAME398 math/90 fingerprints/all scalar/counter/nonoverlap controls PASS. All
nine component medians builder51.58-69.90ms/rows110.31-149.15ms exceed14ms.
[400 joint layout/builder](METH_400_BLOCKED_INTEGER_LUT_RESULT_20261004.md),64d0754:
source-sized geometry/coefficients exact, column planes/new four-palette SIMD
builder/local eight-group tables/integer partial reduction. Scalar full-code
sample reconstruction/LUT/extrema/source controls and90 fingerprints exact319.
Repeat1.04904 PASS;118.697-124.518ms cost FAIL. Before learned codebook fitting
or640-bank work this exact path is closed. No measured400 component/DRAM or
learned original-relative quality; do not extrapolate CPU cache causes.

## Real pretrained384-choice transfer applicability

[401 real-bank applicability](METH_401_PRETRAINED_BANK_UNION_APPLICABILITY_RESULT_20261004.md),
froze a72e4ab/retained f5fce05: ALLsix gates PASS. Fresh qualified whole128/256
hashes/both independent manifests/ALL12 banks384 ordered byte-distinct WI/WO
pairs. Candidate256 core/head once +128 routed functions/router rows stores
22,094,084,608B and fits64.546GB available RAM. ALL236 common nonexpert/nonrouter
core names differ; compatible operators do not prove common hidden semantics.
MAIN294.406s/max41.673MB/22.36GB hashed, exceeds2min estimate but inside10min/
2GiB stop. No model/new weights/training/GPU/T4, payloads unchanged.

[Transfer mechanism and current resumption](PRETRAINED_BANK_UNION_TRANSFER_NEXT_20261004.md):
402 process-classification apparatus failure retained before model observations.
403 (f79b7ba/419fba4) ALL9 apparatus PASS:24 consumed362 case0 books, common
fresh tokenizer hashes and SAME source/prefix IDs; qualified393 source256 traces
freshly bound, original128 new capture complete SHA exact plain389. Actual
workers/negative controls exact. ALL12 identity input interfaces FAIL; norms are
nonzero, median cosine near zero. MAIN75.5s/max1.410GB; no quality/rate claim.

404 (653ad03) ONE fixed rank32 affine PCR fit only18 development books,6 validation
books fixed by403: ALL6 apparatus PASS/ALL12 local input gates FAIL. Median
relative errors0.649-0.897,95th0.728-1.091; squared error0.633-0.936 of mean-only,
exceeding0.25/0.50/0.50 bounds. MAIN1.578s/max102.068MB. Failed maps2,433,024 F32
bytes,589,824 active coefficients; protocol doubled-count typo explicitly
[clarified](METH_404_RANK32_COEFFICIENT_SCOPE_CLARIFICATION_20261004.md). Original
scientific records immutable. Input prediction alone never licenses quality.

405 froze9e7ddac/retained41fb48e: ALL13 apparatus PASS,96 SAME paired source/prefix
cases,72 new128 captures complete output SHA exact plain389 plus24 reused403.
ALL96 source256 traces freshly exact393. Development2088 encoder/1008 decoder
positions. All encoder and decoder2/3 have768 directions at relative1e-6;
decoder0/1/4/5 have761/767/767/767. ALL12 condition gate FAIL so no unregularized
full maps. MAIN137.468s/analysis4.094s/max1.486GB/new594.667MB capture artifacts.

406 froze9518b9f/retainedd4ae8ed: ONE full768 affine ridge per bank, fixed
lambda=(1e-5*smax)^2, same18/6 books and all4 cases. ALL7 apparatus PASS/ALL12
local input eligibility FAIL. Median errors0.521-1.923,95th0.844-2.649, squared
error0.722-3.070 of mean-only. Every decoder map worse than mean-only. Same
0.25/0.50/0.50 bounds preserved. MAIN6.078s/max262.623MB, failed factors28.385MB.
These specific input interfaces closed before source function/output map/new
selector. More affine tuning needs a new uncertainty and independent validation.
No new original-relative quality docs consumed; nonlinear/other-data/weight-based
transfer not globally rejected. Complete qualified374/389 artifacts unchanged.

## Another-family metadata applicability: Ling-mini, not yet a transfer

407 froze6268585/retainede0f2f8b: actual immutable inclusionAI/Ling-mini-2.0 revision
a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab, ALL6 metadata gates PASS.4 shards/14,813
headers/16,255,643,392 source positions;19 banks256 gate/up/down512x2048, top8,
shared512, first dense5120, fused GQA QKV3072x2048, untied157184x2048 embed/head.
Git-bound custom code stored NOT executed, exact dtype/offset/EOF/index extents;
zero values/finite/function uniqueness/quality/native inference. Metadata source
32.511GB value extent, nominal BF16/F32 fit total80GiB but reference overhead open.

SAME317 hypothetical full-width additive selected+core779,091,968 coefficients,
complete addressed descriptor503,905,280B <=560MB gate PASS (including264.069MB
Q6_K head and40.222MB router/controls), complete storage4,969,196,544B. No learned
books or source precision quality/rate inheritance. Active parameter positions
with ONE lookup row1.111B; card-style full embedding counting explains~1.43B.
MAIN16.531s/max58.008MB/13HTTP/3.261MB response bytes, zero tensor values.

408 source-sized native COST now failed;409 component attribution retained.
See [current resumption](LING_MINI_JOINT_FORMAT_NEXT_20261004.md).
Source sigmoid/group/bias/
top8 amplitude, GQA/QK norm/halfRoPE, SiLU/shared FFN/tokenizer/head need explicit
reference and numerical contract. Existing Switch loader D<=1024/vocab<=65536
cannot load it. A byte gate licenses no codebook quality or accepted model rate.

Byte-distinct pairs are not canonical function uniqueness/usefulness. A real384
selector or another-family converted artifact ultimately requires new excluded-
document original-relative whole quality and SAMEartifact FULL accepted rate.
No source quality inheritance. Useful>256/10x/~100B/physical DRAM remain open.
All405-411 sessions terminal; no412 controller/protocol/run yet.

## Another-family complete synthetic operator cost: format not eligible

[408](METH_408_LING_ADDITIVE_COST_RESULT_20261004.md) exact318 AVX2 direct
two-palette decode,20 actual-header-sized layers/top8/shared/dense/Q6 head:
ALL7 apparatus and independent math/archive gates PASS. All180 fullSHA and60
complete archives exact across3/6 profiles. Active503905280B/stored4969196544B;
all18 rep medians>20ms, neither14ms allowance passed. Not accepted model rate:
ALL values synthetic, independent context/layer fixtures omit causal attention/
RoPE/KV/cache/prefill/composition. Stop this format before values or fitting.

[409](METH_409_LING_ADDITIVE_COST_PROFILE_RESULT_20261004.md) inserts only timers/
counters, reverses exactly408, ALL10 gates PASS and complete bytes exact408.
Six-worker coded12.452-14.396ms/head6.695-7.733ms/router1.516-1.580ms; controls/
nonlinearities remainder2.224-2.682ms. No component alone exceeds14ms in every
six-worker rep. This informs a joint precision/cost hypothesis; it does not
repair408 or prove cache/DRAM behavior. MAIN22.281/22.203s, max~5.04GB each.
Default0ff9705 tail and qualified374/389 binaries unchanged.

[410 tested joint format](METH_410_LING_STATIC_PAIR_LUT_RESULT_20261004.md) reduces second
palette256->16, stores static precombined pairs, and changes Q6 head->Q4. New
representability/precision and cost, no inherited408 byte identity or original
quality. Its hypothetical conservative389.370MB descriptor charges active pair
tables,4.364GB total charges every stored table. Neither is an actual export or DRAM measure.410 ALL8 math/complete-output gates
PASS, but both14ms costs FAIL; all18 medians>20ms. Close this joint format before
source values/fitting. Source quality/rate never inferred. Useful RAM-scale
capacity and another-family proven transfer remain open.

## Another-family low-active geometry branch: actual Granite3.1 MoE headers

[411](METH_411_GRANITE_MOE_SOURCE_HEADERS_RESULT_20261004.md) ca13080/363ec2b:
actual pinned1B-named1.334625280B and3B-named3.298788864B original BF16 sources,
ALL6 config/index/header/offset/name/tokenizer/reference-module gates PASS.
218/290 names, two shards each, exact24x32/32x40 packed expert shapes/top8.
Shared tokenizer exact, not hidden-space/causal n equivalence. No finite values/
ordered function distinctness/actual original/native inference/quality/rate.

The ONE proposed all-row I8/A16/F32-scale target keeps all core/experts and full
I8 head, F32 router/norm and BF16 lookup. Original tied embed/head counted once;
target two precision copies charged. Actual descriptor433231872B at1.335B versus
892412928B at3.299B; latter unchanged format closed before values. Storage
1444581376/3469809664B is hypothetical, not an export or DRAM/latency result.
MAIN22.890s/42.353MB/22HTTP/4.239MB bodies, ZERO source-value bytes.

Original Granite math differs: selected-RAW-logit softmax/top8, SiLU packed
input split, GQA/full causal RoPE, embedding12/residual.22/attention.015625/
logits divisor6. Installed official4.57.6 modules stored/hash-bound NOT executed.
New source-specific original backend/cache/generation and native operator
contracts required; generic Switch compatibility and source quality not assumed.
Next [full-I8 cost/reference gate](GRANITE_MOE_FULL_I8_NEXT_20261004.md), planning
only. A tractable second-family case tests method generality and does not satisfy
or remove actual~10B/~100B/useful larger RAM-scale capacity requirements.
