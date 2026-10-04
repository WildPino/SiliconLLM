# Native expert scaling: research control index

**4 October 2026. Branch:** `research/native-expert-scaling`.
**Active research:** no artifact meets the full final goal.

## Goal and constraints

Transfer pretrained capacity to compact reusable core + selectively consulted
useful functions in `benchmarks/phase60/engine.c`. Preserve donor-relative
untouched reconstruction/prediction/generation/task quality; measure >=50
accepted batch1 tokens/s end-to-end on SAME artifact, 100 stretch. Demonstrate
multiple families/scales, real approximately10B/100B when resources permit.
User priority: useful n grows with RAM; CPU routing/LUT/real DRAM cost and
quality remain viable. Count distinct functions/labels/stored/active/read
bytes separately. Synthetic/copied capacity/component rates cannot close goal.
Freeze criteria before observation. Local80GiBRAM/Ryzen3600X/RTX3060;
T4 needs prior reason/budget/stop communication. No model job overlaps timing.

## Two research questions

| Question | Established | Missing |
| --- | --- | --- |
| Useful large-n target | Dense0.5B scoped E1280 functions preserve GPU quality; real conditional-bank cost apparatus available | Genuine larger donor knowledge/useful extra n; actual hierarchical route quality/RAM-scale consultation and full accepted rate |
| Pretrained-to-compact transfer | Complete276 diagnostic archive, qualified314 actual GigaChat MoE targets; complete14.664B Switch original weights and full native encoder/decoder mapping | Scoped363 whole quality PASS; accepted FULL rate FAIL364; cross-family/100B proof |

## Decisive evidence and closed paths

- [276-297 dense case](HISTORY_METH295_THROUGH299_20261003.md): original
  native297 unsupported41 versus E1280 40 FAIL. No unchanged promotion.
- [314 GigaChat source targets](METH_314_GIGACHAT_MIXTURE_TARGETS_RESULT_20261003.md):
  37,381 actual inputs/149,524 selected parent outputs; route/BF16 controls PASS.
  [316 full-mixture bound](METH_316_GIGACHAT_COMPLETE_MIXTURE_BOUND_RESULT_20261003.md)
  ALL primary/joint gates FAIL; unchanged fields closed.315 interruption retained.
- [319 full-width additive decoder](METH_319_FULL_WIDTH_ADDITIVE_CPU_RESULT_20261003.md):
  numeric/format controls PASS, 38-56ms versus14ms and variation FAIL. Unchanged
  decoder closed before book training.318 startup failure retained.
- [321 Switch applicability](METH_321_SWITCH_METADATA_SCREEN_RESULT_20261003.md):
  base128/256 have same123.765M decode matrix coefficients; F32 addressed
  497.537/499.896MB. BF16/W8 precision quality/rate unqualified. Encoder/cross-KV
  prefill and growing cache/attention costs charged. Large128 unchangedBF16 closed.
- [324 official isolated reference](METH_324_SWITCH_REFERENCE_RESULT_20261003.md):
  pinned4.57.6/Hub0.36.0 plus originalTorch2.6; model-free full cache/router gates
  PASS. Project5.13.1 remains unsupported for Switch, [323](METH_323_SWITCH_RUNTIME_DIAGNOSTIC_RESULT_20261003.md).
- [325 actual headers](METH_325_SWITCH_SOURCE_HEADERS_RESULT_20261003.md),
  [326 complete acquisition](METH_326_SWITCH_ACQUISITION_RESULT_20261003.md),
  [327 all actual tensors](METH_327_SWITCH_TENSOR_BINDING_RESULT_20261003.md):
  six original base256 archives+seven side files58,860,061,113B acquired/hash
  verified; all6392 finite F32 tensors bound; four tied aliases byte-equal;
  14,664,154,368 unique parameters; all12 banks each have256 distinct WI/WO
  parameter tuples. Parameter hashes do not prove effective/useful functions.
- [328 full native Tiny contract](METH_328_SWITCH_NATIVE_CONTRACT_RESULT_20261003.md):
  complete encoder/relative attention/cross-KV/autoregressive cache/capacity/
  probability/tied full head, both capacities and nine semantic faults PASS.
- [329 full original source](METH_329_SWITCH_FULL_SOURCE_RESULT_20261003.md),
  [330 stable RMS](METH_330_SWITCH_STABLE_RMS_RESULT_20261003.md),
  [332 all F64 matrix dots](METH_332_SWITCH_F64_MV_RESULT_20261003.md): ALL6392
  native mapped bytes exact, per-state/logit1e-4 and route/capacity/greedy controls
  PASS; original-backend probability1e-6 FAIL in case0 (1.2815/1.6391/1.4901e-6).
  None promoted under original frozen rules. Two consumed engineering inputs,
  no untouched quality or accepted-rate proof.
- [331 actual router trace](METH_331_SWITCH_ROUTER_DIAGNOSTIC_RESULT_20261003.md):
  outputs byte-equal330; both upstream states and same-input classification
  contribute. F64 router alone insufficient; no global unavoidable-floor claim.

- [333 reference primitive](METH_333_SWITCH_SPECIFIED_ARITHMETIC_RESULT_20261003.md)
  norm oracle FAIL before native/source observations, preserved.
- [334 prescribed arithmetic](METH_334_SWITCH_ROUNDED_REFERENCE_RESULT_20261003.md):
  explicit norm primitive rounding; NumPy oracles, both Tiny/all nine faults,
  ALL6392 source bytes and complete source C versus matched independent
  arithmetic PASS. Source probability max1.6391e-7. NEW numerical estimand;
  original329/330/332 FAIL unchanged. No original-donor quality/rate promotion.

- [335 export](METH_335_SWITCH_W8A8_EXPORT_RESULT_20261003.md) writes all6392
  compact arrays/14.818GB but final readback trips20min; failure retained.
- [337 readonly recovery](METH_337_SWITCH_EXPORT_RECOVERY_RESULT_20261003.md)
  checks1722 arrays before20min; full archive rehash consumes budget. Retained.
- [338 complete recovery](METH_338_SWITCH_TENSOR_RECOVERY_RESULT_20261003.md):
  ALL original6392 canonical tensor identities and actual target codes/F32/
  scales/padding exact;12x256 distinct target tuples;14,818,015,744B immutable
  compact payload verified.1032.625s/max25.246GB. Explicit fresh coefficient
  identity policy, no new ZIP-envelope hash. No useful diversity/quality/rate.

## Exact resumption

Complete original source at `results/native_expert_scaling/meth326_switch_base256_source`.
Compact payload at `results/native_expert_scaling/meth335_switch_w8a8_export/weights.bin`,
SHA `e0e5a940b0150b78d0080815a1fddd2a6b50f011351012ed5b4a48a88ba49056`;
manifest at `results/native_expert_scaling/meth338_switch_tensor_recovery/manifest.bin`.
338 raw committed08be22e;335/337 interrupted costs preserved. Qualified isolated
324 environment reusable; source128 full payload absent.

[336 native contract](METH_336_SWITCH_W8A8_CONTRACT_RESULT_20261003.md) PASS:
ALL6392 mapped target codes/F32/scales exact; extreme/Tiny/nine faults and full
C versus independent actual serialized I64/scaling reference EXACT on both
engineering cases.422.859s/max2.018GB. Original331 consumed diagnostic case1
has8 changed routes and1/5 greedy differences, so original quality unproven.
Dynamic n loader within RAM/shape bounds, only real256/Tiny2 qualified.

[339/340 cost](METH_340_SWITCH_W8A8_COST_RESUME_RESULT_20261003.md):339 physical
serialization binding stops before measurements; default body restored exact.
340 complete threads/outputs/counters pass but64-token prefill20.173ms and
repeat ratio1.1547 FAIL. Frozen unchanged recipe not promoted.
[341 exact batched prefill](METH_341_SWITCH_BATCHED_PREFILL_RESULT_20261003.md)
PASS all unchanged gates, all outputs byte-exact340/336. Actual6-thread median
full including prefill per32 forced positions7.643ms(source9)/17.510ms(source64),
decode6.652/8.442ms, repeats1.039/1.048. Logical matrix reads129.017MB/position,
not physical DRAM; one warmup/hash filesystem priming stated. Not accepted rate.
All335-353 observations committed;351 full quality FAIL retained30ca546. No timing overlap.

[342 source-only cohort](METH_342_SWITCH_FRESH_SPAN_MANIFEST_RESULT_20261003.md)
PASS, rawd2762b9/SHA bed91d901b26caf9b23992f1b4198d692e0ca770eeea60571caae9e2c884f89f:
24 disjoint new PG19book rows,36 old-row exclusions, four63-token windows/book,
96 known8-token span labels, encoder57/target11. No model scores at selection.
[343 actual prediction](METH_343_SWITCH_FRESH_PREDICTION_RESULT_20261003.md)
FAIL: original top1 agreement1001/1056=94.791667% versus95%; other seven gates
PASS, all-target NLL upper+.013893/mask NLL upper+.034667. All24books now
consumed. Frozen W8A8 closed before generation/task/accepted-rate promotion.
[344 head attribution](METH_344_SWITCH_HEAD_ATTRIBUTION_RESULT_20261003.md)
PASS all independent oracles on consumed states: HEAD-only activation16 yields
35 versus55 choice differences, same I8 weights/scales and unchanged upstream
states. Descriptive inference, not NEW quality or retroactive343 promotion.
[345 apparatus stop](METH_345_SWITCH_HEAD_A16_CONTRACT_RESULT_20261003.md)
preserved70908cf before compile/model. [347 actual numeric](METH_347_SWITCH_HEAD_A16_CONTRACT_RESUME_RESULT_20261003.md)
PASS all primitive/Tiny/nine faults/complete independent controls/ALL96 consumed
states and344 logits EXACT;152.266s/max3.170GB. [346 SAME binary cost](METH_346_SWITCH_HEAD_A16_COST_RESULT_20261003.md)
PASS unchanged6-thread gates: full7.452/15.372ms per32 forced positions,
decode6.426/7.207ms, repeat1.0174/1.0037;37.297s/max0.928GB. Not accepted rate;
64-source prefill~261ms remains significant for short natural responses.
[348 NEW manifest](METH_348_SWITCH_HEAD_A16_FRESH_MANIFEST_RESULT_20261003.md)
PASS24 source-only books/96 eight-token masks,62 old exclusions,47.171s.
[349 NEW original prediction](METH_349_SWITCH_HEAD_A16_FRESH_PREDICTION_RESULT_20261003.md)
ALL gates PASS:96.7803% original top1; all-token NLL upper-.007338/mask NLL
upper+.008128; known-token loss-.78125pp/lower-1.43229pp within-2pp. Strict
8-token exact0/96 BOTH arms still insensitive.1835.985s/max9.337GB. Source348
books now consumed; old343 FAIL unchanged, no cross-cohort causal improvement.
[352 consumed real-bank visits](METH_352_SWITCH_CONSUMED_ROUTE_COVERAGE_RESULT_20261003.md):
original encoder238-256/decoder208-231 experts/bank consulted; no larger-n/
causal usefulness/DRAM claim. [350 NEW multispan labels](METH_350_SWITCH_MULTI_SPAN_MANIFEST_RESULT_20261003.md)
PASS24 new books/96 four-two-token-mask cases, source29/target14,87 exclusions.
[353 greedy wrapper](METH_353_SWITCH_GENERATION_CONTRACT_RESULT_20261003.md)
PASS complete independent Tiny/original-official cache and actual engineering
trajectories; unchanged345 forward, explicitly new353 binary;73.235s/max1.896GB.
[351 NEW full task](METH_351_SWITCH_MULTI_SPAN_QUALITY_RESULT_20261004.md)
FAIL16/18 gates PASS, ALL96 teacher/natural cases: masked-only original top1
728/768=94.791667% versus95%; prose-edit upper.122823041 versus.10. Overall
96.205357%, original exact-field19.53125% and healthy88.541667%, native healthy
89.583333%; task signal/known-answer noninferiority PASS.1633.579s/max7.121GB.
Frozen1822b83, exec94685 exit0 fully consumed, raw/report30ca546. All350 sources
consumed;349 scoped prediction PASS remains, no unchanged generation/rate
promotion.354 draft ineligible/unexecuted.
[355 attribution](METH_355_SWITCH_MULTISPAN_ATTRIBUTION_RESULT_20261004.md)
ALL oracles PASS, retained3dc671b: original-state/SAME A16 head leaves8/768
changes versus40 actual; native F32 head leaves40.32 first natural divergences,
original-state target head recovers23; native F32 head4.163.812s/max1.895GB.
Upstream precision motivated, no changed whole-model quality claim.
[356 ALL A16 numerical contract](METH_356_SWITCH_ALL_A16_CONTRACT_RESULT_20261004.md)
ALL7 gates PASS,170.750s/max2.679GB, retainedb9c2dab; SAME338 target, full scalar/
Tiny/nine faults/forced/natural/cache/long/consumed independent arrays exact.
357 protocol-path failure before measurement retained23141ef.358 CPU6 repeat
FAIL1.100118992 retainedc5738ff;359 actual topology/consumed diagnosis PASS,
SMT pairs/minimum one-per-core[0,2,4,6,8,10].360 CPU6 physical-core-affinity
repeat FAIL1.136548 retainedf8db510. No unchanged optional repeats/relaxations.
[361 PRIMARYCPU1/one core cost](METH_361_SWITCH_SINGLE_CORE_COST_RESULT_20261004.md)
ALL6 gates PASS, SAME356 binary/338 target; every child affinity[0], actual
full8.1876/17.9727ms per32 forced position source9/64, decode7.0461/8.0331ms,
repeats1.0123/1.0285;24.969s/max.942GB. Cost margin, NOT accepted rate.
[362 NEW source-only cohort](METH_362_SWITCH_MULTI_SPAN_MANIFEST_RESULT_20261004.md)
ALL gates PASS24 NEW books/96 four-two-token-mask cases,111 prior exclusions,
source29/target14,34.797s; raw/reportc0302f3, previously unscored.
[363 whole quality](METH_363_SWITCH_ALL_A16_MULTI_SPAN_QUALITY_RESULT_20261004.md)
ALL18 PASS, original-primary unmodifiedF32CPU1/nativeALL A16CPU1/affinity[0],
ALL96 NEW source362 cases, retained3396ee2. Overall top197.2470%/masked96.4844%,
prose-edit upper.091532738<=.10; original/native health82/81 of96. Bounded
short English four-span task, not identical outputs/instruction chat. All362
sources consumed. [364 SAME full accepted rate](METH_364_SWITCH_SINGLE_CORE_ACCEPTED_RATE_RESULT_20261004.md)
FAIL retained93e7cac:45.1114 ordinary generated IDs/s INCLUDING markers,
book-bootstrap lower42.5362<50; prose-only24.6979/lower23.3052. ALL96 exact363,
81 accepted/895 ordinary IDs/490 prose, ALL96 full time19.8398s including
rejected cases; aggregate repeat1.005005 PASS. Encoder52.67%+crossKV7.44% of
whole time. SAME unchanged356 execution not promoted; no optional rerun.
[365 exact execution](METH_365_SWITCH_ENCODER_BATCHES_PROTOCOL_20261004.md)
NEW prospective encoder O/denseFF/F32router batches and four-token integer
weight-conversion reuse. Same338 payload/math/route order/capacity, new engine
opt-in binary. Freeze independent91 batch primitive/Tiny/nine-fault/full/
engineering/long controls AND ALL96 forced/natural full bytes EXACT363.
Execution-only exactness can inherit only scoped363 quality. Numeric PASS
licenses separately frozen366 SAMEbinaryCPU1 cost, then367 unchanged accepted
FULL >=50 lower/1.10 repeats/ALL96 acceptance, ordinary/prose explicit.
365 ALL9 numerical +ALL96 complete forced/natural bytes EXACT363 PASS,
retainedb906689/exec39812 exit0 fully consumed;369.281s/max2.774GB. Newexe
SHA4fbf5a411d8391cc7f7eb3355bbbf87d1111db3c17ffb620264f3d30590114ce.
366 SAME365 actual CPU1 cost FAIL repeat1.101982>1.10 onsource64, retained
c0611e7/exec84095 exit0 fullyconsumed; source9/64 full8.2531/19.2906ms per32
forcedposition, all othergates PASS. Closeunchanged365 cost/ratepromotion;
367 dependency-ineligible/unexecuted/unfrozen.363 scopedwholequality remains
valid via365 complete byteequivalence,364 actualaccepted rateFAIL unchanged.
[368 identity contract](METH_368_SWITCH_BANK_IDENTITY_CONTRACT_RESULT_20261004.md)
ALL7 PASS retained47820e8, exec3004 exit0 fullyconsumed;315.266s/max2.718GB.
SAME356 binary/math/CPU1profile/payload; actualsmallmanifest0/1/127 fixed
bijections all3072WI/WO pairs, ALL6392 lookupreadbacks/Tiny/full/OWNnatural
independentarrays exact. Matchedmanifest SHAexact338/matchedcontrols exact363;
actualconsultedfunctionidentitysidecarshashed. No bankharm observation yet.
[369 pairedbank usefulness](METH_369_SWITCH_BANK_USEFULNESS_PROTOCOL_20261004.md)
readytofreeze ALL96consumedmatched363 versusBOTHfixedoffsets, ALLfour primary
predictive+generative harm bounds98.75%/Bonferronifamilyalpha.05 before scores.
No new originalteacher loaded or acceptedtimingclaim. This diagnostic tests matched learnedbank usefulness, notnewuntouchedquality
or everyexpert/n-gain. Fixedgenuine64/128/256 nested-bank intervention planned
nextforadditional real alternatives; actual n artifactsnotyetimplemented.
Useful n/LUT/physical DRAM/causal learned bank
usefulness/real128 comparison/cross-family/~100B goal remains open.
Unexecuted old344 EOS-only draft is obsolete/unfrozen; do not substitute it.
Source128 full payload absent. Default engine exact; old GigaChat assets reused,
generic port paused. See [compact path](SWITCH_COMPACT_INTEGER_NEXT_20261003.md)
and [transfer route](SWITCH_REFERENCE_AND_TRANSFER_NEXT_20261003.md).

Genuine base128 to256 availability is2x, not10x; useful larger n, actual DRAM/
routing cost, accepted>=50 SAMEartifact and multiple-family/~100B remain open.

Future discriminating bank-usefulness controls and real scale/resource limits:
[useful-bank/n plan](SWITCH_USEFUL_BANK_AND_N_NEXT_20261004.md). Planning only;
365 must be frozen before execution; no overlapping model/native timing job.
