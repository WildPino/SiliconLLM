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
| Pretrained-to-compact transfer | Complete276 diagnostic archive, qualified314 actual GigaChat MoE targets; complete14.664B Switch original weights and full native encoder/decoder mapping | Untouched whole quality and accepted rate and same-artifact rate; cross-family/100B proof |

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
[356 ALL A16 numerical contract](METH_356_SWITCH_ALL_A16_CONTRACT_PROTOCOL_20261004.md)
frozen5b8ba3c RUNNING authoritative exec17752,30min/16GiB, only model job.
All I8 core/expert/head inputs A16, SAME338 payload/router/lookup/attention;
new opt-in engine entry, old default body BYTE EXACT. Full independent scalar/
Tiny/nine faults/forced/cache/natural/long/consumed multispan controls required.
Consume exit/raw and preserve all. PASS licenses separately frozen357 SAME
binary actual forced-fixture cost, then NEW original-primary whole quality
with unchanged351 criteria, then SAME accepted FULL rate.357 draft unexecuted.
Unexecuted old344 EOS-only draft is obsolete/unfrozen; do not substitute it.
Source128 full payload absent. Default engine exact; old GigaChat assets reused,
generic port paused. See [compact path](SWITCH_COMPACT_INTEGER_NEXT_20261003.md)
and [transfer route](SWITCH_REFERENCE_AND_TRANSFER_NEXT_20261003.md).

Genuine base128 to256 availability is2x, not10x; useful larger n, actual DRAM/
routing cost, accepted>=50 SAMEartifact and multiple-family/~100B remain open.
