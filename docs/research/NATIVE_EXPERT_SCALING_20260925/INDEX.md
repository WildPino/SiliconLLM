# Native expert scaling: research control index

**3 October 2026. Branch:** `research/native-expert-scaling`.
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
All335-342 observations committed.343 quality job live; no timing overlap.

[342 source-only cohort](METH_342_SWITCH_FRESH_SPAN_MANIFEST_RESULT_20261003.md)
PASS, rawd2762b9/SHA bed91d901b26caf9b23992f1b4198d692e0ca770eeea60571caae9e2c884f89f:
24 disjoint new PG19book rows,36 old-row exclusions, four63-token windows/book,
96 known8-token span labels, encoder57/target11. No model scores at selection.
[343 prediction protocol](METH_343_SWITCH_FRESH_PREDICTION_PROTOCOL_20261003.md)
freezeddd0107; RUNNING under authoritative exec59475. Fresh ALL original
source canonical coefficients+original331/native336 bridges BEFORE NEW
scores; original unmodified CPU1 PRIMARY versus native341CPU6 SAMEartifact.
40min after imports/48GiB. Do not duplicate/restart. If PASS freeze free span
generation/health/reconstruction task and accepted batch1 rate. If FAIL preserve
raw/partial NEW sources before specific diagnostics or changed candidate.
Source128 full payload absent. Default engine exact; old GigaChat assets reused,
generic port paused. See [compact path](SWITCH_COMPACT_INTEGER_NEXT_20261003.md)
and [transfer route](SWITCH_REFERENCE_AND_TRANSFER_NEXT_20261003.md).

Genuine base128 to256 availability is2x, not10x; useful larger n, actual DRAM/
routing cost, accepted>=50 SAMEartifact and multiple-family/~100B remain open.
