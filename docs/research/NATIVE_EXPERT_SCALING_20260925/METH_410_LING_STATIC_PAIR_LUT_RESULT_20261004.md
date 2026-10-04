# M410 result: smaller static pair LUT/Q4 joint format still fails CPU cost

Frozen science75649d7; FIRST outcome, no repairs/tuning/reselected runs.
[Protocol](METH_410_LING_STATIC_PAIR_LUT_PROTOCOL_20261004.md),
[controller](../../../benchmarks/native_expert_scaling/meth410_ling_static_pair_lut.py),
[C](../../../benchmarks/native_expert_scaling/meth410_ling_static_pair_lut_cpu.c),
[raw](meth410_ling_static_pair_lut_result.json).
Raw SHA256 bd345a1b3239c5f319432541059006fb94cb6a002593e52d80890a0058707f87.
Binary SHA2567f90505527103016cf30d0032accb4019c864357086296ea89283a4b3c92f43b.

## Decision and observations

ALL8 apparatus/math/archive gates PASS. Reject THIS256+16/static pair-table/
packed3bytes16/Q4-head joint format before Ling acquisition/fitting. No optimized
profile meets the frozen14ms operator allowance; all18 rep medians exceed20ms.
No broad rejection of other representations or Ling pretrained quality.

| Mode/profile | Retained rep medians, ms | Max/min | Repeat<=1.10 | ALL rep/fixed medians<=14ms |
| --- | --- | ---: | --- | --- |
|Direct reference/3 physical workers|34.506,32.674,31.956|not a pooled profile|unqualified|unqualified|
|Static pair/3 workers [0,2,4]|32.850,32.662,32.238,32.777,32.545,33.425,32.495,33.111,32.609|1.036798|PASS|FAIL|
|Static pair/6 [0,2,4,6,8,10]|31.803,31.390,21.280,22.839,20.892,20.952,22.942,20.970,21.883|1.522247|FAIL|FAIL|

Retain slow first six-worker process. Do not attribute it to clock/cache,
delete it or retry to get a repeatability pass. Some fixed-input medians fall
below20ms, so the ALL18>20 statement applies to REP medians only. Do not report
accepted model tokens/s or a precise causal speedup relative to408/409.

## New representation, complete charge and source limits

ALL values synthetic. Geometry407 actual-header-sized Ling-mini20 layers/D2048,
19 banks256/top8/routed/shared512, first dense5120, fused GQA3072x2048,
untied157184x2048 embedding/head.65536->4096 vector-code pairs changes
representability; Q6->Q4 changes head precision. No real weights/fitted books,
original-reference/local source quality/useful experts. Old408 output differs
by design; the numerical bridge is NEW direct versus NEW pair mode.

| Conservative active descriptor component | Bytes |
| --- | ---: |
|Codes for779091968 coded coefficients|146079744|
|F32 row scales|2560000|
|Original256+16 books|1209856|
|ALL556 active static pair tables|18219008|
|Full Q4_K head|181075968|
|F32 full routers/bias|39865344|
|Other F32 controls|356352|
|One BF16 lookup|4096|
|TOTAL|389370368|

Raw addressed_descriptor_bytes is this declared conservative format budget,
NOT measured actual per-mode addressing: books and tables are both charged
though each execute mode uses one. ALL14692 stored tables precomputed,481427456B;
complete stored4364312064B, dynamic payload+outputs4326650368B. All stored
coded positions15601762304; no fit/export or DRAM traffic claim. Each execution
reconciles all active coded/scales/books/tables/router counters. Lookup table
construction occurs once in measured setup, never hidden per-token work.

Same complete408 operator ordering/F32 router/quantizer/norm and geometry;
all attention projections/QK norm/routed/shared/dense/mixtures/head execute.
Independent per-layer/context fixtures omit causal attention/RoPE/KV/cache/
composition/prefill/tokenization/generation/task. Lower-bound operator screen,
not model quality/rate. Full head is scored, not a candidate shortlist.

## Independent numerical controls and complete-output bridge

All32131 decoded/input cases,128524 four-row lanes,14641 mixed adjacent pairs,
extrema/zero/nearest-even/nonfinite/unsupported-S8 controls pass.4448 independent
sampled integer rows/bank,29384 all-bank code/book/table edge checks pass.
Integer relativeL2 3.728536217504576e-8.64 independently F64-decoded Q4 head
rows relativeL2 7.032936593138448e-8. Source error unknown; these errors describe
the synthetic quantized format's arithmetic, not loss versus real Ling.

ALL4096 code A/B combinations/131072 packed lanes,32768 local table-entry lanes,
ALL4096x8 entries for214 first/last checked bank instances=7012352 lanes pass.
Low/high code nibbles, pair-table value, stale book/table inconsistency, first
book code, row-bias, invalid bank and Q4 head-bit negatives detected.19 full
group-router independent reference checks pass; BF16 lookup2048 bits exact.

One3-worker direct process plus balanced pair processes3,6,6,3,3,6, each3 reps
of ten inputs0/1 warmup,2-9 measured. ALL210 complete-output SHA agree;70 complete
first-rep archives directly byte-equal across BOTH modes/profiles/processes.
All matrix outputs/QK norms/mixtures/routes/probabilities/gates/head input/full
logits archived. Fresh post-run hashes verify all70 new archives and bindings.
This does not prove source/pretrained/composed-model quality.

## Resources and reproduction

MAIN21.984s/max checked combinedRSS4430258176B/new output253083347B;
per-native peak~4.377GB. All positive processes/wrapper terminal exit0,
injected binding-negative expected exit2 before allocation. Fresh actual388
setup/event/end Windows worker masks/topology374, ACTIVE/infinite/dynamicFALSE/
KMP affinity none/PROC_BIND absent. Two exact unrelated user daemons recorded
untouched; no model overlap. No network/actual values/training/GPU/T4.

Command in protocol; frozen75649d7 checkout/fresh output locations required.
All helper/compiler/runtime/source hashes in raw. Engine old409 reversal
45c2386, old40002afce0/default0ff9705; opt-in410 only. Existing qualified Switch
artifacts/binaries remain separately authoritative. Setup/all-bank table filling/
fixture preparation/cryptoSHA/archive/printing/cleanup excluded from operator
timer; all actual quant/routing/matrix/FF/head included.

## Next research decision

Do not fit this rejected format or repeat its cost with deleted observations.
Another minor packing/head precision change needs a stated new cost/quality
hypothesis. This failed gate adds no useful n/10x/physical DRAM/another-family
transfer. Investigate a pretrained source with smaller mandatory active geometry
as a credible second-family transfer case, using actual operator/metadata and
complete cost prerequisites first. Do not resume a generic high-cost donor port
or promote synthetic capacity as knowledge transfer. Goal remains incomplete.
