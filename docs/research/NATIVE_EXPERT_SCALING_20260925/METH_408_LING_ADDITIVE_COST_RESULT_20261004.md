# M408 result: Ling full-width additive format rejected on CPU cost

Frozen science: `fb897e9`; first outcome, no repairs or selected reruns.
[Protocol](METH_408_LING_ADDITIVE_COST_PROTOCOL_20261004.md),
[controller](../../../benchmarks/native_expert_scaling/meth408_ling_additive_cost.py),
[C operator stream](../../../benchmarks/native_expert_scaling/meth408_ling_additive_cost_cpu.c),
[raw](meth408_ling_additive_cost_result.json).
Raw SHA256 `094b080e2d5f0b890878153d8a2507418395039ad6c68ee4fa44d7497ad0dee9`.
Binary SHA256 `a6e6ad659aff2cc7e76022d5999d6fd80a0462a6264afca2bbdc0381d335e2ff`.

## Decision

Reject THIS full-width two-palette additive representation/kernel for Ling-mini
before acquiring its 32.5GB source values or fitting books. All seven apparatus
and numerical/archive gates PASS, but neither declared physical-worker profile
passes the frozen <=14ms operator screen. All eighteen rep medians also exceed
20ms. These are synthetic operator-stream costs, not accepted token rates.
No inference from this result to all representations or pretrained Ling quality.

| Profile | Nine rep medians, ms | Max/min | Repeat <=1.10 | All rep/fixed-input medians <=14ms |
| --- | --- | ---: | --- | --- |
|3 physical [0,2,4]|40.777,38.245,37.691,39.438,39.037,37.846,37.475,37.338,37.531|1.092090|PASS|FAIL|
|6 physical [0,2,4,6,8,10]|39.412,32.987,30.871,24.989,24.377,24.841,26.095,23.719,24.358|1.661626|FAIL|FAIL|

The six-worker first process is much slower than the later two; retain it in the
pooled repeat result. No outlier deletion, frequency attribution or replacement
run. Three-worker stability does not rescue its cost failure.

## Scope, source geometry and accounting

Reuse actual M407 pinned header geometry: inclusionAI/Ling-mini-2.0 revision
`a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`,20 layers/D2048,19 banks256/top8,
FF512 plus shared512, first dense5120, GQA16/4/128, vocabulary157184. ALL values
in M408 are synthetic, including head/router/norm/embedding. No actual Ling
weights, learned books, source-relative quality or useful added experts.

| Addressed per operator execution | Bytes |
| --- | ---: |
|Two-index codes,779,091,968 active coded coefficients|194,772,992|
|Row scales|2,560,000|
|Active palettes|2,277,376|
|Full F32 routers and bias|39,865,344|
|Complete Q6_K head|264,069,120|
|Other control weights|356,352|
|One BF16 embedding row|4,096|
|Complete unique descriptor|503,905,280|

All180 executions reconcile exact active counters. Stored coded positions
15,601,762,304; hypothetical complete source-sized storage4,969,196,544B;
allocated dynamic codes/scales/books/head/embedding/matrix outputs4,931,534,848B.
Static control arrays/runtime overhead separately reflected in RSS. All banks
allocated and filled. Addressed bytes are not physical DRAM traffic.

This reuses M318 direct AVX2 two-I8-palette decoding and four-row integer dot
math. Its palette is a static lookup; it does NOT test M398/400 per-input
activation-LUT optimization. Independent layer/context fixtures prevent a
complete causal model interpretation. All coded attention projections, shared/
dense/routed matrices, QK norm, sigmoid group router, mixtures and head execute;
causal attention, RoPE, KV/cache, prefill, tokenization, composition, task and
generation remain absent. Omitted mandatory work would require additional cost.

## Numerics, output and placement controls

Every positive process passes4448 independent sampled integer rows,29384 code
and palette bank-offset edges,32131 exhaustive decoded/input cases,128524 tile
lanes and14641 mixed adjacent pairs. Integer relative L2
`3.638331237912454e-8`;64 independently decoded Q6 head rows relative L2
`1.2120883801471723e-7`. Extrema/nearest-even/zero/nonfinite/unsupported-S8,
row-bias/palette/code/head-bit and invalid-bank negatives detected. BF16 lookup
all2048 values exact.19 group-router reference checks pass: biased group and
expert selection, unbiased selected sigmoid normalization times2.5. This is
synthetic fixture correctness, not qualified Torch original parity.

Balanced process order3,6,6,3,3,6, each three repetitions of ten inputs (0/1
warmup). ALL180 complete-output SHA256 values match;60 archived complete streams
(first repetition, every input/process) directly byte-equal across profiles.
Archives include all matrix outputs, QK norms/mixtures, routes/probabilities/
gates, head input and all logits. Fresh post-run archive hashes checked again.
This is stronger than the64-bit fingerprint-only319/398-400 comparisons.

Actual388 worker setup/end/event Windows thread IDs/masks/groups pass and
match fresh374 physical topology. Negative worker fault exits2 before allocation.
OMP ACTIVE/infinite, dynamic FALSE, max active levels1, KMP affinity none,
PROC_BIND absent. Two exact unrelated tiktok_publisher user daemons recorded and
untouched; no other model overlap. Windows group0 is not cache/DRAM evidence.

## Reproduction and resource cost

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth408_ling_additive_cost.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth408_ling_additive_cost_result.json
```

Use a fresh checkout of frozen science and fresh output locations; controller
refuses existing output. Exact helper/compiler/runtime SHA identities in raw.
Clang21.1.8, AVX2 x86-64-v3, no fast-math/contraction, OpenMP. Old engine branch
reversal exact02afce0 and default tail0ff9705. Opt-in408 prefix only; qualified
Switch binaries remain separately bound to their original quality/rate evidence.

MAIN22.281s; maximum checked combined RSS5,035,597,824B;
new output217,142,285B; per-native peak roughly4.982GB. All process handles
terminal exit0, including the wrapper; injected negative expected exit2. No
network/source value acquisition/training/GPU/T4. Serialization/cryptoSHA,
initialization/team/fixture preparation/cleanup excluded from operator timer;
activation quantization/routing/projections/FF/head included.

## Remaining uncertainty

The aggregate failure does not identify whether coded matrices or the large
vocabulary head must change first. A separately frozen component attribution
with complete-output byte bridge can decide whether a head-only or coded-only
change could plausibly satisfy the budget. Instrumentation is diagnostic and
cannot establish a precise causal slowdown relative to M408. Full-width fitting
and useful-n expansion remain stopped on this path pending a joint cost and
source-quality hypothesis. Final project goal remains incomplete.
