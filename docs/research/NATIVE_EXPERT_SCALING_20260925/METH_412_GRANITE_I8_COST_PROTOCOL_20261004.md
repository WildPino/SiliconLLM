# M412 protocol: complete Granite source0 row-I8/A16 CPU cost prerequisite

Prospective science: commit C/controller/protocol/engine opt-in before FIRST
compilation or native execution. First outcome is retained even on failure;
repairs require a new experiment. No source values, fit, network or quality data.

## Uncertainty, reused evidence and decision

M411 actual Granite source0 headers/config/index/tokenizer, frozen ca13080 and
retained363ec2b, establish the geometry and a433231872B complete hypothetical
row-I8 descriptor. Raw SHA a3260f95f5d93b4fd457487dda9a586d1c7dbc0d50596176f84042d4e0cb76bc.
This is another-family tractable conversion prerequisite, not a replacement for
real~10B/~100B/useful RAM-scale experts. Original Switch7.415B/14.664B results
and closed Ling408/410/Giga formats remain unchanged.

Test whether ALL mandatory matrix/control work of this candidate leaves room
within the20ms target interval. PASS ONLY if at least ONE actual3/6 physical
worker profile has BOTH: every one of its9 repetition medians and24 fixed-input
medians <=14ms, and max/min of its9 repetition medians <=1.10. No exclusions,
threshold changes, extra repeats or profile tuning after outcome. Numerical,
placement, whole-output and resource controls must also all pass. PASS licenses
bounded real-source acquisition/reference/export/NEW quality protocol ONLY.
FAIL stops this exact full-row-I8 cost profile before source values; it does not
prove every representation or donor family impossible. Apparatus failure is
retained before a separately numbered repair. No accepted rate inferred.

## Exact source and prospective target

Source ibm-granite/granite-3.1-1b-a400m-base,
408b6e90baab8cf24f4aa9f8e19703ffa0a53b29, ALL218 BF16 names/two headers in411.
L24,D1024,H512,E32,K8,attention heads16/KV8/HD64,V49152.
Per-layer Q[D,D], K/V[512,D], O[D,D], all32 packed gate/up[1024,D] and
down[D,512]; selected8 complete banks. Full head[V,D], BF16 lookup[V,D].
All24 full F32 routers[32,D], two F32 norms/layer and final norm, epsilon1e-6.
No bias, QK norm, shared/dense expert or grouped/sigmoid routing.

| Complete descriptor | Bytes |
| --- | ---: |
|Core plus selected experts I8|377487360|
|Their F32 row scales|1867776|
|Full head I8|50331648|
|Head F32 row scales|196608|
|Full routers F32|3145728|
|Norms F32|200704|
|One BF16 lookup row|2048|
|TOTAL|433231872|

Total I8 coefficients executed427819008; scales2064384B; stored I8
1333788672B. Complete hypothetical stored descriptor1444581376B. Source tied
embed/head counted once as source knowledge; target separate BF16/I8 copies
both charged. Actual dynamic payload plus all active matrix outputs1443299328B
= stored minus static router/norm3346432 plus active matrix output2064384.
Static structs, activation/code workspaces and allocator/runtime overhead are
additional, covered by RSS. These byte counts are format budgets, NOT measured
physical DRAM traffic or cache residency. No source function uniqueness inferred.

Reuse EXACT388 head_integer_dot and head_activation_codes including bounds,
integer arithmetic and F64-scale-to-F32 order. Controller source-region equality
includes both definitions, excluding earlier forward declarations. The SIMD
dot uses I32 lanes and I64 global reduction; for cols<=4096 each lane has at
most512 products,512*127*32767=2130641408<INT32_MAX, global17045131264>I32.
Weight range[-127,127]; activation[-32767,32767], F32 absmax/32767 scale,
nearest-even of F32 division, zero scale1. Final row output
F32((F64(integer_sum)*F64(row_scale))*F64(activation_scale)). Target source
conversion would use absmax/127 nearest-even weights; NO actual conversion here.

## Synthetic fixture and timed operations

Every stored matrix code initialized from little-endian64-bit mix(seed+byte/8),
signed byte-128 mapped to-127; independent arithmetic expected-code reader.
Each positive finite F32 scale is
(1+(mix(row+seed)&127)/128)/(127*sqrt(input_dimension)). Distinct deterministic
seeds for each layer/matrix. Full BF16 lookup is upper16 bits of the SAME
synthetic I8-head decoded fixture, not an acquired original embedding. All
stored positions physically allocated/filled; first/last8 bytes and first/last
row scales checked in every bank/matrix. No virtual expert duplication/quality
claim. Full router/norms initialized from explicit seeded fixtures.

Input-ready independent residual/context fixtures per layer; layer0 residual
is BF16 lookup*12. EACH of24 layers executes attention norm->A16->ALLQ/K/V,
context A16->O, residual ra+.22*O, FF norm/A16/full32-logit F32 router,
top8 then selected-logit softmax, packed gate/up, SiLU(firstH)*secondH,
eight A16 quantizations, down, sorted-ID weighted sum, residual rf+.22*mixture.
Final layer residual->final norm/A16->ALL49152 head rows->F32 logits/6.
Router dot is exact generic F32 products summed in F64 then F32 for the
fixture; source Torch accumulation/backend is NOT asserted equivalent.
Top-logit ties choose lowerID; selected softmax summed in top-logit order,
then gates/IDs sorted by expertID for accumulation. Original Torch ties still
need a reference contract. Norm uses F64 squares/reduction->F32 inverse.

Timer includes all above math, full activation validations/quantizations,
OpenMP launches, routing, nonlinearities, mixture and head. Timer EXCLUDES
startup/team setup/allocation/fills, preparing fixtures, selftests, SHA/archives,
finite output checks/printing, descriptor assertions and process cleanup.
CAUSAL ATTENTION, attention multiplier.015625, RoPE, actual GQA assembly,
KV/cache, layer-to-layer composition, prefill, tokenization, generation and
task scoring are OMITTED. This lower-bound cost screen cannot establish50
accepted batch1 IDs/s, source parity, pretrained quality or physical DRAM.
All source multipliers remain mandatory in any later complete reference.

## Independent controls and complete outputs

- I64 scalar sum exact on8 spread rows in EVERY active bank/matrix including
  head:8*(24*(4+2*8)+1)=3848 rows. F64 scaled relativeL2 <=1e-6,
  head/divisor control on another64 spread rows <=1e-6. ALL outputs finite.
- EVERY stored bank/matrix first/last code+scale checks:
  2*(24*(4+2*32)+1)=3266. Full pool fills are executed before timings.
- I64 extrema at4096, positive/negative bound and deliberately truncated I32
  negative;255 weight values x11 representative A16 extremes=2805 mixed-sign
  cases at31 dimensions, testing SIMD and scalar tail. No exhaustive allI16 claim.
- Zero/nearest-even ties/NaN/Inf, invalid bank, code mutation, row scale mutation,
  head-bit mutation and exact original388 primitive bridge. Quantize wrapper
  rejects nonfinite before the fatal qualified primitive. Negative scale checks
  the declared row formula; code/head negative invokes native dot and head cast.
- ALL11 activation inputs/layer and head checked by independent floor/parity
  rounding of specified F32 quotient; scalar F32-product router logits,
  independently sorted top8/softmax/bank bindings/tie fixture onALL24 layers.
- Independent long-double norm error <=1e-6; F64 SiLU error <=1e-6*max(1,abs(ref));
  packed-half SiLU product and exact sorted residual/mixture F32 checks; BF16*12.
- Worker actual masks/readbacks at setup/event/end, one actual binding fault
  subprocess must exit2 with worker_affinity_readback; positive processes exit0.

Complete binary archive S412OUT1, little-endian IEEE F32/I16/U32, exact3515716B:
8-byte magic,7U32(token,D,L,E,K,V,H), embedding_row[D]; then EACH layer's six
matrix blocks(shape5U32,activeIDsU32,ALL active outputF32), ra/context/xa/rf/xf/
mixture/residual[D] each, zg[K*H], parent[K], scores[E], gates[K]; qa/qo/qf,
then8qz input blocks(nU32,scaleF32,ALLnI16); finally head_residual/head_input[D],
qhead input block, head matrix block INCLUDING full divided logits.
No struct padding serialized. ALL180 executed complete outputs cryptoSHA;
rep0 ALL60 full archives independently parsed for lengths/shapes/IDs/input
extents and directly compared byte-for-byte across processes/profiles. All
repeated complete SHA per input must equal, not a64-bit-only fingerprint.

## Placement, resource envelope and immutable identity

Balanced process sequence3,6,6,3,3,6 workers, each3 repetitions x10 fixed
input IDs0..9. IDs0/1 declared warmup, IDs2..9 measured. Same fixtures across
profiles. Each profile has9 rep medians and24 per-process fixed-input medians;
repeatability pooled over the9 medians. Selected union counters <=32 and >=8
per layer, not useful32 or source route coverage.

Reuse388 physical placement3cores[0,2,4],6[0,2,4,6,8,10], frozen374 topology
and compiler/OpenMP identities. SHA374
4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1.
Sanitize inherited OMP/KMP/GOMP/worker fault environment; retain374 runtime
settings. Before every native process exclude all nonancestor Python/meth.exe
jobs except exact two-argument pythonw.exe with argv1 absolute
D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py (user daemons).
MAIN<=600s, each compile/native<=120s, combined sampledRSS<=6GiB and native
peakRSS<=6GiB, output<=2GiB checked repeatedly. No GPU/T4/training/network.

Bind physical filtered-HEAD C/controller/protocol/engine/helpers/388 source,
411/374 raw results and all source0 assets/headers plus both stored official
modules. Fresh output directory, never overwrite. Engine opt-in ONLY
SILICON_GRANITE_I8_COST_PREFLIGHT; reversal must equal3f87d37 engine, default
0ff9705 tail exact. Qualified Switch binaries/weight artifacts untouched.

Command after freeze:

```
.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth412_granite_i8_cost.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth412_granite_i8_cost_result.json
```

Retain first raw success/failure and report with actual hashes, commands,
resource costs, all profile medians and decision before changing science.
If PASS, next separately frozen source-value acquisition ALL2.669GB/LFS SHA/
finite/source-identity and qualified original operator reference, then exact
declared export and NEW excluded prediction/generation/task controls and SAME
artifact complete accepted rate. Useful>256/10x/real~100B/physical DRAM and
another-family quality remain open until real end-to-end evidence.
