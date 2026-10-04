# M413 protocol: ONE exact four-row SIMD variant after412 cost failure

Freeze all science before FIRST compilation/native execution. No outcome yet.
M412 FIRST outcome frozen004411c/retainedb5ceacf: ALL8 apparatus/math gates PASS,
both3/6 joint14ms/repeat1.10 FAIL. Raw SHA
cb02f6413dfde4748872de1fefaf4df7d0462e03f41d2283451cbf3fcc6fe7bb.
Do not reinterpret or rerun its unchanged profile. Retain failure if413 fails;
no in-place repair. No source values, new quality data, network/training/GPU/T4.

## Specific new variable and stopping decision

Same I8/A16 arithmetic and FULL source0 geometry/weights/scales/controls/head/
fixtures/operations as [412 protocol](METH_412_GRANITE_I8_COST_PROTOCOL_20261004.md).
The qualified388 row kernel reloads identical A16 inputs once per output row.
New integer_dot4 loads each16-element A16 vector ONCE for FOUR contiguous rows,
uses four independent AVX2 I32 accumulator vectors, and reduces each into I64.
This can reduce redundant input loads and improve instruction parallelism;
performance is an untested hypothesis. It cannot reduce the I8 weight traffic.
No new precision, donor, warmup rule, worker profile, data or representation.

Tiles are four rows within the SAME bank; all output dimensions divisible4.
Input dimension, code range and I32 lane/globalI64 bounds unchanged: <=4096,
512*127*32767=2130641408<INT32_MAX per lane, global17045131264>I32.
Every output F32((F64(integer_sum)*F64(row_scale))*F64(activation_scale))
identical to412; head subsequently divided6 in same F32 order. Existing
head_integer_dot/head_activation_codes retained EXACT388 as independent bridge.

Controller removal of marked helper/test blocks, restoring ONLY the apply
function from412 and the new selftest call/diagnostic label, must recover the
ENTIRE412 C source exactly. Router/norm/SiLU/residual/lookup/head math, fixtures,
full archive schema/bytes and timed scopes otherwise unchanged. Do not copy a
different reduction/softmax/head or silently remove projection/control work.
All427819008 coefficients,2064384 row-scale bytes, full433231872B weight
descriptor, stored1444581376B and dynamic payload+outputs1443299328B unchanged.

Same gate: at leastONE actual3/6 physical profile must have ALL9 repetition
and24 fixed-input medians <=14ms AND pooled9-median max/min<=1.10, all apparatus
and complete outputs passing. No post-outcome exclusions, relaxed gates, warmup
expansion or additional repeats. PASS permits ONLY bounded source acquisition/
qualified original-reference/export/NEW whole quality protocols. FAIL stops
this Granite full-row-I8 kernel route before actual source weights; no extra
kernel tuning/attribution sweep. It does not exclude every source/representation.

## Independent new tile controls and full old controls

New four-row kernel tested on positive/negative4096-dimension extrema at all
four lanes, independent scalarI64 sums and deliberately truncatedI32 negative.
255 weight values x11 representative A16 extrema with mixed signs and31-input
SIMD/tail geometry:2805 cases x4 row lanes=11220 independent scalar comparisons.
All old412 scalar3848 active sampled rows,3266 bank/code/scale edges, additional
head64 relativeL2, all fixture activation codes, zero/ties/nonfinite/bank/code/
scale/headbit/globalI32 negatives and ALL24 scalar routers/selected-softmax/
sorted accumulation, independent norm/SiLU/residual/BF16*12 controls retained.
ALL numerical tolerances as412. New tile outputs compared with independent
scalar F64-to-F32 controls on every sampled matrix, not only qualified row dot.

ALL180 new executed complete output SHA and ALL60 full3515716B S412OUT1 archives
must exactly match ORIGINAL412 input-specific output SHA AND direct whole bytes.
No new archive magic; intentional byte identity is the scientific constraint.
ALL60 parsed through final head (lengths/shapes/IDs/input extents), and all
repeated/process/profile outputs exact. Prior412 full archives freshly SHA
verified before direct comparison; immutable raw is HEAD/physical-byte-bound.

## Identical schedule, controls and envelope

Balanced3,6,6,3,3,6 worker processes, each3 repetitions x10 fixed inputs0..9;
0/1 warmups per repetition, measured2..9. Inherited388 actual affinity3cores
[0,2,4],6[0,2,4,6,8,10], actual setup/event/end mask/group/thread readbacks.
One binding fault subprocess must exit2 with worker_affinity_readback, all
positive exits0 consumed before dependencies. No model/timing overlap; exact
two-argument pythonw.exe user daemon exception and OpenMP sanitation from412.
MAIN<=600s, each compile/native<=120s, combined sampled/native peakRSS<=6GiB,
output<=2GiB repeated guard. Compiler/runtime/topology/411 actual assets SHA
identical412; new binaries/directories/controller/protocol SHA retained.
Source0 source revision408b6e90baab8cf24f4aa9f8e19703ffa0a53b29; no actual values.

Opt-in SILICON_GRANITE_I8_FOUR_ROWS_PREFLIGHT; removal restores b5ceacf engine
exactly, default0ff9705 tail unchanged. Prior qualified Switch binaries intact.
Old HEAD-bound controllers reproduced at their freeze commits, not currentHEAD.

```
.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth413_granite_i8_four_rows.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth413_granite_i8_four_rows_result.json
```

All tensor values synthetic, independent layer/context fixtures, normF64 and
lowerID ties. Causal attention/scale.015625, RoPE/actualGQA, KV/cache/layer
composition/prefill/tokenization/generation/task/original parity/quality and
physical DRAM remain OMITTED. Time is a lower-bound screening operator stream,
not accepted batch1 token rate. Synthetic32-bank union is not source-function
identity/usefulness or useful>256/10x/~100B/RAM-scale capability. Small second-
family transfer prerequisite does not replace user large-useful-n priority.
Source-quality/backend/real weights required before model claims; SAMEartifact
whole quality and>=50 accepted end-to-end tokens/s remain final gates.
