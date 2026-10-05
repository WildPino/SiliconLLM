# METH-454: actual C primal and cost for compact WI /exact sparse WO

5 October 2026. PROSPECTIVE. New C/controller uncompiled/unimported at writing.
Freeze both sources and protocol before first compilation/import/numeric work.
ONE controller execution, first failure retained before any NEW numbered repair.
Goal ACTIVE/INCOMPLETE;453 local success is not whole-model quality or rate.

## Decision and bindings

Resolve whether direct packed-I4 WI or adjacent-activation pair-LUT WI followed
by exact zero-only native I8 WO improves actual selected-FFN cost. Keep ALL453
weights/scales/signs; no new encoding/rounding/fit/data/threshold/seed. Two fixed
kernel variants, separately gated; no expansion based on outcomes.

Bind453 raw SHA256
`63d0fca69ad180b5024e1152b3d2fd9e13699fcfc3efd662850e011387ba3bbf`,
retention453 `75c7ee7e36d7632d40c8f384308efec79e0ca833f7954c971dafddec04603d93`,
374 raw `4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1`.
Fresh all453 helper/protocol hashes and TEN complete output hashes, original
source128 whole payload7,541,946,880B/manifest380/tensor parser, engine54194c36...,
original374/389 binaries and numerical runtime DLLs. Original source/manifest and
engine are unmodified. Reusing components never reruns previous controller mains.

Original C baseline is INCLUDED UNCHANGED from374's qualified physical-worker C:
source SHA16aa50a697d99ddabd158b6cc840e1279bf25f4208ea72a295d58634bbe12cfa,
thread helper SHA0b82fd9baf244bdf20cc30caaf254084371a9e7c1a46b70fb170f8a9dfb61bd4.
Fresh bind its four source entries and374 math reversal/gates. Use original `mv`,
`head_activation_codes`, `head_integer_dot`, model/manifest loader and worker setup;
no new slow source surrogate. One-worker original OpenMP region/quantizer/shape
checks are charged as in that primitive. Original full3worker rate is separate.

Compiler SHA8ba7ddd7fce5574275dec0302caa1eb9f3d9ae812c6e1f44276a27b6d14fb9b7;
libomp SHA6fc163dd513538a92a187d987438bebc5509afd1f824b20a88dd51463b3d5698.
Compile -O3 -std=c11 -march=x86-64-v3 -fno-fast-math -ffp-contract=off -fopenmp
and -lpsapi; preserve original runtime ACTIVE/none/infinite/FALSE/one-level settings.
One physical CPU0, one OpenMP/Torch/BLAS thread. No concurrent model jobs; preserve
only exact approved publisher pythonw two-argument invocation. No GPU/download.

## Exact layout/export, no learned or coefficient changes

Serialize ALL128 original453 functions to one readonly mapped C bank:

    4096B header MC454B01 /D768/M3072/E128/header4096
    768 shared WI signs
    for each original expert:
        WI packed1179648B, WI block scales147456B,
        WO original columns2359296B, WO original row scales3072B

Stride3,689,472B; expert0 offset4864; every field and column64 aligned. Total
472,257,280B. Independently recover every saved field/expert byte from the actual
file; C verifies mapped alignment, all I4 non--8 nibbles, I8 non--128 codes and
finite positive scales. No matrix omission/unpacking/expansion as deployment storage.
Map complete original model and new real bank; report mapped/resident/work bytes.

Trace MC454T01 contains336 ORIGINAL normalized F32 inputs, original IDs/books18..23,
shape768/3072/128/336/bank11. Each record3080B, file1,034,908B. Source normalization,
router/probability and upstream state remain fixed. These are consumed prefixes;
the cost screen and forced-ID primal are not fresh evaluation or n scaling.

## Native arithmetic

Signed block256 Walsh F64 in place, divide16/castF32, original A16 quantizer.
Compare all actual C transformed inputs/codes/scales to an independent explicit
Walsh-checked Python reference. Then private original-neuron-coordinate ReLU.

DIRECT WI: decode signed nibbles into AVX2 I16, exact I32 dot per block64. Bound
64*7*32767=14,679,616<2^31; F64 weighted scales summed in ascending block order,
F64 activation scale multiplied last, castF32. No FMA/reassociation.

PAIR-LUT WI: every call builds384*256=98,304 I32 entries (393,216B), one table for
each adjacent pair of actual A16 codes. Entry q=low+16*high decodes signed
coefficients[-7,7]; products and pair sum exact. Reserved nibble8 entriesINT32_MIN;
whole bank validation rejects them BEFORE release lookup, separately qualify fault.
Lookup all1,179,648 packed pairs/query. Fixed block-major loop12 blocks then3072
rows,32 lookups/block; I32 partial and SAME ascending F64 weighted output. All
tables built each query, every lookup/add/read/write charged; no cached-input reuse.

WO: original A16 quantizer on actual ReLU, scan all3072 codes into increasing U16
nonzero indices. Sum exact column products in I32 partials of<=512 nonzero columns,
bound2,130,641,408<2^31, convert/combineI64. Full dot may exceed I32. Unchanged
F64 row scale THEN F64 activation scale, castF32. Actual columns768 contiguous I8
bytes, no WO rotation. No approximate omission, teacher mask or epsilon clipping.

Timing charges basis, both quantizers, WI decode OR table construction/lookups,
block sums/scales, in-place ReLU, actual scan/indices, WO partials/reduction/scales/
output. Primal capture copies/extra reference quantizations occur only in untimed
qualification. Bench passes NULL capture pointer; source/candidate ReLU in place.

## Apparatus qualification BEFORE timing

Native tiny: all225 valid coefficient pairs times9 signed/zero activation pairs
(2025 direct-dot checks); all98,304 table entries independently scalar-qualified
including unused sentinels; explicit -8 fault rejected in separate child exit2;
zero activation scale1; full3072 signed-extrema/alternating/zero/single-column WO
against FULL scalar I64 integer/scaled oracle (five cases); explicit dyadic Walsh
parity sums. Existing source A16/I8 operators stay qualified and unchanged.

C then reconstructs original336 functions and BOTH candidate kernels at correct/
ID+1 modulo128 (672 each). All1680 raw/down states, basis/A16 codes/scales, IDs and
nonzero counts must be BYTE identical to original captures and453 respectively.
Python reference actually verifies336 basis inputs/quantizers; all1344 candidate
records compared, no subsampling. Primal MC454Q01,26,140B/record,43,915,224B file.
No head recomputation here: exact local FFN output inherits the same453 prefix
suffix only under those fixed controls, not new C whole model/changed routing.

Only after Python confirms every primal record, launch native timing. ALL eight
apparatus gates must pass: fresh bindings, complete128 wire recovery/alignment,
tiny cases,336 original records,1344 candidate records, fault,7056 exact timed
outputs/fixed trace/order and independently reported physical-worker affinity.

## Fixed timing experiment and prospective cost gates

Two warm WHOLE336 trace sweeps per arm; five measured WHOLE336 sweeps per arm.
Loop repetitions[-2,-1,0,1,2,3,4]. Arm order rotates by(rep+2)%3. Each arm completes
ALL336 queries BEFORE the next arm. No same-query direct/LUT interleaving, hot
single-function timing, empty loop substitution or claimed cache residency.

Every query measured with QPC around full FFN. After timer, memcmp down to already
qualified primal and update volatile checksum. Retain all7056 warm/measured query
times, IDs/books/order; validate exact ordered tuple list. No disk output/reference
memcmp/clock-reading caller overhead is silently counted as whole-model performance.
Timers include the function and QPC boundary cost; writes/comparison are afterwards.

Per candidate independently:

- Mean measured cost/original mean<=.80.
- EACH of six book mean costs/original book mean<=1.00.
- P95 measured query cost/original query p95<=1.00.

Compute over5*336 values per arm, retain per-sweep totals/mean/median/p95/max/book
means and ratios. Every apparatus valid AND at least ONE candidate costPASS licenses
NEW all-bank composition/export work. Retain failed LUT even if direct wins. If
both fail, close THESE fixed kernels before all-bank export, no kernel/threshold
sweep or extra repetitions under454. A different exact layout needs new variable.

## Resources, failures and full objective

Expected60..180s, approximately520MB output, parent1..2GiB/native1..1.5GiB.
Bound admission300s, numerical600s, total900s; compile120s, fault30s, primal240s,
timing300s further limited by remaining numerical budget. Parent peak3GiB, native
peak2GiB, conservative sum of separately observed peaks<=4GiB; outputs600MiB,
free disk>=2GiB. Native watchdog600s per process, controller timeout kills/waits
that child and retains stdout/stderr/partial hashes. Retain any FIRST failure.
No completed controller/run restarted after observation timeout.

ONE command after scientific freeze and committed operational instructions:

    .venv\Scripts\python.exe benchmarks\native_expert_scaling\meth454_switch_sparse_wo_cost.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth454_switch_sparse_wo_cost_result.json

Source payload size/mtime, engine and original binaries preserved. Hash all output
files, report source/compiler/runtime/commands/actual work/physical worker/peak.
No optimizer/new coefficient rounding/engine edit/new model download.

Scope selected-FFN ONLY: full core/router/head/cache/prefill/integration, actual
hardware DRAM traffic and new complete-model accepted rate remain unmeasured.
Real bank working set without assumed cache residency is not a DRAM counter.
Still require all-bank sparsity/changed routes, fresh donor-relative prediction/
generation/tasks AND>=50 batch1 acceptedIDs/s SAME artifact, useful RAM-scaled n
increments/winner AND mass selection, actual other-family/~100B applicability.
