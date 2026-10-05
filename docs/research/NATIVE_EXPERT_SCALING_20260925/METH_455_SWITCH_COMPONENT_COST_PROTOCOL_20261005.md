# METH-455: equivalent stage costs with separation and timer controls

5 October 2026. PROSPECTIVE: new C and controller uncompiled/unimported.
Freeze both sources, this protocol and attributes BEFORE first compilation,
import or numerical observation. ONE controller execution; preserve any first
failure before a NEW numbered repair. Goal ACTIVE/INCOMPLETE.

## Uncertainty, decision and new observable

454 exact C functions passed all8 apparatus gates but both fixed cost recipes
FAILED all3 prospective cost gates. Those recipes remain closed. New455 measures
the ordered component costs responsible for the failure, with controls for the
physical/code-generation perturbation caused by splitting and adding timers.
No coefficient changes, new rounding, fit, source masking, threshold tuning or
data acquisition. This is a diagnosis, not a candidate speed/quality promotion.

ONE physical modification is selected only under the predeclared conditions
below. No sweep of output tiling, paired WO, symmetric LUT and native WI.

## Reused evidence and fresh bindings

454 raw SHA256
`efc087e32c2f88631654efaa4d0f30bf9b23288f66243deb51ed5f07a947d277`;
retention454 SHA256
`4747e27123ea23c2aaf94b415e57b58683f9d35696c7f079fb15684f9e6233a2`.
Freshly bind ALL fourteen454 complete output hashes/sizes, helpers/parent raw
records, original128 payload7,541,946,880B and manifest, preserved engine54194c36,
original374/389 binaries, prior numerical runtime DLLs, original compiler8ba7ddd7
and libomp6fc163dd. New scientific source/protocol/controller physical bytes must
match committed HEAD; attributes must match raw HEAD. No old controller execution.

Reuse454 bank MC454B01 /472,257,280B, trace MC454T01 /1,034,908B and qualified
primal MC454Q01 /43,915,224B. ALL128 functions/coefficients remain identical,
with all fields64-aligned; native loader validates every sign/code/scale. Actual
inputs336 from books18..23, bank11 D768/M3072/E128, original forced ID and
ID+1 controls. No new held-out quality or newly routed input claims.

Original454 C is included UNCHANGED with its main renamed, never invoked. It
includes the unchanged qualified374 primitives. The diagnostic adds copies of
the same loop bodies with boundaries removed from interleaved arithmetic:

- Original: split original A16 quantizer and SAME I8/I16 dot, F64 scales and
  OpenMP row loop. Code buffer/stack placement and shape-check placement change;
  measured explicitly by the split control, not assumed free.
- Candidate: SAME signed F64 Walsh/cast, original A16 quantizer, unchanged direct
  WI operator. For LUT, build454 followed by the EXACT remaining lookup loop,
  with zeroed weighted outputs and ascending block accumulation.
- SAME ReLU, original WO A16, ascending nonzero scan, SAME512-column partial
  I32/I64 accumulation, SAME F64 row-scale then activation-scale and F32 cast.

All native454 tiny tests retained. New split column operator independently
compared to scalar FULL I64 for3072 positive/negative/alternating/zero/single
inputs and coefficient extrema. Bound512*127*32767=2,130,641,408<2^31 preserved.
No floating reassociation/FMA or signed overflow permitted.

## Three timing modes for each of three arms

Arms: original I8, direct I4/sparse I8 WO, pair LUT I4/sparse I8 WO.

0. UNCHANGED454 whole selected FFN, no internal timers.
1. Equivalent split code, no internal timers. New physical/codegen separation
   bias is mode1/mode0; numerical equality alone does not prove same cost.
2. Equivalent split code, with internal QPC boundaries. Actual instrumentation
   bias is mode2/mode1. All raw elapsed costs retained, never subtracted to rescue
   454 or claim a candidate cost improvement.

Ordered component slots:

| Slot | Component | Arms |
| --- | --- | --- |
|0|Signed basis, F64 Walsh/cast|direct/LUT|
|1|WI original A16 quantizer|all|
|2|Complete pair-LUT construction|LUT|
|3|WI dot/decode/reduction/F64 scale/output|all|
|4|Original-coordinate ReLU|all|
|5|WO original A16 quantizer|all|
|6|All3072-code zero scan and index construction|direct/LUT|
|7|WO zero initialization, column products, partial reductions/I64 combination|direct/LUT|
|8|WO F64 row/activation scales and F32 output|direct/LUT|
|9|Original WO I8/I16 dot and F64 scales/F32 output|original|

WI decode, integer reduction and scales are interleaved per row/block; original
WO dot/scales are interleaved per row. They remain COUPLED in this measurement.
Timer per block/product or materializing a new tensor of intermediate dots would
alter the operator being diagnosed. No isolated causal claim about those subparts
is supported by this protocol. Whole selected-FFN elapsed surrounds the function
including boundaries/return. Component sums cover the interior interval; remaining
boundary/return time reported explicitly. Extra writes, checks and log are outside
the selected-FFN timer as in454; internal timestamp bookkeeping is charged.

10,000 EMPTY boundary sequences per arm (5/8/9 stage ends plus begin and outer
timers), consumed checksum after timer to preserve stores. Report mean and ratio
to profiled whole; this is a calibration only, not a full overhead model or
subtraction. Actual profile/split comparison governs admissibility. All selected
FFN clocks use SAME Windows QPC frequency; report tick resolution/frequency.

## Primal first, then fixed timing schedule

Before timing: native tiny qualification, then3360 full records:
mode1+mode2, each original336/candidate correct672/wrong672 in unchanged454 order.
MC455Q01 header: N336,D768,M3072,count3360,recordbytes26144. Each mode U32 plus
UNCHANGED26140B454 record. File87,843,868B. Python byte-compares ALL fields of
each record to matching454 record: IDs/nonzeros, scales, basis/raw/down, both A16
code arrays. No tolerance-only comparison. First mismatch stops before profiling.

Timing: 2 warm whole336 traces and5 measured whole336 traces for every one of9
arm/mode combinations. Repetition -2..4; group combination=arm*3+mode. Start
combination `(rep+2)%9`, rotate nine complete traces. Finish one complete336 trace
before next combination; never adjacent same-input mode alternation. ALL21,168
outputs byte-compared to qualified454 down AFTER the selected-FFN timer.
Checksums/logging after timer. IDs/books/order/nonzero counts and all zero inactive
component slots validated. Native whole-clock checksums/fixed affinity reported.

Report5 measured sweep sums, mean/p95/max and every book mean for all9 groups;
raw21,168 records including warmups retained. Profile each component mean and
fraction of profiled elapsed; no significance/cycle/DRAM-causality assertion.

## Frozen admissibility and decision gates

For EACH arm, require ALL:

- Split/no-internal-timer mean /unchanged454 mean in[.90,1.10].
- EACH book split/original mean ratio in[.85,1.15].
- Internally profiled mean /split mean in[.90,1.10].
- EACH book profiled/split mean ratio in[.85,1.15].
- Empty boundary calibration mean /profiled whole mean<=.02.

All fresh bindings/primal/output/order/partition/worker gates also required.
If any arm fails admissibility, no physical recipe is selected from this profile;
retain perturbation evidence and prepare a NEW diagnostic, never rerun455.

If all diagnostics admitted, examine DIRECT candidate fractions in this order:

1. WI dot/decode/scale fraction>=.50 -> prepare ONE NEW exact output-tiled WI
   geometry. Preserve all coefficients/scales and ascending F64 block order.
2. Otherwise WO column accumulation fraction>=.30 -> prepare ONE NEW paired
   active WO column operator with exact integer sums and512-term bound.
3. Otherwise no predeclared dominant component: reassess, do not sweep formats.

These conditions choose the next inquiry; no speedup is promised. Whole-model
benefit still requires matched full-model f and `1/((1-f)+f/S)` only under the
additive nonoverlapped cost model. Existing small contexts/worker counts cannot
provide a missing matched391 full-context fraction.

## Resources, command and stops

Same compiler flags -O3 -std=c11 -march=x86-64-v3 -fno-fast-math
-ffp-contract=off -fopenmp -lpsapi; unchanged native environment ACTIVE/none/
infinite/FALSE/one-level/one thread. CPU0/group0/mask1, one physical worker.
No GPU/BLAS arithmetic, downloads, optimization updates or engine edits.
Expected tens of seconds and <100MiB new files; hard stops:

- admission<=300s, new numeric phase<=600s, parent total<=900s;
- compile120s, primal240s, native profile300s wrapper; native total600s;
- parent peak512MiB, each native peak2GiB, conservative sum<=2.5GiB;
- outputs<=128MiB and initial free disk>=1GiB;
- first mismatch/resource/format/affinity fault stops with retained failure record.

Fresh process inventory excludes actual current process ancestors only. Reject
other Python/model jobs; preserve only exact pythonw two-argv publisher allowlist.
All child exits/stdout/stderr retained; subprocess timeout kills AND waits.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth455_switch_component_cost.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth455_switch_component_cost_result.json
```

After ONE run: retain raw/archives/logs/hashes/first failures, update brief INDEX,
METHOD and exact resumption. This diagnosis cannot establish new whole fresh
quality, >=50 acceptedIDs/s, useful-n growth, real DRAM or another family/~100B.
