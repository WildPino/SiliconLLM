# Native selected-p6 LUT: correctness admitted, acceleration rejected

13 October2026. **COMPLETE / FULL INDEPENDENT CORRECTNESS PASS / COST FAIL.**
[Protocol](TERNARY_READOUT_NATIVE_PROTOCOL_20261013.md),
[native result](ternary_readout_native_result_20261013.json),
[full audit](ternary_readout_native_adjudication_20261013.json),
[native closure](ternary_readout_native_session_closure_20261013.json).
[Cost protocol](TERNARY_READOUT_COST_PROTOCOL_20261013.md),
[cost observations](ternary_readout_cost_result_20261013.json),
[independent full ledger/custody review](ternary_readout_cost_review_20261013.json).

## Representation transferred to actual C

Same FIT-selected p6 codes/scales from the qualified source-only precision
assay. Original source O2048x3072,own-normalized3072 BF16 input,all144 histories/
sites0/12/23,no new recurrence/generator/parameters/fit/activation quantization.
Actual new C float64 LUT:five-trit codes,615 groups,sixplanes,eight-group
block243-entry tables15,552B,six output accumulators98,304B,AVX2 gather4
outputs,increasing-group/plane accumulation,F64 row scale/F32/BF16 round.
All C calls one native thread/core0. This is a new backend versus the actual
engine's2-trit int8/pshufb/nibble path;old integer qualification/rates distinct.

Each history also executes original engine AVX dotf/hsum256 copied byte-identical
in source under original compiler flags,to compare every original F32 score.
Separate no-vectorization scalar q-weight reference on16 fixed FIT positions/
site;all243 table entries checked exactly with input[1,3,9,27,81];known code
mutation changes only one output by predicted scaled delta. Those producer
receipts are not independently re-executed by full stored auditor;scope clear.

## Complete numerical and local quality evidence

Independent auditor reconstructs every coefficient from packed digits,then
all32-row promoted-q F64 responses. ALL22,671,360 packed codes/18,874,368 q
coefficients and138,528,768 native LUT F64 words verified. Maximum relative
raw error5.6963e-17 under frozen1e-10. ALL138,528,768 original F32 control words
BIT-EXACT to stored original scores;ZERO mismatches. ALL138,528,768 LUT BF16
words independently checked/exact RNE of native raw,all equal prior p6 precision
outputs,ZERO independent differences/smooth0. Metadata max6.37e-21.

All144 extra raw/BF16 each-case1% gatesPASS and all original donor local mean/
worst/centered/domain gatesPASS FIT+DEV at0/12/23. Prior p6 accuracy unchanged:
all-case extra raw worst.367007%/BF16.431457%;source DEV local BF16 means
.0763946657654/.0872459267091/.0251148531471. This is supplied-source-input
readout fidelity,not OWN-history or end-to-end chatbot quality. No gate loosened.
Scalar-control relative max6.60e-17/mutation delta max1.58e-13,all patterns exact.

## Isolated cost on the same p6 parameters

Correctness replay executes LUT PLUS original dense control and cannot isolate
LUT latency. After successful full audit,a separate generated timing wrapper
renames the LUT export and deletes ONLY the original dense-control assignment;
all remaining LUT arithmetic unchanged. Original dense-only path uses same
engine primitives/rounding. Every measured output raw/BF16 required exact
archived native bits. This wrapper parity is sampled,not a second full-history
wrapper audit;coefficients/scales/inputs remain the identical accurate artifact.

18 fixed FIT input-ready samples:minimum/max history cases/site,first/middle/
last positions;5 paired alternating method repetitions,CPU0/batch1. Warm mode
immediately repeats each method without forced flush,not a cache-residency
proof. Source-flushed mode verifies CPUID CLFLUSH/64B lines,evicts every code/
scale line for LUT or O line for dense,including unaligned final line,MFENCE
before timing. Flush cost separately recorded. Input/workspace not flushed;
table build/allocation/free/lookup/combine/scale/BF16/ctypes overhead included.
QPC and perf_counter data retained;shared desktop/dynamic frequency unisolated.

Median call times in microseconds,30 observations per site/mode/method:

|Site|Warm LUT /dense|Source-flushed LUT /dense|Flushed LUT/dense ratio|
|---:|---:|---:|---:|
|0|5158.45 /1221.05|5278.45 /1210.50|4.36x|
|12|5195.35 /1206.35|5278.00 /1196.15|4.41x|
|23|5062.10 /1184.35|5275.55 /1158.65|4.55x|

All360 timed observations reconciled independently,all18 samples/every measured
and warmup raw+BF16 output parity checked by producer. Full observation/source
transform/duplicates/record counts/source-byte totals/statistics/immutable hashes
reviewed. No rerun to choose faster figures. LUT sources7,573,504B including
scales versus dense25,165,824B,about3.32x smaller but4.2--4.6x slower. Reject
CURRENT F64-LUT implementation as acceleration. Result does not prove every
LUT layout is slow. Cost of table/gathers/scratch versus other primitives needs
an explicit next variant;no token/model rate inferred from these operator calls.

Architectural source eviction is evidence of initial cache state,not PMU
measurement of all physical DRAM traffic or useful expert-n scaling. Warm
versus flushed similarity does not prove source always cache-resident;original
matrix large and desktop timing noisy. Raw records/min/max/flush times retained.

## Resources/custody/provenance

Native source/protocolb043503,launchbb95ff2,audit launch0393ee9,
admission477fedf. Binding
69d16f93ffa7122f34f402984ff6a1e1bcd698ac86f71d15a85760fd712b8447;
result306410f58fa2c5326e81a431f33d42e6522da21601978e9a801fb28c4c208f87;
audit2e727117e2385c8e4f4bc468f5dc6db19dbb17833576b5d707b9c43ecbca0f7e.
Prepare11.172s/OS41,197,568B,producer532.094s/OS406,597,632B,
full audit45.125s/OS403,910,656B. Two compiled DLL/logs plus144 raw-score/BF16
NPZ1,939,648,768B total family,under2GiB. Every extent/terminal/PID+creation/
protected engine/foreign hash checked,no owned native process live.

Cost source1e59325/launch3a0a00b/review37142c1,prepare2.328s/OS39,751,680B,
run7.109s/OS120,004,608B,both terminal0/matching processes absent. New generated
source/DLL/log under16MiB,actual extents in ledger review. Cost result
9981751919050297df5168792e7c60dfe9cd8ccb385de30afd7a7f28d72b553b,
review0237b7f942916a4e3e2afd1a7bc1efd95cdd99dfd0bfbc961f626fd21f6703c7.
No source/GPU/T4/RESERVED/recurrence/optimizer or new mechanical scientific
command fault. Principal actual NumPy/BLAS/Python/psutil/compiler/source/parent
data pins checked;not a complete transient DLL-tree guarantee.

## Next and whole objective

[Bounded integer readout](INTEGER_READOUT_NEXT_20261013.md)
PROPOSED/UNIMPLEMENTED/UNBOUND:preserve p6 q/scales,new explicitly quantified
int16 input approximation,AVX2 PMADDWD block128/int64 accumulation with formal
overflow bounds,full independent integer equality BEFORE accuracy/cost. More
weight bytes than ternary pattern code may still reduce expensive operations;
no untested speed promise. The accurate native F64-LUT remains an immutable
reference/cost failure. [T-MAC primary-source check](NATIVE_LUT_PRIMARY_SOURCE_NOTE_20261013.md)
suggests bitwise/table-size/hardware distinctions,not transferable benchmark rates.

Source CHATBOT->compact shared causal core+useful selective ternary functions->
engine.c still requires generator/FFN/attention/tokenizer/head/OWN histories/
useful RAM-driven n/10B/100B/multiple families/scales/conversion O(n) storage
and SAME useful>=50 tokens/s. Accurate or fast one readout cannot complete
that goal. Goal ACTIVE/INCOMPLETE,all current jobs closed.
