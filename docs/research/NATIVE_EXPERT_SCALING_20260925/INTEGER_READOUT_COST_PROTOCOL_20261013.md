# Fixed complete audited integer readout call cost,with input conversion

Require full corrected V3 integer auditPASS/all144 exact-dot/BF16/numeric and
extra1%/incremental0.1%/source local gatesPASS. Same p6 q/scales and EXACT
integer DLL from full audit;no new binary or arithmetic variant. Dense-control/
CLFLUSH functions reused from prior reviewed timing DLL. No LUT retiming.

Samples and schedule IDENTICAL to prior LUT/dense comparison:18 fixed FIT
min/max(history,id) cases/site0/12/23,first/middle/last positions,5 paired
alternating reps,warm-repeat/source-flushed. CPU0/batch1,input ready as actual
BF16 normalized bits. Both methods decode BF16 toF32 INSIDE timed region.
Integer additionally F32->F64 and literal qualified Python quantize function
(INCLUDING diagnostic norm/bound calculations and temporary allocations),
then actual same DLL PMADDWD/int64/scale/output call. Dense original F32 dot/
BF16 call. Preallocated output buffers for both. Quantizer not excluded from
timing;its Python overhead makes current call more expensive than bare kernel.
Cross-turn old LUT timing excluded BF16 decode,so do not present old/new cost
ratios as one perfectly simultaneous matched-code benchmark.

EVERY measured and warmup integer input r/a/dot int64/raw F64/BF16 value
bit-equal full audited archived case position. Every dense raw F32/BF16
bit-equal original native reference. Fail if any mismatch. No sampled parity
substitute for full history quality:the same actual integer binary was already
fully audited,and sampled source decode/quantizer equality checked here.

Warmup immediately before every measurement,warm means repeat/no forced
flush (not cache fit guarantee). Cold source:CPUID CLFLUSH64B required,flush
every q/scales line for integer or source O for dense including unaligned tail,
MFENCE before timed call,flush time separately. Source inputs/output/workspace
not flushed. Architectural initial eviction,not PMU all-DRAM traffic proof.
Raw records,source bytes,clock QPC/perf resolution,summary median/min/max.
Literal source weights:integer12,599,296B incl scales,dense25,165,824B.

360 observations/full duplicate/count/sample/summary/source/hashes review,
sample parity recorded;no retiming to select better numbers. Shared desktop/
CPU frequency effects,no full model accepted-rate/useful-n/physical expert
DRAM admission. All before/after parameter/runtime/DLL/source/report/adjudication/
selected-data/foreign extents pinned;no source/GPU/T4/RESERVED/recurrence.
Prepare/run900s/OS6GiB/new orderedE16MiB(no compiler/new matrices);terminal
PID+creation/exit/resources;goal remains original full pipeline/incomplete.
