# Fixed same-parameter isolated batch1 LUT/dense cost

Require completed native LUT all144 histories/full independent auditPASS
and all numeric/quality gates. Selected p6 parameters NEVER changed. To isolate
LUT time,generate a separate source from exact qualified native.c by renaming
tr_history to tr_lut_only and deleting ONLY its original dense-control assignment.
All LUT arithmetic unchanged,compile same flags. Append exact original dotf
dense-only output and CLFLUSH/support functions. This timed wrapper is a new
binary;its sample outputs must be raw/BF16 bit-equal archived native outputs
on EVERY measured call. Do not claim full wrapper histories re-audited.

Samples fixed before timing:minimum(history,id) and maximum(history,id) FIT
case/site0/12/23,positions0/history//2/history-1:18 source-derived samples.
No DEV selection or fastest sample search. CPU0 one thread;5 repetitions,
warm-repeat/source-flushed modes,methods LUT/dense alternating order by rep.
Each call batch1;warmup identical method before EACH measurement. Warm mode
means immediate repeat/no forced flush,not proof all weights fit any cache.
Source-flushed mode:CPUID CLFLUSH support and64B cache lines required,flush
each source code/scale line for LUT or original O line for dense,including
unaligned final line,MFENCE before timed call. Flush time separately retained.
Input/output/workspace not flushed;LUT builds inside timed call each time.

Python perf_counter_ns around ctypes call includes FFI/call/malloc/free/table/
lookup/scale/BF16 overhead (dense includes output BF16 conversion too). Record
native QPC frequency/perf_counter resolution,each observation/source bytes/
latency,site/mode/method medians/min/max. All360 timed outputs must match
previous scores/bits exactly;warmups checked too. Reject current wrapper if
any difference,retain fault and fix before admitting sampled cost.

Cache flush asserts architectural initial source eviction,NOT hardware PMU
measurement of all DRAM bytes. Shared desktop/dynamic CPU effects retained;
no expert-n scaling/admitted model-rate/50 claim from these input-ready calls.
Keep native full-history correctness and timed-wrapper sampled parity scopes
separate. Reconcile every measured record/count/summary and immutable hashes
in custody review;timing measurements cannot be independently re-created as
identical values. No repeat benchmark unless a concrete risk remains.

Prepare/run900s/OS6GiB/new orderedE16MiB,compiler45s. Hash producer/source/
protocol/storage/compiler/principal NumPy/Python/psutil/parent report/audit/
source parameters/sampled actual-input and prior-output packets/foreign files
before/after. Source/protocol freeze before prepare/binding/launch,terminals
with PID+creation/status/resources. No GPU/T4/source/recurrence/RESERVED or
new bank/precision/activation quantization. Full goal active/incomplete.
