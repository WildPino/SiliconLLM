# Same qualified C operator, separate batch1 timing

13 October2026. Execute ONLY after full numerical independent audit PASS.
Fixed AVX_F32 ownnorm path,all48 captured histories,two complete passes,three
shortest FIT history warmups chosen by length thenID. Native history loops
sequential batch1 tokens,reset384x96 state per history;each position only current
supplied operands/prior state. Every raw F32 score/BF16 output/final state must
match qualified artifact bits after each timed history. Flags6 skips oracle/F64
output paths;quality/admission uses the already qualified identical F32 artifact.

One native thread,affinity0;all source data loaded/arrays allocated before each
call. Per-call perf_counter plus internal QPC projection,scan,decode,gate/RMS,
normalize+output stages. Store QPC frequency,perf_counter metadata,all per-history
times/sums/unattributed call remainder and process snapshots before/after.
No data-loading/hash/output-verification time inside call,so this is explicitly
supplied-operand component time. State allocation/output writes inside C call
are charged. No clean-system/expert-DRAM or whole-chatbot rate admission.

Retained in_proj6704x2048 GEMV timed separately:one warmup/two passes of128
arbitrary deterministic vectors `ldexp((arange(128*2048)%31)-15,-4)`.
Full actual source coefficient matrix exportedF32,54,919,168B (52.375MiB);
do not infer source quality/conv/activation/gate behavior from arbitrary inputs.
All858,112 generator output values independently checked against full NumPy
F64 source-coefficient contraction1e-5 relative,outside the timing window.
Generator and operator timings are separate workloads,not a measured sum of
a composed source layer. No actual source forward,source hidden input capture,
T4/GPU/RESERVED/new fit. Conv/state/attention/MLP/head and source input functions
still required for complete pipeline50. All before-after hashes/bit checks.

Budget900s,OS6GiB,no new large arrays written;parameters and quality arrays
remain immutable on E. Exact source/protocol/binding/adjudication/compiler/
runtime launch frozen before timing. Full aggregate arithmetic checked after
completion;component stage observations are not an end-to-end success claim.
