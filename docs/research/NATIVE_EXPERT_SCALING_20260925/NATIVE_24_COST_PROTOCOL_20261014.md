# Frozen actual 24-layer recurrent cost screen

Question: how much of the 20 ms useful-50 budget does the qualified native
recurrent implementation consume? This screen measures its existing code,
not a new core, a dense source decoder, or final end-to-end throughput.

Reuse the admitted full-width bridge traces, row_state DLL and all 24 distinct
canonical parameter packets. No model/GPU/optimizer/RESERVED queries.
Complete source-active audit and closure before launch of this timing worker.

Select shortest and longest existing history in each of FIT and DEV, sorting
by (history length, id), with lexicographically first id on ties. Freeze the
last 16 token positions of each: four histories, 64 whole-24 observations
per mode. No selection from observed timings. This is a small deterministic
cost screen, not a representative statistical language benchmark.

Fresh contexts reconstruct every prefix from actual causal BF16 input rows.
Layer zero uses the qualified uint32-ID F0 table; subsequent layers use the
actual preconv rows archived by the own-history quality bridge. Prefix
initialization may run layer by layer and is outside timing. Never load
teacher snapshots into a context. Each timed token executes layers 0..23
sequentially, one rs_block(1) each. Within-layer C remains single-threaded;
worker affinity [0], OPENBLAS_NUM_THREADS=1. Load all distinct parameter
arrays at once, preserving owners and reporting addresses, types and bytes.

Two modes in fixed order: resident (no deliberate eviction, NOT an all-cache
claim), then parameter-evicted. Before EACH whole-24 evicted sample, CLFLUSH
every cache line intersecting each of the loaded 24 parameter arrays and
MFENCE. State, input/output rows and executable code are not deliberately
evicted. Flush is outside the measured interval and recorded separately;
its cost is not subtracted from a deployment rate. This intervention is not
a memory traffic counter. Actual DRAM remains UNRESOLVED; report an operand
ledger including parameter multipass and state reads/writes explicitly.

Prebuild input/output pointers outside timing. Measure perf_counter_ns whole
24 span and each call; include Python loop/ctypes/clock overhead. A same-ABI
C no-op is measured for 128 whole-24 iterations, reported without subtraction.
Check every prefix/timed BF output against archived output words. Store ALL
16 continuation intermediates per site, final state/ring/seen/last, exact
per-layer/whole spans and cache-control spans. Final state/ring/last must
match the original trace snapshots bit for bit, in both modes.

Independent stored auditor replays fresh prefixes/continuations using the
qualified DLL, verifies every saved array and final snapshot, every prefix
BF word, and performs independent int64 matmul, RNE input quantization,
ordered F64 scales and BF rounding on ALL saved continuation readouts.
This shares the already qualified recurrence DLL and does not claim a new
independent recurrence equation proof or authenticate wall-clock values
beyond stored consistency. Verify timing identities, all samples, arithmetic
summaries, array ownership/extent metadata and input/output hashes.

Report median/p10/p90/min/max whole-24 and per-site, split/length strata,
prefix cost, loaded parameter/context/buffer footprint, shared desktop CPU
activity and frequency snapshots. No heavy concurrent research worker
during timing; do not stop user processes. No privileged PMU installation.

Decision fixed before observation: if resident whole-24 median >=20 ms, the
measured recurrent implementation itself exhausts the useful-50 budget and
requires a real implementation/core change before expanding FFN banks. If
below, report remaining median budget 20 ms minus measured recurrence,
explicitly still owed to generators/FFN/routing/attention/head/transport.
Also assess the 10 ms improvement budget. This is not an impossibility bound,
a final TPS claim, or quality evidence for a changed implementation.

Preparation/run/audit/closure exclusive outputs and PID/create terminal
receipts. Per phase <=1800 s, OS <=2 GiB; new E family <=512 MiB.
Freeze code/protocol in Git before binding, all selected trace/parameter/DLL/
support compiler/runtime/source pins before observing timing. No code or
threshold repair in place; preserve first defect and version any correction.
