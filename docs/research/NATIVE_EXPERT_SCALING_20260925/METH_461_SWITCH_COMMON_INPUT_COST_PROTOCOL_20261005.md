# METH461: exact shared-A16 fanout, primal controls and whole generation cost

5 October 2026. Prospective protocol, frozen together with ALL461 C/H/Python
before first import/parse/compile/numerical execution. ONE execution; no completed
controller rerun. Full goal ACTIVE/INCOMPLETE. Parent460 admission passed; no461
timing or arithmetic result exists at this freeze.

## Question, evidence and decision

Does sharing the ORIGINAL activation quantization and one static OpenMP region
among common-input QKV/KV maps reduce whole generation cost by at least5%, while
preserving complete original outputs? One primitive and three callsite patterns;
no coefficient/precision/layout/fit change.460 establishes legal input/shape/
output contracts and overflow bounds.458 establishes matched whole cost shares
and golden outputs, not the cost fraction removable by this transformation.

Accept execution recipe only if ALL apparatus and ALL three cost gates pass for
E128 AND E256 separately. If admitted, prepare fresh same-artifact donor-relative
quality/rate verification. If any cost gate fails, close this ONE execution recipe
and return to transfer/useful-capacity composition. No tile/shape/compiler sweep,
changed threshold, outlier trimming or timing retry. Regardless of outcome,
reassess the full transfer path after this one native cycle.

## Frozen apparatus and fresh admission

Controller: benchmarks/native_expert_scaling/meth461_switch_common_input_cost.py.
Native files: meth461_switch_common_input.c, _cost_entry.c, _entry.c,
meth461_switch_fanout.h and meth461_switch_fanout_controls.h in that directory.
New source/protocol physical bytes must equal filtered HEAD. Physical .gitattributes
must equal raw HEAD. UTF8 explicit; source/protocol LF, raw/retention JSON CRLF.

Direct retained raw SHA bindings:

* 458:3762fd4e6c86a6ea7e64f54d4350cc2bd91761f2f856a06bb0a0a442f12d285f.
* retention458:65ec6408bc1de79d0a3bad869b57de7298285901c5c91ca502330250ea9ea8b0.
* 460:6e1f921d05f08eba48e59c05d167420fc9160732ef3060617e23a4cdbf4646b0.
* retention460:51b658380511c9fb654c7f30e1b493ba95288feec234d69c24ef4227128f1e36.

Require parent apparatus/admission/retention gates. Refresh physical HEAD and SHA
of every458 helper and retained record, all2304 retained458 output files, both
FULL original payload SHA and size/mtime, manifests, original qualified binaries,
compiler and libomp. Inherit exact source quality/numeric/cost/manifest bindings
through these fixed parent records.460 artifact identity must equal458. Every
new/reused hash/byte inventory is retained. No historical catalogue rewrite.

Original128: trained distinct7,415,217,408, E128, payload7,541,946,880B;
original256: trained distinct14,664,154,368, E256, payload14,818,015,744B.
Both D768/F3072/12encoder+12decoder/top1/6 sparse banks per side, original
row-I8/A16 arithmetic and native loader. Different pretrained cores, not an
E-only causal comparison. Original128 CPU3[0,2,4]; original256 CPU6[0,2,4,6,8,10].

Fresh physical topology must expose parent allowance0..11 and physical choices
[0,2,4,6,8,10]. Preserve original runtime environment (ACTIVE, KMP_BLOCKTIME
infinite, KMP_AFFINITY none, OMP_DYNAMIC FALSE, OMP_MAX_ACTIVE_LEVELS1) and respective
worker counts. Copy argv before recording/launch. Read back process affinity,
all worker slots/masks/group0/distinct Windows thread IDs and binding events.

Original engine SHA54194c36b62571df014e8ebc251147ddd37436b5fb81c75db76f6fd4d8da0414
and physical HEAD exact; no engine edits. Source reversal must reconstruct
qualified388 exactly after removing ONE include and reversing three callsites;
cost/entry reversal permits only include/name change and appended control dispatch.

## Algebra and implementation

For each common query x, ORIGINAL head_activation_codes returns a F32 scale alpha
and A16 q=clamp(RNE(F32(x/alpha)),[-32767,32767]). Reuse exactly this returned
q/alpha across maps; each map keeps independent row-I8 codes and F32 row scales.
For map m,row r, output is ORIGINAL F32((F64(sum_i w[m,r,i]*q[i])*
F64(row_scale[m,r]))*F64(alpha)), preserving parentheses/order. FE_TONEAREST,
finite inputs/positive scale, ORIGINAL head_integer_dot, no reassociation/fastmath.

ONE static OpenMP region over flattened map*rows+row (if rows>64). Each output
coordinate belongs to one iteration; no reduction shared among workers. Maps2/3,
rows1..65536, cols1..4096, tokens1..CTX256. Require each I8 matrix has matching
dimensions/scales; output spans disjoint from input and other outputs. Single
query uses I16[4096] stack; batched queries one tokens*cols heap buffer and
F32[CTX] scales, freed after region. Layout/token strides unchanged. Replace only
encoder selfQKV, crossKV preparation and decoder selfQKV. Preserve normalization,
nonlinearity, other matrices, attention, expert selection/mass, head/greedy/stop.

I8 abs bound128 and A16 abs32767: pair8,388,352; cols768 I32 lane402,640,896;
cols4096 I32 lane2,147,418,112 <2^31. Final abs I64 sum17,179,344,896 <2^63.
Logical calls/maps*tokens, I8 bytes/maps*rows*cols*tokens and row-scale bytes
maps*rows*4*tokens preserved exactly. These counters are NOT DRAM measurements.
No product/storage/E/knowledge reduction. Profile1 group timer exists but this
whole contrast uses profile0 only; all matrix_seconds must be zero.

## Compile and independent arithmetic controls

Compile ONE standalone entry with original compiler/libomp and flags:
-O3 -std=c11 -march=x86-64-v3 -fno-fast-math -ffp-contract=off -fopenmp.
Preserve original binaries. Copy qualified libomp to new output directory.

Five fixed shapes (maps,rows,cols,tokens), CPU3 AND CPU6:
(2,2,17,1), (3,65,17,3), (3,65,768,3), (2,2,4096,1), (3,65,4096,1).
Nonmultiple16 scalar tails; serial/parallel rows; zero/extrema/RNE ties; two
query scales; cols4096 lane bound. Fixture order is frozen in controller.
Rows use all-128, all127 or fixed mixed original-I8 values; scales exact powers2.
First shape query zero. Last two all+32767/all-32767. Other shapes: query0 zero,
query1 cycles[32767,-32767,.5,-.5,1.5,-1.5,2.5,-2.5,0], query2 same*4.
Alpha is exactly1 or4; divisions are exact binary halves/integers. Python round
is an independent ties-to-even scalar reference; Python unbounded integer sums
followed by exact F64 powers2 products and final F32 struct.pack. Controls contain
-128 although real exports restrict codes to +/-127. Full original-model golden
outputs additionally cover natural continuous scales and non-power2 products.

Input binary MF461I01/count5, shapes4U32, token-major F32 x, each map row-major
I8 then row scalesF32. Output MF461O01/count5, shape4U32 then each map token-major
F32 outputs. Require native memcmp with separate ORIGINAL mv/mv_batch AND full
candidate output bytes equal independent scalar binary for both worker counts.
One expected input/output alias negative must exit2 with exactly one stderr line
switch_reference_error:fanout_input_output_alias (splitlines normalizes LF/CRLF).
These synthetic arithmetic controls do not establish useful expert capacity.

## Complete natural generation comparison

Source order128 then256. Each own parent458 24books*4cases=96 fixed consumed
short infilling contexts, same source IDs/S29/cap64/closing32095/profile0. Arms:
0 original qualified binary,1 one candidate binary. For case index j order
[j%2,1-j%2]; each process warm1,measured3,reserved0. Fixed source/arm/repetition
schedule, no adaptive batching or random reselection.384 generation commands,
1536 complete binary outputs (384warm,1152measured). Retain ALL stdout/stderr/
whole binaries; no removed slow/rejected case.388 commands including compile,
two controls and negative, expected2318 new files and2304 freshly rebound old files.

Require every warm/measured output SHA EXACT parent458 original profile0:
complete encoder/decoder states, logits and route records, generated IDs/count,
stop reasons. Require all logical phase/kind counters, source/closing/profile/
thread identity, finite positive phase/step times and phase sum agreement1e-7.
First-token/step arrays retained. Only warm rows excluded from cost aggregation.

For arm a, book b: T[a,b]=sum over its four cases of arithmetic mean of their
three measured full_generation_seconds. T[a]=sum ALL24 T[a,b], charging ALL
rejected-case time. Fixed accepted numerators128:96cases/1142ordinary/662prose;
256:81cases/895ordinary/490prose,15rejected timed (ALL1065generated).
Accepted ordinary/prose rates=N/T[a], descriptive; no confidence interval gate
specified here. Median-case total and repeat totals descriptive only.

P95 uses pooled288 measured full-generation durations per source/arm, sorted,
linear interpolation at .95*(N-1) between floor/ceil ranks. No per-book p95
replacement. Frozen usefulness gates SEPARATE for each original source:

1. T[1]/T[0] <= .95 (at least5% whole mean reduction).
2. EVERY24 T[1,b]/T[0,b] <=1.05 (no book >5% slower).
3. P95[1]/P95[0] <=1.00 (no pooled tail worsening).

All three for BOTH sources required. No rounding gate inputs or changing5% to3%.
This is one bounded apparatus decision, no general statistical confidence claim.
Whole timer includes encoder/crossKV/cached decoder/head/greedy/stop, excludes
startup/load/team setup/fixed-ID tokenization/serialization/cleanup. Repeated
consumed quality controls do not become fresh held-out evidence or a new50 proof.

## Budget, stops, retention

CPU only, no fit/GPU/download. Expected <15minutes, parent<1GiB, native<2GiB,
~5GiB new outputs. Admission<=300s; total main excluding imports<=1800s;
compile/control/native/aggregation portion<=1500s. Compile120s, each control60s,
negative30s, each generation60s. Require availableRAM>=16GiB/free disk>=10GiB.
Parent peak<=1GiB; conservative parent peak+largest command-root peak+current
compiler-descendant RSS <=16GiB; child-prefix<=128MiB/known+live outputs<=8GiB.
GetProcessMemoryInfo72B on retained root process handle, .25s monitoring and
terminal peak query for every command, including compile/exit2. Compiler
descendant RSS is sampled, NOT a terminal exact descendant peak; record that
limit. Working set is not physical DRAM traffic. No hashing/parsing while child
live; monitor only resources/current-prefix sizes. No interfering scientific job.

Live parent/actual ancestors excluded by PID. Sole daemon allowance: pythonw.exe,
exactly2argv, argv1 resolves D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py.
Preserve it. On error/resource stop kill/wait only own current child. Retain
exclusive first failure JSON and outputs before a separately numbered minimal
repair. Output directory/raw/failure must not preexist. Success raw written once;
SHA inventories/gates/resources/commands and complete records retained. Metadata
audit follows terminal execution; never rerun scientific controller for retention.

## Scope relative to goal

Exact common-input reuse may reduce E-independent core overhead. It cannot add
pretrained knowledge, unify incompatible core coordinates, reduce stored expert
information or remove O(E) score/winner/mass work. No useful-n increase, actual
DRAM/LUT scaling, composed compressed artifact, other-family/~100B or fresh final
donor-relative quality/SAME50 is established by461. Record result and return to
those transfer conditions; local cost success alone does not complete the goal.
