# Fixed native blocked selected-p6 LUT correctness

Same FIT-selected6 code/scale artifact from successful full precision audit,
sites0/12/23/all48 supplied own-norm BF16 input histories. No new q/scales/
selection/recurrence/generator/source/GPU/T4/activation quantization/RESERVED.
New five-trit FLOATING backend;old2-trit int8 pshufb proof/rates not transferred.

C group-major codes[p,615,2048],8-group block F64 tables243 each/15,552B,
six2048 F64 accumulators98,304B. Fixed nested base3 pattern decode/5 terms
in input-coordinate order,final3 input padding zeros. AVX2 gather4 output
coordinates,increasing group order/plane index. Each plane summed separately
across all groups,then6 terms times1/3/9/27/81/243 in order,F64 row scale once,
F32 cast/BF16 RNE. No implicit FMA fusion,compile -ffp-contract=off.
Original O F32 uses exact engine dotf/hsum256 bodies/same original flags.
Every original raw F32 score must match actual stored source-native bits.

Separate scalar reference compiled no-vectorization:integer-q decoded weights
times F32 BF16 inputs in coordinate order,F64 row scale. Fixed first16 positions
of minimum(history,id) FIT case/site,relative native/scalar<=1e-10.
All243 LUT patterns exactly verified with input[1,3,9,27,81] ->code-121.
Change plane0/group0/output0 code to(code+1)%243;output0 must change by known
scale0*code_delta within1e-10 relative,all other outputs bit-identical.
These producer receipts remain distinct from full independent stored audit.

All144 histories:archive LUT F64 raw scores,original F32 control scores,LUT
BF16 outputs. Independent auditor does not import producer:reconstruct every
q coordinate from packed digits/padding,all32-row promoted-q F64 responses,
every138,528,768 LUT F64 words/original F32 words/BF16 words. Original F32
bit-exact;native LUT raw relative<=1e-10,versus prior BF16 smooth<=1e-4,
all bit mismatches retained. Verify BF16 RNE of native raw exactly. Per-case/
centered/label/domain/worst extra raw/BF161% and original donor local gates
unchanged. All metrics/flags/aggregates/hash controls independently verified.
Full audit even quality/numeric failure;no native cost admission until success.

Prepare source-freeze before compile/binding (compiler children allowed only
during bounded compile);record exact commands/exits/DLL/log extents. Actual
resolved NumPy ABI/BLAS/Python/psutil/compiler/executables/engine/source/helper/
protocol/storage/parent reports/144 native+source+precision packets/3 p6
packets/original exports/foreign files pinned before/after. Source coefficient
provenance from successful parent,actual O control from verified native exports.
No full transient DLL-tree guarantee. No fresh model load or gradients.

Compile2 DLLs/45s each/prepare900s;producer900s/audit1200s/OS6GiB/new family2GiB.
BLAS6/CPU0..5 for Python reference calculations,C calls/scalar on CPU0 only,
single native thread. New family rootE and flat run subdirectory ordered and
tracked;outputs~1.94GB decimal below2GiB. No timings in correctness C or admitted
token rate. After full audit,freeze separate SAME-artifact batch1 warm/cold
cost protocol with source-cache state/DRAM limitations and bounded samples.

Full CHATBOT conversion/OWN histories/useful conditional ternary n/compact
causal core/other donors/scales/conversion masters/SAME useful50 remain open.
