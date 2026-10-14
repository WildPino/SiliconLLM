# Frozen actual23 contextual generator cost screen

Before numerical observations. Source BF16 in_proj matrices at sites1..23,
each6704x2048, source checkpoint values. Keep BF16 coefficient storage and
decode toF32 inside AVX2/FMA registers; no precision compression or F32/F64
inflated weight storage. Caller supplies actual BF generator h from admitted
own-history traces. First generator uses F0 lookup and is excluded here.

Input decodeBF2048=>F32 once/call; eight FMA lanes over256 blocks, ordered
scalar lane reduction, rawF32=>GG BF RNE, decodeGG*decode actual6704-entry MuP
vector inF32=>field BF RNE. No invented bias, source multipliers or early MuP.
Known source-GPU versus independent CPU arithmetic is checked, not assumed bit
identical. Same compilerO3/AVX2/FMA/znver2/ffp-contract=off; affinity0/thread1.
FE_TONEAREST and MXCSR control mask0xffc0==0x1f80 are required (FTZ/DAZ off).

Same four shortest/longest FIT/DEV histories and last16 positions as qualified
whole24 screen. One shared set of23 BF matrices simultaneously owned/mapped to
original source ranges, reused for all four histories, total631570432bytes.
One generator per site,23 sequential calls per measured position, no parallel
layer batching. Both resident and parameter-evicted64-position modes, evict each
source matrix and shared MuP before each evicted23-call group; flush outside
timing.128 same-ABI no-op23-call overhead samples, no subtraction/outlier removal.
Store ALL rawF32/GG BF/fields BF for1472 calls per mode (2944 total), timestamps
whole23/per-site/flush, source range/addresses/payload and machine CPU sharing.
Input/precomputed source fields supplied only for validation/cost isolation;
preceding decoder work, recurrence16.8ms, FFN/attention/head/transport omitted.

Full auditor independently reads actual BF source values, computes F64 products
on every stored input, relative rawF32 discrepancy<=1e-4. Reproduce RNE(rawF32)
and ordered F32 MuP from saved raw exactly, all BF words. Smooth F64=>BF=>MuP
field control relative<=.01; recorded actual-source field control<=.01 both
overall and each of gate/x/B/C/dt slices for EVERY16-position record/site.
Auditor also replays every C call and checks all raw/rounded stored words exactly.
Every input/output finite; no record/domain exclusion. Scalar/metadata/timing
sample/summary accounting exact. Shared lib/source/capture limits explicit.
Full before/after input/output hashes, worker PID/create receipts and closure;
numeric audit completes on cost failure. No source model/GPU/RESERVED/fitting.

Decision fixed: compare measured generator-only median with3.18275ms resident
and2.97615ms evicted room left by the qualified recurrent screen. A larger value
means this measured dense-source baseline does not fit the final budget; it is
not a theorem about conditional/nonlinear encodings. Do NOT sum separately
measured component medians and call them end-to-end throughput. DRAM unresolved;
report exact operand ledger and cache conditions only. After this choose one
substantive source-dependent active-byte/operator/width/depth/conditional change
with own-model quality, not another small kernel grid or unpriced FFN bank.

Per phase<=1800s,OS<=4GiB,new ordered E family<=192MiB. Source and traces reused,
new output payload157892608bytes calculated from8bytes per field*6704*16*23*4*2,
plus eight uncompressed NPZ headers, compiler log and DLL, below192MiB.
Freeze code/auditor/protocol, then compiler/runtime/source/trace/budget binding,
before measurements. Preserve first faults and repair with new outputs/revisions.
Final same CPU artifact quality/useful50,useful n,actual DRAM,multiple donor
families/scales and conversion cost remain the original incomplete objective.
