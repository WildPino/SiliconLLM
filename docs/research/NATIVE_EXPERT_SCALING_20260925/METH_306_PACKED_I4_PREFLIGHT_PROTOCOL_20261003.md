# METH-306: packed-I4/four-row operator cost candidate

Freeze new C/controller/protocol/phase60 selector BEFORE observations. 305's
exact signed-I8 kernel passes fidelity but FAILS14ms at21.633–22.077ms.
Change storage/precision to packed signedI4 and computation to four-row tiles,
with tiny child query matrices executed serially. This is a combined native
candidate, not an isolated estimate of either change. No training yet.

## Representation, resident capacity and arithmetic

Keep original303 geometry:26 layers/D1536/MLA8heads/KV512/routed128/shared256/
dense1024,64 source parents×TEN children=640 functions per layer,top4×top1.
All9,437,184,000 routed coefficients and9,651,773,440 total generated matrix
coefficients remain, but stored code bytes halve to4,825,886,720. Allocation
5,106,289,792B plus static globals/input buffers. Active273,571,840 coefficients
become136,785,920 code bytes; FP32 row scales unchanged1,902,656B. Full source
Q6 head/router/norm/bias/one embedding row and selected child keys priced:
complete descriptor310,571,680B/token required<=560MB. No physical DRAM claim.

Seed the SAME original303 logical signed weight samples, map old-128 to0,
then q4=round_even(w8*7/127) in[-7,7], store nibble q4+8, two consecutive
coefficients low then high. Row scale=(1+hash%128/128)/(7*sqrt(input_width)).
This defines synthetic Q4 fixtures, not a validated quantization of trained
donor/student parameters. Precision loss and whole-model quality are UNKNOWN.

Inputs retain303 dynamic signed-I8[-127,127] max/127/ties-even quantization,
plus sum_x. Expand16 packed bytes to32 unsigned nibbles[0,15] in correct
input order, multiply signed input bytes using AVX2 unsigned×signed pair sum,
widen pairs toI32, accumulate and subtract8*sum_x. Pair absolute maximum
2*15*127=3810<32767, full unsigned<=1536*15*127=2,926,080 and correction
<=8*1536*127=1,560,576, safelyI32. Domain controls include representable q=-8
although fixture initialization only generates[-7,7]. No pair saturation.

Four adjacent output rows share each input vector load; row-major packed
weight rows/selected-bank offsets retained. Every output width divisible4;
tiles cannot cross a bank. One non-inlined general matrix dispatcher handles
all projections. Matrices with total active coefficients<32768 run serially
(only16-output child query at this geometry); other matrices retain six
threads. This changes scheduling, not the defined packed dot or source router.

## Independent controls and unchanged gates

- Bind immutable303 C/controller/prior and old source/helper hashes; fresh
  source/spec/component bindings exact303. Entire execute/source routing/head
  fixture composition byte-exact303, though new matrix values/child IDs differ.
- Independently regenerate FIRST/LAST eight packed bytes of EVERY bank,
 97,194 edge checks. On eight spanning rows of EVERY active bank, scalarI64
  nibble decoding matches single-row AVX2 and tiled/scaled output byte-exact;
 7,176 rows, pooled FP64 scaled relativeL2<=1e-6. All outputs finite.
- All4,080 representable nibble/input pairs exercise four shifted row weights,
 16,320 tile-lane checks, all exact scalarI64/single-row/four-row native.
 30,976 adjacent pairs (all16 unsigned nibbles at each position×eleven source
  signed input extrema/odd/zero values at each position) verify pair lane,
  dot and row correction. Signed extrema, quantizer zero/ties/NaN, sum_x+1
  and a flipped packed low nibble faults must be detected independently.
- Keep source macro/child independent25-layer probability/rank/gate/address
  controls, actual Q6 full-head64-row decoded relativeL2<=1e-5 and packed-head/
  bad-bank fault sensitivity. Same10 fixtures×3 repetitions, two warmup/eight
  measured each; all30 own output/route hashes repeat exactly. Changed Q4
  precision explicitly DOES NOT require equivalence to303's I8 hashes.
- Verify resident/active coefficients/packed byte counts and allocation.
  Max/min three medians<=1.10 or stop inconclusive; EACH median<=14ms or
  reject this UNCHANGED candidate before teacher collection/training. No
  subset selection/regrading. Precision/scheduling effects cannot be separated
  from one combined candidate.303/304/305 failures remain unmodified.
- Pass licenses only a separately frozen real complete teacher-function fit
  and precision-quality assay; no automatic whole source quality/useful new n.

Timing includes original input conversion/norm, Q4 unpack/dots/scales, source
macro routing/child search, selected complete nonlinear FFNs/mixture and actual
original Q6 full head/Q8_K activation conversion. Initialization/control/hash/
logging/input-ready context preparation outside. Causal attention/cache/RoPE/
embedding/layer-to-layer residual composition still missing; keep6ms balance
for those before any50token/s claim. No assumed cache/DRAM or large-parent rate.

## Resources and reproduction

Local Ryzen5 3600X/six CPU threads, no overlapping model/performance job,
GPU/T4/download/delegation. Expected seconds to CPU minutes;600s/24GiB child,
120s compile/>=16GiB available. Fresh~172MB source components and~5.1GB bank
allocation; isolated write-once files and preserved failures before repair.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth306_packed_i4_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth306_packed_i4_preflight_result.json
```

Final method still needs real pretrained knowledge transfer, heldout/generation/
task quality, useful RAM-dependent n, SAMEartifact accepted>=50batch1token/s
and demonstrated multiple donor families/scales. Synthetic bank cost is enabling.
