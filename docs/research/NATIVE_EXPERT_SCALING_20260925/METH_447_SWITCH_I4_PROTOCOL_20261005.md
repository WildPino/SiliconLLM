# METH-447: fixed symmetric row-I4 native predictive probe

## Prospective question and decision

Frozen BEFORE first controller/helper import, compile, quantization or forward.
446 rejects the specified private/full Frobenius low-rank recipes, not scalar
precision reduction. 443 proves a native last-bank identity effect and qualifies
the original full readout. Test ONE calibration-free encoding of all128 real
source128 finalbank11 functions. Does it halve coefficient storage while keeping
local original-relative predictive information and a bank identity effect?

If all apparatus and five feasibility gates pass, prepare a NEW separate C format
and active-cost gate. Otherwise retain/close this fixed encoding before C export
or timing. No outcome-dependent precision, block size, clipping or data grid.
Whole quantized-model quality, changed upstream routes, generation/tasks and SAME
artifact>=50 acceptedIDs/s remain required regardless of this local outcome.

## Bound inputs and reproducibility

Original source128 google/switch-base-128 revision
86c815ec05361a33a8b49fc717277da9c0a4e711, 7,415,217,408 unique source parameters.
384 complete420 baseline archives and1936 inventory418 outputs freshly bound,
parents443 and446 raw/helper/output hashes freshly checked. Original source128
I8 payload7,541,946,880B SHA6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe,
manifest3a4892ad2bc846937a3ad7898048df64b9a26dc391a3d02c110fe75cd614511a,
full descriptor/recovery parser, original374/389 binaries unchanged.

All128 finalbank11 experts, WI3072x768 and WO768x3072, encoded in original order.
Exactly336 consumed validation positions: books18..23, cases0..3, positions0..13.
Original teacher prefixes/pair identities fixed by418/420/443; no new data or
training. No inference of fresh held-out quality from consumed diagnostics.
Original core/native RMS input, selected ID and probability, residual, final RMS,
tied I8 head unchanged. Original complete336 heads and all captured intermediate
states/A16 scales/codes must be byte-exact BEFORE I4 predictive conclusions.

CPU0, Torch2.6.0+cu124 threads1/inter-op1, NumPy2.4.6, BLAS1. Actual loaded
runtime paths/binary hashes recorded, including torch_cpu.dll and Torch_C.
All own scientific files/protocol physically equal filtered committed HEAD.
No overlapping model jobs; preserve only the exact separately authorized
pythonw.exe argv[1] D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py.

## Single fixed coefficient transformation

For each original I8 row q8 and represented positive F32 scale s8, define the
source dequantized row w=q8*F64(s8), without a new source float export.
Set s4=F32(max(abs(w))/7), or1 for an all-zero row. Reject nonfinite/nonpositive
represented scales. q4=I8(clip(rint(w/F64(s4)),-7,7)); nearest-even rounding.
No activation-based calibration, learned base, neuron alignment or rank cut.
This does not guarantee unchanged matrix rank or function geometry.

Store q4 two's-complement nibbles, even column in low nibble, odd in high.
Two nibbles per byte, original row/column order, little-endian F32 row scales.
Nibble8 (unsupported-8) is a fault. Allowed signs are0..7 and9..15.
Every256 complete matrix pack inverse must reproduce derived q4; saved bank
reloaded byte-exact. Record zero rows, Frobenius coefficient error, source and
packed byte counts and128 expert fingerprints. Distinct arrays are not useful-n.

Source per-expert bytes2DM+4(D+M)=4,733,952; packedDM+4(D+M)=2,374,656.
128 bank coefficient/scales bytes605,945,856 versus303,955,968 before container
metadata. Actual archive bytes reported. No nominal byte saving implies speed
or DRAM residency/traffic; diagnostic decoded caches are not deployable costs.

## Independent integer/primal qualification

Retain original qualified A16 input/up quantizer, signed-32767..32767, all-zero
scale1. I4 operator decodes cached low/high coefficient halves as I32, sums two
half dot products, applies F64 rowScale then F64 activationScale, casts F32.
Independent reference decodes the complete packed matrix afresh into I64,
computes one I64 dot, then SAME dequant multiplication order. Each actual WI/WO
projection in both I4 arms must have exact integer sums and F32 byte equality.

For width3072, |sum|<=3072*7*32767=704,621,568<2^31; both half sums and their
addition safe. This width/activation proof is checked before operations and does
not certify larger families. The source WI width768 is also safe.
Tiny qualification: all225 signed weight pairs times9 activation extrema pairs;
full3072 positive/negative/cancellation sums; low and high forbidden nibble8;
zero rows and zero A16 input; nearest-even half ties. All must pass.
There are2 I4 arms*336 positions*2 projections=1344 actual projection checks.
This qualifies a Python arithmetic definition, not a C kernel or measured rate.

## Fixed controls and information estimands

Three controls, SAME original p at every position:

1. I4_original_ID: original selected e, both projections quantized.
2. I4_ID_plus1: (e+1) mod128, both projections quantized. No router change.
3. removed: zero function contribution; recompute original final norm/head.

ReLU stays inside the selected function. No cross-expert activation mixing.
Record raw up, down, post-residual, final and head-input states, full32128 F32
logits and actual function IDs for every control. Removal heads must additionally
match parent443 byte-exact; original heads and complete source prefixes retained.

Define p0=softmax(F64(original logits)); KL(p0||pc)=CE(p0,pc)-H(p0).
CE uses the qualified stable helper with explicit F64 caller cast. Independently
compute dot(p0,logp0-logpc), agreement<=1e-10 per position; minKL>=-1e-10,
all finite. Report mean/median/p95/max, six book means, full per-position KL/CE,
argmax change masks and relative state effects. No unlike-metric attenuation.

Identity diagnostic is mean(KL_I4_plus1-KL_I4_correct) on SAME original posterior,
input and amplitude. This is joint bank intervention evidence, not the number of
individually useful experts or observed-routing mutual information E=f(X).

## Predeclared gates

ALL nine apparatus gates must pass. Fixed encoding considered locally promising
only if ALL five feasibility gates pass:

- mean original-relative KL<=.01 nats;
- EVERY book mean original-relative KL<=.05 nats;
- argmax changed fraction<=.01 (at most3 of336 positions);
- I4_ID_plus1 increases mean source KL over I4_original_ID by>=.01 nats;
- nominal packed bank coefficient/row-scale byte ratio<=.51 of original I8.

These are screening thresholds, not a full donor-equivalence or task-quality
budget. Failure closes this SINGLE fixed row-I4 encoding on this diagnostic;
it does not refute all I4, mixed precision, data-aware quantization or nonlinear
conditional representations. Do not relax gates or rerun after observing results.

## Resources, retention and stops

CPU-only. No optimizer updates, model downloads/new corpus, GPU or C edits.
Hard limits: admission<=300s, numerical/conversion/reporting<=300s,
total<=600s, peakprocess<=3GiB, retained outputdirectory<=512MiB, free disk>=2GiB
at admission. Expected roughly100..220s total, output about520MB including
packed bank, complete prefixes, original and three counterfactual full heads.
The bounds are checked during hashes and after each conversion/forward.
Do not extend a failed budget; retain FIRST failure before a NEW numbered repair.

Output results/native_expert_scaling/meth447_switch_i4/ must not preexist;
raw docs/research/NATIVE_EXPERT_SCALING_20260925/meth447_switch_i4_result.json
or its .failure.json must not preexist. Hash EVERY retained output. Preserve all
source/helper/protocol files after observation, original engine and binaries,
unrelated work. No repeated controller or overwritten output.

Command ONCE after freeze commit:

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth447_switch_i4_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth447_switch_i4_result.json

Resume whole goal from result. No use of original source-specific rates for a new
I4 artifact; no full model/GPU/runtime superiority or100B/n scaling claim.
