# METH-448: one block64-I4 native predictive probe

## Prospective question and decision

Freeze this protocol, math and controller BEFORE first import, compile,
quantization or forward. Previous447 row-I4 ALL9apparatusPASS/4of5feasibilityPASS
but4argmax changes>3; that fixed recipe stays CLOSED and immutable. New variable
is scale granularity64; all coefficients/IDs/source input and original readout
stay present. No precision/blocksize/clipping/grid, calibration or optimizer.

At D768/M3072, bytes_blockI4(B)=2DM*(1/2+4/B), sourceI8=4,733,952B/expert.
B32 gives2,949,120B, exceeding a prospective60% storage cap; B64 gives2,654,208B,
below it.64 is the SMALLEST power of two satisfying that budget; selected by
algebraic bytes, not validation error.128-bank nominal339,738,624B. No guaranteed
monotone function/KL improvement from smaller scales is assumed.

If ALL apparatus and five local gates pass, prepare a NEW native C format and
active-cost gate. If not, retain and reassess representation before C export,
timing or blocksize/precision sweeps. Full native whole-model source-relative
held-out/generation/tasks and SAMEartifact>=50 acceptedIDs/s remain required.
This consumed local assay does not prove useful-n scaling or generality.

## Bound inputs and arithmetic retained from447

Original google/switch-base-128 revision86c815ec05361a33a8b49fc717277da9c0a4e711,
7,415,217,408 unique source parameters. Payload7,541,946,880B,
SHA6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe,
manifest3a4892ad2bc846937a3ad7898048df64b9a26dc391a3d02c110fe75cd614511a.
Full source tensor descriptors/recovery parser freshly checked. Parents447,
443/446/418/420 raw/helper/output hashes freshly checked, including ALL384420
complete baseline archives and418 inventory. Original374/389 binary hashes bound.
All own scientific files/protocol physically equal committed filtered HEAD.

All128 real finalbank11 functions, WI3072x768 and WO768x3072, original order.
Exactly336 consumed original teacher-prefix positions: books18..23,cases0..3,
positions0..13, same pairing/source IDs as418/420/443/447. No new documents,
samples, source float download, router/readout fit or teacher-mask deployment.
Original native RMS input and A16 quantizer, ReLU, selected probability,
residual/final RMS and tied I8 head stay fixed. ALL336 original complete heads
and states/scales/codes must replay byte-exact before predictive conclusions.

CPU0/Torch2.6.0+cu124 threads1/inter-op1, NumPy2.4.6/BLAS1. Actual runtime
paths/SHA256 recorded, including torch_cpu.dll and Torch_C. No overlapping model
jobs. Preserve only exact approved pythonw.exe with2argv, argv[1]
D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py. No daemon/job management.

## Single fixed coefficient transformation

Reuse frozen447 signed-nibble pack/unpack as a qualified component; do NOT run
447 controller. For every original I8 matrix row, original positive F32 scale s8,
decode w=q8*F64(s8). Split consecutive ORIGINAL columns into nonoverlapping64.
For each block define s4=F32(max(abs(w_block))/7), or1 for all-zero block.
Reject nonfinite/nonpositive represented scales. Derive q4=I8(clip(rint(w/s4),
-7,7)) using F64 division by represented F32 scale and nearest-even rounding.
Keep every coefficient, no neuron alignment/rank truncation/input permutation.

Two two's-complement nibbles/byte: even column low, odd high, nibble8 unsupported.
Same packed coefficient layout as447; per-block F32 scales shape WI[3072,12],
WO[768,48], original row and ascending block order. All256 full matrix inverse
packs exact and saved complete bank must reload byte-exact. Record each matrix
Frobenius error/zero-block count/maximum coefficient error and128 fingerprints.
Fingerprint uniqueness/106 original routed IDs are not useful-function counts.

## Integer block products and floating accumulation

Native original A16 activation quantizer returns F32 activationScale and I16
codes-32767..32767; zero input scale1. For every I4 matrix application:

1. Prospective pair implementation caches low/high I32 halves, shape rows x
   blocks x32. NumPy einsum(optimize=False,dtype=I32) computes two half block
   dots, then adds them. No sum across blocks in I32.
2. Independent reference freshly decodes FULL packed matrix into I64, forms
   elementwise products against A16 codes and sums each64 coefficients in I64.
   All block sums must agree exactly for every actual application.
3. Starting F64 vector+0, ascending blocks b=0..B-1: add
   F64(integerBlockSum_b)*F64(representedBlockScale_b). After all blocks multiply
   by F64(activationScale), cast F32 ONCE. Reference uses explicit same order
   through separate multiply/add primitives; resulting F32 bytes exact and finite.

Each half sum and full block sum safe:64*7*32767=14,679,616<2^31. This proof
certifies integer block accumulation only, not arbitrary F64 overflow or C SIMD.
Cache/unpacking in Python diagnostic is not measured native active traffic.
ReLU remains inside each selected function; WO A16 quantizes actual new up.

Tiny qualification BEFORE real conversion/quality: reuse common447 pair/extrema/
nearest-even checks; all225 weight pairs times9 activation extrema repeated into
full64 block;768/3072 positive/negative/cancellation block bounds; dyadic mixed
scales checked against independent exact Fraction arithmetic through F32;
zero blocks/activation and invalid scale/last low/high forbidden nibble faults.
All must pass.2I4 arms*336positions*2projections=1344 real block-primal checks.
No scientific parameter is changed if a qualifier fails; retain FIRST failure.

## Fixed interventions and information metrics

Three controls at SAME original source input/probability:

- I4B64_original_ID: original selected function, WI/WO both block64-I4.
- I4B64_ID_plus1: (original ID+1) mod128, WI/WO both block64-I4.
- removed: no function contribution; original finalnorm/head recomputed,
  complete heads additionally byte-exact443 removal.

Retain raw up/down/post/final/head_input, full32128 F32 logits and actual IDs.
Original complete336 heads and structured prefixes/scores/IDs retained too.
Target p0=softmax(F64(original logits)), KL=CE(p0,candidate)-H(p0), qualified
stable loss called with explicit F64 logits. Independently dot(p0,logp0-logpc)
must agree per position<=1e-10; finite, KL>=-1e-10. Mean/median/p95/max, six
book means, per-position KL/CE, changed argmax masks and state effects recorded.
Identity effect=mean(KL_IDplus1-KL_correct) relative to SAME original posterior.
No prediction-error/Frobenius ratio or observed-routing mutual information claim.

## Fixed gates

ALL nine apparatus gates required. Locally promising only if ALL five pass:

- correct-ID mean source KL<=.01 nats;
- EVERY book correct-ID mean source KL<=.05 nats;
- changed argmax fraction<=.01, at most3/336;
- ID+1 increases mean source KL over correct-ID by>=.01 nats;
- NEW nominal packed coefficient/block-scale ratio<=.60 of source I8.

The four predictive/identity gates are unchanged447. The byte cap and
block-dot operator are NEW explicit prospective definitions. No gate relaxation
after observation or selecting only previously changed positions. All336 used.
No local gate implies fresh held-out whole donor equivalence or error composition.

## Resources, retention and stop

CPU-only, zero updates, no new data/download/GPU or C modification. Expected
roughly130..240s total; about555MB retained including larger block scales.
Hard admission<=300s, numeric/conversion/reporting<=300s, total<=600s,
peakprocess<=3GiB, outputdirectory<=576MiB (larger prescribed block scales,
prospective change), free disk>=2GiB at admission. Guard hashes, each converted
matrix and each forward; retain FIRST resource/numeric failure without extension.

results/native_expert_scaling/meth448_switch_block_i4/ and raw/failure files
must not preexist. All nine complete retained outputs SHA256/size inventoried.
All447 scientific files/outputs remain immutable; do not rerun a completed stage.
Current engine/source payload/original374/389 binaries and unrelated work kept.

Command ONCE after freeze commit:

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth448_switch_block_i4_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth448_switch_block_i4_result.json

Update INDEX/METHOD/current resumption from result. Goal ACTIVE/INCOMPLETE;
no inherited original rate, CPU LUT, actual DRAM, useful larger n, second-family
or100B proof from the local block encoding.
