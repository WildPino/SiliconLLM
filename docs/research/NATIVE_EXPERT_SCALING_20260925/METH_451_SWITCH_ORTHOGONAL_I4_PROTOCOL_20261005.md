# METH-451: fixed signed orthogonal input bases before block64-I4

## Question, reused evidence and decision

Freeze math/controller/this protocol BEFORE first import, compile, transform or
forward.447 row-I4 and448 block64-I4 arithmetic passed but argmax4/9 versus<=3
failed.450 same-math margin diagnosis after449's retained first stop proves
all4row/9block changed pairs cross in smooth F64 source readout too. Genuine
head-input state displacement suffices for these pairs. Do not change gates,
sweep precision/blocksize/sign seeds or tune on these consumed examples.

NEW variable: ONE fixed signed block-orthogonal INPUT basis on WI/WO before
SAME block64 symmetric scalar-I4. It may spread coefficient magnitudes before
rounding; useful decision-margin improvement is unknown. All coefficients,
original128 IDs and original-coordinate private ReLU retained. No learned basis,
clipping, calibration, rank truncation, new data, GPU or optimizer.

First a rotated UNCOMPRESSED control must show that the new activation arithmetic
alone preserves strict source-relative local quality. Only then evaluate compact
correct-ID/ID+1/removed. If either eligibility or compact gates fail, retain and
reassess before C timing/export; no new seed/grid. If ALL pass, only a separately
frozen actual C primal/complete-cost gate is licensed. Whole-model fresh held-out/
generation/tasks AND>=50 acceptedIDs/s on SAME artifact, useful RAM-scaled n,
routing/LUT/actual DRAM, second family and actual~100B remain required.

## Source, cohort and immutable component bindings

Original google/switch-base-128 revision86c815ec05361a33a8b49fc717277da9c0a4e711,
7,415,217,408 unique source positions. I8 payload7,541,946,880B SHA256
6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe,
manifest3a4892ad2bc846937a3ad7898048df64b9a26dc391a3d02c110fe75cd614511a.
Fresh whole hashes/export descriptor/ALL source tensor names/offset recovery.
Parents443/446/447/448/450/418/420 and every referenced helper/output freshly
bound, ALL384 complete420 baseline archive hashes and418 capture inventory.
Original374/389 and engine54194c36b625... preserved; actual Torch_C/torch_cpu.dll/
NumPy OpenBLAS/OpenMP paths/SHA recorded. No old controller is run.

All128 REAL final-bank11 functions: WI3072x768, WO768x3072. SAME336 consumed
original teacher-prefix positions: books18..23,cases0..3,positions0..13, exact
418/420/443/447/448 source/pairing keys. No selection based on old flips, fits,
new corpus or donor float download. ALL336 original native FFNs/state/A16/
probability/full32128 heads must replay BYTE-EXACT BEFORE real coefficient
transformation. Original input/router scores/p/residual/final RMS/tied head
remain local controls. Their captured values cannot shortcut all-bank deployment.

## Fixed signed basis and exact-real identity

Column convention: f_e(x)=WO_e ReLU(WI_e x). Set

    x'=Q_D x; WI'_e=WI_e Q_D^T
    a_e=ReLU(WI'_e x'); a'_e=Q_M a_e; WO'_e=WO_e Q_M^T.

Then WO'_e a'_e=f_e(x) EXACT in real arithmetic before quantization. ReLU stays
in original neuron coordinates BETWEEN the two linear projections and BEFORE
Q_M. No rotation through ReLU or changed output/router/readout coordinates.

Q_D=diag(H256,H256,H256)S_WI; Q_M=diag(H1024,H1024,H1024)S_WO. Walsh matrix
H_N[r,c]=(-1)^popcount(r AND c)/sqrt(N), Sylvester output ordering. These are
block orthogonal, not full768/3072 Hadamard or guaranteed global incoherence.
For label WI,width768 or WO,width3072 and indexi=0..width-1, signs[i]=+1 iff
first byte of SHA256 ASCII 'SiliconLLM|METH451|'+label+'|'+decimal(i) has low-bit1;
else-1. One specification fixed before observation, shared across experts, stored
as I8 vectors and SHA256. No random generator/library seed ambiguity or seed search.

F64 activation transform: promote represented F32 x; multiply signs; reshape
contiguous N-column blocks; ascending butterfly steps1,2,4,...,N/2. At each stage
copy both halves BEFORE left+right/left-right assignment. Divide F64 by16(WI) or
32(WO), cast F32 ONCE, then original frozen A16 quantizer absmax/32767 F32,
nearest-even F32 division, clamp+-32767, zeroScale1. WO input is actual new ReLU
activation. Every actual transform independently checked against explicit F64
Walsh matrix-vector products: relative norm<=1e-12 with denominator max(norm,1e-12).
Independent transform is a qualifier, not deployment cost; actual FWHT overhead
must later be measured in C. No native equality inferred from real orthogonality.

## Real coefficient witnesses, precision and operators

Original coefficients are represented q8*s8. Require actual q8 within[-127,127]
and positive finite F32 scales. Sign q8 in I32, unnormalized fixed FWHT in I32:
max absolute forward<=127N, inverse conservative bound<=127N^2; at1024 latter
133,169,152<2^31. ALL256 matrices must inverse-transform to N*(q8*signs) EXACT.
Additional independent parity/I64 sums at rows0/middle/last, ALL3 blocks and
output coordinates0/1/N/2/N-1 qualify orientation. These selected sums plus tiny
full Walsh witnesses are distinct from exhaustive independent ALL-matrix products.
Retain max magnitude and SHA of each transformed I32 matrix. No intermediate I8
requantization. Exact dyadic scale normalization F64(s8)/16 or32, coefficient
integer*normalized scale has <=41 significand bits in this geometry, fits F64.

Encode transformed F64 coefficients with unchanged448 block64 rule: positive
F32 scale absmax/7, zeroBlockScale1; F64 division/nearest-even/clamp-7..7; even
column low nibble/odd high, nibble8 rejected. All256 pack inverses and complete
saved bank/signs reload byte-exact. SAME448 PackedOperator: separate low/high
I32 block sums versus fresh full-decode elementwise I64 reference, then ascending
F64 block-scale accumulation, activationScale once, F32 cast once. Block bound
64*7*32767=14,679,616. No floating-order/numeric threshold changes.

Rotated UNCOMPRESSED control uses transformed source I32 coefficients, F64 source
scale/16 or32, I64 dot with ACTUAL new A16 codes. Independent I64 elementwise
product/sum must be EXACT; integer then normalizedScale then activationScale in
F64, F32 cast once must match separate multiply primitives. Bound
width*(127N)*32767<2^53 AND<2^63, so integer promotion exact; scaled products may
round in F64 as specified. This higher-precision source control is NOT a compact
export. Derive selected matrices from the unchanged source payload via cache<=8;
record ALL256 coefficient witnesses, not a new multi-GB precision-control bank.

Tiny BEFORE real transformation: unchanged448 complete pack/primal/Fraction/
extrema/zero/fault checks; full explicit integer Walsh orthogonality, all tested
coefficient products/inverses at N4/16/256/1024, dyadic activation casts and exact
4x4 private-ReLU/WI/WO identity, zero transformed block scales. All required.

Compact nominal bank339,738,624B plus3840B shared signs=339,742,464B before archive
headers; original605,945,856B.128 fingerprints are parameter-representation
identities, not counts of useful functions or uniqueness of effective maps.

## Staged interventions and fixed local gates

At SAME source original input/p and ORIGINAL posterior p0:

1. rotated_source_original_ID: higher-precision transformed source coefficients.
2. RI4B64_original_ID: original selected ID, both projections compact.
3. RI4B64_ID_plus1: (sourceID+1)mod128, same compact basis/codec.
4. removed: original residual without function, full heads byte-exact443 removal.

Retain original complete336 heads/prefixes and each computed arm's full logits,
up_raw/down/post/final/head_input/actualIDs. Conditional amplitude is ORIGINAL p.
Target p0=softmax(F64(original logits)); stable source CE callerF64; KL=CE-H,
independent dot(p0,logp0-logpc)<=1e-10 per position, finite and KL>=-1e-10.
Mean/median/p95/max/per-position KL/CE/argmax/state effects and six book means.

UNCOMPRESSED eligibility gates, all required BEFORE compact predictions:

- mean source KL<=1e-6 nats;
- EVERY book mean source KL<=1e-5 nats;
- ZERO changed argmax positions among336.

If any fails, complete raw outcome records the failed eligibility, compact gates
null and interventions explicitly SKIPPED. Keep built compact bank and source
control outputs; no compact quality conclusion or parameter repair inferred.
If eligible, ALL five compressed gates unchanged448:

- mean source KL<=.01;
- EVERY book mean source KL<=.05;
- changed argmax fraction<=.01, at most3of336;
- mean source KL(ID+1)-KL(correct)>=.01;
- nominal stored compact coefficient/block scales/SIGNS ratio<=.60 of source I8.

ALL implemented apparatus gates required, actual projection counters recorded:
672 rotated-source projections;1344 compact projections if eligible, zero if not.
Independent actual basis checks equal2016 in eligible run,672 otherwise. These
consumed-prefix gates are prospective local feasibility, not fresh whole quality.

## Prospective resources, first-stop retention and command

CPU0, NumPy2.4.6/BLAS1, Torch2.6.0+cu124 threads1/inter-op1, no GPU/updates/new
source data/C change. No concurrent Python/model jobs; preserve ONLY exact
pythonw.exe with2 argv and argv[1] D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py.
No daemon/job management or routine Graphify.

Expected200..400s/2.5..3.5GB process/about607MB output when all arms run. NEW
prospective bounds charge FWHT qualification/conversion, higher-precision cache
and extra complete source-control heads/states: admission<=300s,
numeric+conversion+reporting<=600s, total<=900s, peak<=4GiB, outputdir<=640MiB,
free disk>=2GiB. Guard every hash block/converted matrix/forward. FIRST apparatus,
numeric or resource failure retained before any NEW numbered repair; no extension
or scientific edit after observations. Eligibility failure is valid completed
local evidence, not a silent gate change or invalid apparatus.

All new output/raw/failure paths must not preexist. Complete output sizes/SHA256
and source/runtime bindings retained. Original payload/engine/binaries, all prior
failed recipes and unrelated work unchanged. No complete all-bank export or C
speed/DRAM inference from coefficient counts.

Command ONCE after source/protocol freeze and operational resumption:

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth451_switch_orthogonal_i4_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth451_switch_orthogonal_i4_result.json

Update INDEX/METHOD/current resumption from complete outcome. Goal ACTIVE /
INCOMPLETE. No larger useful n/families100B/physical DRAM/SAMEartifact50 claim.
