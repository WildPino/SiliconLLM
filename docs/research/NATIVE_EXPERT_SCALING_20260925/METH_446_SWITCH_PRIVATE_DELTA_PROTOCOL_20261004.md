# M446: single-donor private matrix deltas, coupled permutations and full-matrix control

Freeze NEW controller/math/THIS protocol before first import/compile/matching/
eigensolve/model forward. Previous turn is PROGRESS:443 established native
identity effect,444/445 ruled out the fixed shared-output bases and weighting-only
correction. Their collective output rank does not bound a PRIVATE matrix's rank:
many independent rank1 private matrices can span many output directions.
This pilot changes to per-expert matrix bases and preserves private ReLU gates.
It does not reopen440 or reinterpret444/445 asPASS.

## Fixed source and population

Original source128 decoder bank11 only, source core retained. Expert0 is the
fixed shared reference; target IDs1,64,127 predeclared independent of utility.
ALL1008development and336consumed validation positions from the existing24paired
books/4cases/14positions, original source128 normalized inputs. No18-anchor
subsampling. Each of the FOUR functions is forced at every input for this
functional diagnostic. Natural captured selected-ID counts reported separately;
forcing does not create new routed exposure or an independently excluded test.

Bind immutable443 raw
b8e8614fb28ad50b91865dba7486c6e4cc6bb8c00cee3bbfe899ab495af5bf80,
445 raw ddd1cd0d8c49e5e6a6b5305eca54e46e12f528ebe8e508dcb175661020b8685f,
418 capture4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829,
420 first-failure/native baselines17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427.
Require prior apparatus gates/native identity and no passing balanced445 ranks.
Fresh all inherited raw/helper/output inventories,384baseline archives,
source128 descriptor/full7.542GB payload/manifest/recovery parser, original374/389
binaries. Physical helper equality to filtered HEAD. No source256 values read
or mapped anew. Exact SWFUN001 dimensions/length/source and decoder IDs/states;
keys[1008:] exactly443's validation keys. Source payload size/mtime unchanged.

Before any compression: ALL1344 original captured RMS input/WI raw/ReLU/WO
outputs byte-exact native418. Then evaluate four forced native functions at
all1344inputs; their naturally selected positions byte-match captures.
No new whole head/softmax/route/generation replay is claimed by this FFN test.

## Permutation is an exact native candidate, scaling is not assumed

For each target, one alignment minimizes summed squared distances of normalized
WI rows against reference0, using scipy.optimize.linear_sum_assignment. Zero
rows have denominator1 and remain zero. Cost is F64 norm²+norm²-2dot, with no
metric-dependent clipping; allow only1e-10 rounding negativity. Returned row
indices and permutation bijection validated, matched total<=identity total at
1e-10 relative scale. This is optimal for THIS unit-WI assignment cost, not
optimal function alignment, joint WO alignment or best delta compressibility.

Only permutation P is applied, no positive rescaling: target WI rows/scales are
permuted, target WO columns permuted by the same indices, WO row scales unchanged.
This preserves real ReLU and the qualified native path: up max/A16 scale is
permutation invariant; I64 dot products with jointly permuted WO/codes have
the same exact sum. Absolute3072*127*32767<2^63, so this rearrangement introduces
no integer overflow. F32 row products/scales/ReLU output values merely reorder.

Qualify ALL3*1344 matched native raw outputs as exact original raw[P], ALL3*1344
native down outputs byte-exact original. This is stronger than assuming a
positive-diagonal gauge is byte invariant. The separately dequantized F64
permuted forward must match its original F64 twin within max relative1e-10.
No source values or original C arithmetic are changed by this diagnostic.

## Three fixed representations, five fixed ranks

Use decoded F64 matrices W=I8codes*F32row_scale, exact represented scalars.
For each target, compare:

1. Full-matrix compression of WI,e and WO,e, no shared base.
2. Private delta in identity coordinates: DeltaWI,e=WI,e-WI,0,
   DeltaWO,e=WO,e-WO,0.
3. SAME private deltas AFTER the coupled neuron permutation.

Fixed ranks16/32/64/96/128 on BOTH projections; factors have PRIVATE bases per
matrix/expert. No rank/grid/reference/ID/metric/seed selection from outcomes.
All parameter fitting uses pretrained weights ONLY; no source-input supervision,
gradient training, target readout or router. Source input populations are used
only for prospective functional evaluation. This is a matrix Frobenius recipe,
not activation/curvature/KL-weighted factorization.

Smaller Gram eigensystems in F64, descending values with deterministic largest-
coordinate sign. PSD, orthonormality, eigen equation and trace checks at1e-10.
Balanced factors split sqrt(singular value) between U and V. Independent
explicit matrix residual agrees with spectral tail within1e-10 of matrix energy.
Store U/V in F32 and report ACTUAL reconstructed F32-factor Frobenius energy as
well as the F64 optimal tail. Only the F64 spectral result is the optimum under
this metric; stored factors are a quantized approximation. All values finite.
Tiny qualification: independently exhaustive3! assignment and both tall/wide
factor orientations with full-rank reconstruction.

## Candidate arithmetic and private nonlinearities

Full dequantized reference F_e^64(x)=WO,e ReLU(WI,e x), without A16 rounding.
Every1344-position relative L2 discrepancy from the native forced function must
be<=1e-3 for each of4experts, before eligibility. This qualifies a local smooth
shadow, not exact native arithmetic. Failure stops before compression utility.

Full-matrix factor model:
Fhat=Uo Vo ReLU(Ui Vi x).

Private delta model:
Fhat=(WO,0+Uo Vo) ReLU((WI,0+Ui Vi)x).

Implement the full base plus the private low-rank products separately, add BEFORE
the private ReLU, then apply the common down to that activation plus its private
down correction. Do not separate the common/private nonlinearities. Decode
stored F32 factors to F64 for these matrix evaluations, preserving factor
precision without claiming a native factor kernel. Base I8 scalar values decoded
F64. F64 matrix-delta full recomposition must match target matrices<=1e-14 relative.

Compare each candidate's down vector to its target NATIVE forced response at
the same original input. Report every1344 relative error, development/validation
medians, validation p95 and per-validation-book medians. Relative denominator
max(native vector norm,1e-12); not posterior error or truth accuracy.

Controls: remove BOTH private projections, giving the SAME full-base F64 shadow;
cycle target private WI/WO pairs1→64→127→1 at each fixed rank/gauge, retaining
reference0/input/arithmetic. Report median-relative permutation harm with fixed
.01 DESCRIPTIVE threshold, not a capacity/generation gate. Full-matrix controls
have the same number/precision of private factor coefficients; delta arms pay
the additional shared base. No mask/ID/exposure selection or unavailable foreign
input/core is introduced.

## Prospective candidate eligibility and complete cost boundary

A candidate is locally eligible ONLY when ALL:

- validation median function relative error<=.05;
- validation p95 function relative error<=.10;
- each actual stored-F32 matrix factor retains>=.95 Frobenius energy;
- nominal128-bank storage is less than original I8 bank storage.

An eligible recipe requires those gates for ALL3fixed targets at the same rank/
gauge. Report all45 candidates and ALL controls even after failures. These
geometry/storage gates are not original-relative head/task/accepted-rate gates.
No cheap shadow is assumed to preserve native factor/head behavior.

Source dimensions D768/M3072, each I8 expert has2DM+4(D+M)=4,733,952B including
row scales. F32 private coefficient bytes=2r(D+M)*4. Shared base is one source
I8 expert; private-bank extrapolation uses127private targets+reference base.
Conservatively charge3072*4 permutation provenance bytes per private expert for
the matched arm (conversion metadata, not an active gather requirement). Full
factor control extrapolates128factor experts. Headers/fallback/cache unknown.
Only3target experts transformed; these extrapolations do NOT create128experts.

Top1 arithmetic counts: full factor2r(D+M); base/delta2DM+2r(D+M). A full shared
base ADDS arithmetic at top1. Latency/actualDRAM/LUT is not predicted by counts.
For larger k the shared up and shared down on sum(p_e a_e) can be amortized, but
private nonlinearities must remain; no other-family cost or quality inherited.

## Resources, files and first failure

Admission<=300s, all replay/matching/spectra/candidates/controls/reporting<=600s,
total<=900s from main, module import excluded. Peak RSS/Windows working set<=3GiB,
experiment outputs<=384MiB, admission disk reserve>=2GiB. Guard every hash/native
row/matching/eigensystem/candidate/stage, before crossing the admission boundary.
First binding/replay/shadow/nonfinite/numerical/resource failure retained with
partial output SHA inventory before NEW numbered repair. Never rerun446 or
overwrite paths. Raw JSON retains FIRST error with phase, not a science PASS.

Save reference NPZ (full inputs/pre/probability/keys/four native down arrays,
original base codes/scales), F32 factors/permutations NPZ and all F64 validation
candidate/control down arrays NPZ. Complete raw metrics/spectra/IDs/costs and
SHA/byte inventory. No full-head artifacts, native export or training checkpoint.

CPU0/Torch1/interop1/BLAS1; NumPy2.4.6/Torch2.6.0+cu124/SciPy1.18.1, runtime
BLAS/assignment library hashes recorded. No concurrent modeljob/native timing,
GPU/T4/network/new corpus/source download/C edit. Exact approved publisher daemons
preserved. Original374/389 binaries/engine unchanged. Goal ACTIVE/INCOMPLETE;
useful RAM-n/routing/LUT/DRAM/whole quality/SAMEartifact50/families/~100B OPEN.

After freeze, run ONCE:

`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth446_switch_private_delta_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth446_switch_private_delta_result.json`
