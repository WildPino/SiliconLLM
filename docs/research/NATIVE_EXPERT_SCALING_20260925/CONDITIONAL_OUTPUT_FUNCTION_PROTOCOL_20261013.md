# Fixed observable-route redundant affine output pilot

13 October2026. Freeze before binding/run. Sites0/12/23, both fixed arms
tree16/r128 and tree64/r32, all48 original histories/site. Source-input native
state96/3072 supplies own-normalized BF16 Z and raw F32 Y; original captured
source output gives donor reference. FIT24 objects at ALL3 sites saved before
reading any DEV array. No source/native recurrence/GPU/T4/RESERVED/new teacher.
All object, control and evaluation choices fixed below; no DEV selection.

## Input-only structured routing

PCG64 via pinned NumPy default_rng(seed20261013),16 calls choice(3072,64,
replace=False) in feature order, followed by integers(0,2,size=(16,64))*2-1.
Same indices/signs for all sites. Decode normalized BF16 Z to F64. Each feature
has64 fixed-order signed F64 additions, then divide8 (1024 additions/input).
No Y/domain/case/labels/positions/answer in sketch or route. Sketch arrays
int32 indices/int8 signs, saved with global reference factors.

Balanced full binary tree,16 or64 terminal leaves, heap numbering; left if
feature<=cut, right otherwise. FIT rows follow bound case/token order.
Weights1/(24*n_case), positive, sum1. At node compute weighted feature
variances. Try features in descending variance, feature-index tie order;
only positive variance. Stable sorted distinct value boundaries permitted
if each child has at least (r+1)*2^(remaining_depth-1) rows. Pick boundary
nearest half node weight; lower boundary wins equal distance. Cut is LOWER
observed distinct value, so equality/tie semantics explicit. First feature
with admissible boundary wins. Tree splits recursively, full depth4/6,
every terminal has at least r+1 FIT rows. No admissible split means explicitly
UNSUPPORTED_PARTITION/precondition failure, preserved terminal and no fabricated
operator/admission; no alternative route or post-DEV repair. Hard uint32 IDs,
F32 normalized mass1, exactly one expert. No all-expert per-input scan.

## FIT affine functions

Leaf j, mass m=sum FIT weights. mu_Z/mu_Y are weighted means using w/m.
Centered matrix M=(Y-mu_Y)*sqrt(w/m). Full thin rectangular SVD; leading
r right singular vectors U2048xr, each column sign chosen positive at its
first largest-absolute coordinate. Store all singular values, offsets/counts,
all means, U, B=U.T O and bias=mu_Y-U B mu_Z, original source BF16 O2048x3072
promoted F64. Compare O bits against the qualified native parameter packet.
No covariance inversion/floor/ridge/response-dependent routing. Store F64
bases/factors/bias; dtype means stored bank bytes exceed BF16 source bytes.

Execute ONLY FROM INPUT: bias_j+U_j B_j Z. Also compute diagnostic response
projection mu_Y+U U.T(Y-mu_Y) and leaf-mean-only prediction, explicitly controls.
Globally shared uncentered rank128 U from qualified prior FIT C_Y eigensystem,
B=U.T O/bias0 control. Original dense F64 O Z control quantifies difference
against stored original native F32 scores. These are F64 operator assays,
not new native kernels. Preserve score->F32->BF16 RNE semantics and original
donor comparison scope; source multiplier outside this stored mixer output
is not applied again. Other layer composition remains unproved.

Weighted uncentered FIT Y energy = sum leaf mass*||mu_Y||^2 + sum leaf mass*
centered_energy. Discarded_energy=sum mass*tail(sigma^2). Replay all FIT
response projection errors independently and compare to discarded energy.
This is a finite fixed-partition local affine proxy; no claim it excludes
other routes/nonlinear banks or proves future capacity. Both coefficient
budgets and runtime input-only computation counted; copies are representational
redundancy and do not demonstrate new donor knowledge or huge useful n.

## Frozen gates and complete independent audit

Store both F64 main predictions/all48/site, both uint32 routes/sharedF32 mass
for every history; all bank/global matrices and full local spectra. Controls'
full predictions are recomputed by independent auditor and their per-case/
centered/label/domain/worst metrics retained, not duplicated as large arrays.
Each case/each arm raw-score relative RMS<=.01 and rounded BF16 output versus
baseline<=.01. Donor-relative rounded output gates: mean<=.1, worst<=.2,
mean centered<=.1, worst domain mean<=.2, for FIT and DEV separately, unchanged.
Poor quality DOES NOT stop complete evaluation/audit or loosen criteria.

Numerics: F64 array relative RMS<=1e-10; metadata abs(x-y)/max(abs(y),1)<=1e-10;
local U orth max<=1e-10/eigen residual ||M.T M U-U sigma^2||/||M||_F^2<=1e-10;
independent full squared singular spectrum/max sigma^2<=1e-8. Bias subtraction
backward error scaled by ||mu_Y||+||U B mu_Z||<=1e-10, avoiding artificial
cancellation amplification. Explicit denominator guard1e-30 for numeric
norm metrics, including zero-energy leaves; zero modes/counts retained.
Dense real versus original native score each history<=1e-6. Full independently
recomputed IDs/masses exact; compare independent predicted BF16 bits, no waiver
on differences. Original native is reference, no bit-equivalence presumption.

Auditor does not import producer. Independent recursive tree/weighted mean
construction, separate full thin SVD/spectral checks every leaf, all bases/
factor identities/source coefficient bits. Full32-row independent factor/
projection/dense/global predictions, independent low-bit BF16 RNE, every
stored F64 prediction word/route/mass, all case/source metrics/gates/aggregates,
energy decompositions and before/after hashes. Numeric failure remains visible
even if quality fails; full audit required. Finite F64 bounds only.

## Custody/resources/decision

Prepare/run900s/audit1200s/OS6GiB/new ordered E<=3GiB. CPU0..5,
OPENBLAS/OMP/MKL threads6 before interpreter; pinned Python/actual resolved
NumPy ABI/linalg/BLAS/psutil, original source/checkpoint/config/native exports,
all144 native and source-output packets/prior rank arrays/report/adjudications,
code/helper/protocol/storage/canonical corpus/engine/foreign files. Full
before/after hashes, source freeze before binding and launch, terminal with
PID+creation/exit/resource snapshot. No cold-DRAM/timing/whole-model rate.

If quality conserved, actual rounded native kernels and physical expert DRAM
next; then source generator/ternary LUT/full source-informed layers and OWN
histories. If these fixed banks fail, isolate observed route/local geometry/
generalization defect and choose a mathematically motivated correction rather
than repeating the unchanged bank. Compact shared causal hidden, useful ternary
n/10B/100B, multiple donors/scales, conversion O(n) storage and SAME useful50
remain required. Goal remains active/incomplete regardless of local outcome.
