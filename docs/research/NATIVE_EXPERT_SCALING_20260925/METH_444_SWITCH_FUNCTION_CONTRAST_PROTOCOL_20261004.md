# M444: single-donor common/private functional geometry, offline oracle

Freeze NEW444 controller/THIS protocol before first import/forward/eigensolve.
443 established source128 final-bank function identity affects original native
predictions (ID+1 mean KL.17646).441's almost identity-insensitive foreign fit
does not establish original functions are interchangeable or isolate a readout
failure.440 stays CLOSED. This changes the problem to source functional geometry
before any new common/private representation training.

## Fixed population and interventions

Only original source128 bank11, same qualified418/420 I8/A16/F32 operators.
One anchor per original paired book b=0..23:
case=b mod4, decoder position=(5*b) mod14. Books0..17 are18 DEVELOPMENT anchors;
books18..23 are6 VALIDATION anchors. These anchors are fixed by index, not losses,
IDs, probability, activation magnitude or outcomes. Original S29/T14 consumed
teacher prefixes; no independent corpus, whole rollouts or new held-out claim.

At every anchor, evaluate ALL128 native F_e(x)=WO_e ReLU(WI_e x) using the same
captured original normalized x. Reuse original pre-residual h and chosen router
amplitude p for all alternative identities. This is an intervention on the
function pointer, no actual uniform model routing or score renormalization.
Uniform functions/anchors define this diagnostic estimand; naturally routed
exposure and rare-domain usefulness are different quantities.

Fresh bind443 raw SHA
b8e8614fb28ad50b91865dba7486c6e4cc6bb8c00cee3bbfe899ab495af5bf80;
require all apparatus and primary native-identity gates. Bind fresh parent/
418/420 raw/helpers/outputs, all384 baseline archives, original374/389 binaries,
source128 full7.542GB payload/manifest/export descriptor and manifest parser.
No source256 payload is mapped/read anew. Filtered-HEAD physical helper equality.
Exact capture headers/lengths/IDs/paired keys/router input and chosen ID required.
Parent443 validates validation key suffix indices. Source size/mtime unchanged.

BEFORE geometry: replay ALL24 original states/WI/WO/head A16 codes/scales,
full128-score selected probability and full32128 logits byte-exact418/420.
For all128 functions at the FIRST fixed anchor, compare complete WI/WO outputs
with the independent418 matrix/quant primitive byte-exact. Whenever an enumerated
function matches the captured chosen ID, raw/ReLU/down must match its capture.
Every collection/analysis value must be finite. No new derivative or C code.

## Functional algebra, not averaging internal weight coordinates

For each anchor i, define in F64

barF_i=(1/128) sum_e F_e(x_i), d_i,e=F_e(x_i)-barF_i.

Pythagoras: sum_e||F_e||²=128||barF_i||²+sum_e||d_i,e||²,
qualified relative1e-10. These residual-output coordinates are shared within
one donor. This avoids assuming internal ReLU neurons of different experts
have matched permutations/scales. It does not align or decompose their matrices.

Fit ONLY on development: Z contains18*128=2304 centered F64 responses;
C=Z^T Z/2304. Symmetric eigh, eigenvalues descending, each eigenvector's largest
absolute coordinate signed positive. Qualify orthonormality/eigen equation,
trace/spectrum and PSD within1e-10 (relative where stated by controller).
Save full covariance/eigenvalues/basis. Full768-dimensional F64 reconstruction
must have relative error<=1e-10. Negative rounding-scale eigenvalues are retained,
not reinterpreted as physical negative covariance.

Fixed diagnostic ranks r in {0,32,64,128,256}; B_r is the first r development
eigenvectors. Reconstruct Fhat_i,e=barF_i+B_r B_r^T d_i,e, then cast F32 BEFORE
the native residual/final RMS/head. No validation-specific refit/subspace or
rank grid expansion. Coefficients and mean are ORACLE responses, not available
runtime inputs. Rank0 is the common-only control, all functions identical there.
Native full-rank control passes original features directly, so selected heads
are byte-exact; the F64 eigensystem recomposition check is separately numerical,
and is NOT claimed to be byte-exact after subtraction/summation/quantization.

Measure weighted development and validation private energy retained
1-sum||d-BB^T d||²/sum||d||², plus all24 per-anchor fractions.
Residual-energy computation must agree with sum||d||²-sum||B^T d||² within
1e-10 relative to private energy. Development top-r PCA gives the minimum
summed Frobenius residual among common linear output subspaces for THESE rows.
It is not a lower bound on nonlinear compression, a different downstream metric,
an arbitrary shared base, other inputs/banks or optimal KL-aware subspaces.

## Full-vocabulary preservation and information

For EACH of6 validation anchors and ALL128 functions, compute original native
z_i,e=H(h_i+p_i F_e) and projected zhat_i,e=H(h_i+p_i Fhat_e). Compare q_i,e=
softmax(z_i,e) to softmax(zhat_i,e): paired source-native KL, not truth accuracy.
Use F64 self-CE entropy and independent log-softmax difference, same1e-10
identity/agreement/nonnegative checks. Save every per-function KL/argmax mask,
selected-original-function subset and complete native/projected F32 logits.

For each anchor, also measure uniform interventional information:
JS=(1/128) sum_e KL(q_e||qbar), qbar=(1/128)sum_e q_e.
Check independently H(qbar)-mean H(q_e),1e-10 absolute, range[0,log128].
Rank0 must have abs(JS)<=1e-10. This is artificial randomized do(E) predictive
distinguishability. An actually deterministic router E=f(X) has no such observed
conditional entropy; do not call this observed routing I(E;Y|X) or useful capacity.
JS retained alone cannot prove correct identity assignment; paired KL is required.

For each nonzero FIXED rank, prospective GEOMETRY eligibility requires ALL:

- weighted development private energy>=.95;
- weighted validation private energy>=.95;
- mean KL across6*128 paired functions<=.01nats;
- each validation anchor's mean over128 functions<=.05nats.

These are simultaneous descriptive representation gates, not task/latency
acceptance. Report ALL fixed ranks and the smallest passing rank, if any.
No rank is selected by held-out task quality or used to reopen440. Failure of
all fixed bases closes this specific dev-covariance oracle recipe, not all
rank-r/common-private decompositions. A passing oracle cannot certify a learned
common mean or cheap private coefficient evaluation.

## Cost boundary and first stop

Admission<=300s, native collection/eigensolve/all controls/reporting<=300s,
total<=600s from main, import excluded; peak RSS/Windows working set<=4GiB,
experiment outputs<=768MiB, admission disk reserve>=2GiB. Guard every hash chunk,
native row and stage. BEFORE first native replay, explicit admission boundary
<=300s required. FIRST binding/replay/nonfinite/numerical/resource failure
retained in .failure.json with partial inventory before a NEW numbered repair.
Never overwrite/rerun444. Original payload size/mtime must remain unchanged.

Outputs: geometry NPZ (all anchors/features/mean/contrasts/covariance/basis),
native6x128x32128 F32 logits NPY, five same-shape rank NPY, full raw statistics
and SHA/byte inventory. This enumerates functions OFFLINE. It pays the original
full function/head work and reports no runtime storage/active-compute saving.
Availability of barF(x), coefficients and their cost must precede any fit/export.

CPU0/Torch1/interop1/BLAS1, Torch2.6.0+cu124/NumPy2.4.6, no concurrent modeljob,
GPU/T4/network/new corpus/new fit/native benchmark/C edits. Preserve exact approved
publisher daemons; original binaries/engine unchanged. UsefulRAM-n/LUT/realDRAM,
whole original-relative quality/SAMEartifact50/families/~100B remain OPEN.

After freeze, run ONCE:

`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth444_switch_function_contrast_geometry.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth444_switch_function_contrast_result.json`
