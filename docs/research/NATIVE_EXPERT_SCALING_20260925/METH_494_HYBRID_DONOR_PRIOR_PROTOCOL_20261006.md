# METH494: donor-informed full-rank linear term and shared even dictionary

Prospective6 October2026, before any494 source tensor value/kernel/dictionary/
prior/control observation.493 supervision is fully admitted and immutable.
Goal ACTIVE/INCOMPLETE. This stage compiles a concrete weight-informed prior
and exact logical cost contract; subsequent supervised learning is specified
below but not executed by494. No native/model/donor FFN replay/GPU/new resources.

## Question and changed hypothesis

The proposed B_e ReLU(Ax) r512 has a local-rank ceiling and many unidentified
readout directions. Use the exact REAL ReLU identity (z+abs(z))/2 to preserve
the donor's entire linear component instead:

```
W=dequantized original I8 WI; V=dequantized original I8 WO
F_e(x)=.5 V W x + .5 V abs(Wx)
G_e(x)=L_e x+B_e abs(Ax)+c_e.
```

L_e is private D768x768, full possible rank. A is shared r512xD768 ternary,
B_e private D768xr512, c_e private D768. The decomposition is exact for the
ideal unquantized ReLU operator with these quantized weight values, NOT for
the original F32/A16 program or weighted pF. Physical precision and learned
weighted contribution must be independently validated. Private parameters
remain derived from distinct original experts, no duplication/unobserved-zero
initialization. No global r512 Jacobian ceiling remains with private L.

This is a different transfer geometry from472 input-PCA/474 output floors,
and from the provisional pure shared-feature template493. It adds a source
weight prior and preserves all input directions in L, paying their active cost.
Required donor: non-gated bias-free two-matrix ReLU FFN; Switch source qualifies.
Gated/other activations require another derivation, no universal family claim.

## Actual dependencies and bank11 extents

Fresh493-R2 RAW/RET/admission/binding/UID wire and original472 query array,
current380export RAW and manifest, all scientific/runtime/preserved files.
Each of256 original bank11 WI/WO code matrices and256 F32 row-scale extents,
plus original128x768F32 router, receives an independent fresh extent SHA
against380 per-tensor descriptors. Check unchanged payload stat around reads,
manifest offsets/shapes/encoding against all3320 export entries. Fresh current
bytes of unused payload portions are not claimed. Entire payload historical
SHA stays provenance only; the actual bank extents are exactly source-qualified.
No old main/capture/controls/audit replay. Qualified saved p/input/codes retain
their original UID/firstqid/expert/role/hash/alpha identities.

## Dictionary selection and development geometry

ALL128 experts; select four hidden rows per expert. For each source neuron j,
compute F64 individual Gaussian contribution score
(.5*||V[:,j]||^2*||W[j,:]||^2). Sort descending computed F64 score, ties smallest
hidden index. No observed response/validation chooses rows/width. For selected
WI direction (positive row scale cancels), keep256 largest abs I8 coordinates,
ties smallest coordinate; sign ternary t. Zero source row/scale -> zero t,
scale1. Otherwise sigma=roundF32(sum_kept abs(code)/(nonzero_kept*||code||2)).
Packed group4 U8 code=sum_(k=0..3)(t_k+1)*3^k in0..80; all-zero group is40.
A is exactly the decoded ternary matrix times stored F32 row scale.

Use ONLY11721 canonical development inputs (owner roleflags&5), once each,
not A16 similarity or occurrence-weighting. S=X^T X/11721, streamed blocks256
in originalUID order, then S=(S+S^T)/2. C=S+1e-3*(trace(S)/768)*I. Zero-mean Gaussian N(0,C) is
an explicit approximation of source geometry, not the source distribution.
Per-ID p calibration a_e=roundF32(mean development selected source p); with
zero development rows use a_global=roundF32(all-development mean), explicitly
marked unsupported amplitude initialization. ALL5819 consumedval remain
unused for covariance, selection, calibration, kernel/prior construction.

## Gaussian prior and normalization

For U=a^T x and V=b^T x under x~N(0,C), let u=||a||_C,v=||b||_C, c=a^T Cb/(uv):

```
E abs(U)abs(V)=uv*(2/pi)*(sqrt(1-c^2)+c*asin(c))
E abs(U)=u*sqrt(2/pi); E 1*1=1.
```

The ReLU covariance is HALF the arc-cosine kernel normalization in Cho–Saul
eqs1/3/6. Absolute covariance follows from ReLU(z)+ReLU(-z). Primary source:
[Cho–Saul2009](https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf),
page2/eqs1–6; distribution/weight-prior construction here is our derivation.
No empirical quality/speed claim is imported from that paper.

H=[abs(Ax);1], K=E[HH^T], lambda0=1e-3*trace(K)/513, Kreg=K+lambda0 I.
Kcross_e=E[abs(W_e x) H^T]. Unweighted Gaussian prior:

```
L0_e=.5 V_e W_e
C0_e=.5 V_e Kcross_e Kreg^-1 ; C0=[B0,c0].
```

Main uses F64 matrix products/Cholesky solve, F32 serialized L0/C0 as chosen
approximate coefficients. This is a regularized Gaussian operator projection,
not development-target fitting; original quantized forward is not replayed.
Save K/Kreg, eigendecomposition and geometry. Any nonfinite/invalid source,
extent/covariance/positive-definiteness/solve/budget failure stops; no jitter,
width/selection/distribution/scalar sweep. Require numerical Kreg eigmin>=.99
lambda0. Independent audit reconstructs C with different block sizes, uses
acos angle formula for absolute kernel, source integer/scales factored
separately for L0, and certifies prior normal-equation residuals/objective
bounds. It does not import main/math and never calls old scientific helpers.

Numerics are fixed BEFORE invocation: row whitening by Cholesky(C); main
asin formula, audit acos-angle formula. Assert absolute computed correlations
<=1+2e-12 before clipping to[-1,1]. Symmetrize K before ridge. Audit covariance,
Gram and Cholesky use3e-12 relative/absolute tolerances; eigen orthogonality
Frobenius<=1e-10. Computed eigen lower=eigmin*(1-orthogonality_error)-
eigen_reconstruction_Frobenius-1e-10*max(1,||Kreg||F), required>=.98*lambda0.
For ALL serialized C32 coordinates audit R=T-C32*Kreg. Allowed envelope is
E32*abs(Kreg)+1e-7*max(1,abs(T)+abs(C32)*abs(Kreg)),
E32=abs(C32)*2^-24/(1-2^-24)+2^-150 (entrywise; products are matrix products).
Independent L0 factor ordering must agree within4*2^-24*max(1,abs(L0)).
Coefficient/audit calculations are numerical, not formal outward-rounded real
interval proofs. ||R||F^2/eigen_lower is the algebraic objective-gap expression
for the chosen computed Gram, reported with that limitation. Selection dominance
also checks all four original computed-score IDs and exact index tie ordering.

## Physical prior wire and future inference

Multiply serialized unweighted L0/C0 by stored a_e in F32. Quantize L/B rows
separately I8[-127,127], scale=roundF32(rowmaxabs/127), zero rows scale1.
Codes=clip(RNE_integer(roundF32(value/scale)),-127,127). Bias remains F32.
Auditor independently checks calibration via exact F32 rational means, row
max/scales/quotient/half-even codes and every packed code/source coefficient.

bank_prior.bin header <8s8I: M494BNK1,D768,F3072,r512,n128,bank11,group4,
keyrows8,keycols16. Then packedA98304B, A_scales2048B, centroidkeys122976B,
then128 expert rows: L_I8 589824B/L_scales3072B/B_I8 393216B/B_scales3072B/
bias3072B. Perexpert992256B; total127232136B. Keys are INITIAL source-router
centroids, not trained: raw-x coefficients mean row/column groups minus half
grandmean, abs-feature coefficients/bias0. OriginalID=16*i+j, no permutation
or searched label rearrangement.

L0_prior.bin M494LIN1/2359296/768/128 +128*768*768F32=301989912B;
C0_prior.bin M494ABS1/1575936/513/128 +128*768*513F32=201719832B.
geometry.bin header <8sIII: M494GEO1,D768,r512,H513; then S/C/Cholesky and
K/Kreg/eigenvectors/eigenvalues in F64 C-order, total20475956B. Small
selection/calibration/kernel controls/cost JSON retained.
This is one-bank PRIOR artifact, not a trained/end-to-end model.

Physical future evaluator: source input F32 -> A16 qx/alpha_x; group4 I32
table -> exact ternary dot -> roundF32((dot*sigma)*alpha_x), abs; then A16
qphi/alpha_phi. L dot uses I64 (768*32767*127>2^31); B dot I32-safe at512.
Each block roundF32((integerdot*row_scale)*activation_alpha), add blocks in
F32 then F32 bias. G already estimates weighted source pF; no second source
p multiplication. Original reject flag skips residual branch entirely.
Keys use F32 x/absphi, F64 ordered dot then F32 logits. Separate row/column
softmax has exact ideal Cartesian mass/choice at O((I+J)(D+r)); chosen physical
probability is F32 product of separately rounded probabilities, diagnostic
control mass. G and decision must be evaluated jointly; a direct weighted
function is not a claim of uniform source-p scalar reconstruction.

## Complete cost contract, no borrowed timing

Across12 equivalent banks: private slope11907072*n B; shared A1204224B;
n128 centroidkeys1475712B. Read-only inference weight bytes per12bank token:
14587008B, plus logical LUT reads4718592B/table writes746496B, quantizer/input/
feature/output/state traffic. No cache/DRAM locality assumed. Core/head/attention/
state/workspace must be included in eventual SAME whole rate. Perbank intMAC
768*(768+512)=983040, LUTlookups512*192=98304, keyFP64MAC24*1280=30720,
24exp/2normalizers. Original2*768*3072 MAC is a count, not measured speed.
Stored private n increases capacity and conversion cost, not automatically
usefulness; n1280 keys must change and source-weighted priors must be
recalibrated. Candidate routing structure can preserve its own mass cheaply;
actual donor quality/fresh utility/DRAM/SAME>=50 remain separate gates.

## Subsequent ONE learning recipe, frozen now but not invoked494

Fixed A; fixed L from quantized a_e L0. Fit B/c on development only against
493 Y minus PHYSICAL L output. H=[dequantized physical A16 qphi;1]. For each
nonempty expert minimize mean squared residual +.01*(C-a_e C0)Kreg(C-a_e C0)^T.
One convex solve C=(R H^T/N+.01*Cprior Kreg)(H H^T/N+.01*Kreg)^-1.
Empty-development expert retains full source prior/calibration fallback,
never disappears. No head/rank/update/ID sweep or consumedval selection.

Fit two key heads jointly on hard donor winner axes using x/absphi, normalized
by development RMS per feature (zero RMS->1), affinebias. Initialize above
source centroids. Full-development128 Adam updates, LR.01/beta.9,.999/eps1e-8,
L2 to normalized centroid prior1e-4. No checkpoint choice; final update128.
Restore coefficients to physical feature scale, serialize F32. No duplicated
expert functions; report all original128 useful identities/exposures.

Then compile C physical evaluator/export/auditor BEFORE first learned
invocation. Frozen local necessary gates: coupled weighted RMS<=1% in each
role/mode and ID fidelity>=99.9% in each role/mode, all rare/source views,
finite values/budgets/integer bounds. Failed gate closes THIS fixed recipe;
retain oracle-ID versus coupled errors to distinguish representation/control.
No local pass concludes quality: composed476 ALL6649 contexts/head logits,
then allbanks/ownstates/fresh prediction/generation/task/SAME>=50/useful n/
CPU LUT/physicalDRAM/otheractualfamilies/scales are required. No495 fitting
until full learning/evaluator/native/audit/resource implementation is frozen.

## Actual494 limits, controls and sole execution

Freeze helpers/runtime/input/extent/schema/protocol before sole builder90s/
256MiB. Commit binding before main600s/512MiB and audit900s/512MiB, CPU0/BLAS1.
All hashes/extentreads/waits/matrix/kernel/solve/report/output charged. New
combined1GiB includes all raw priors/bank/geometry/partials/audit/metadata,
each directory768MiB/raw terminal8MiB. Stream inputs/development covariance
and source experts; no all-expert dense F64 matrices. Parent OSpeak/watchdog
active through full record write; late terminal receipt and actual exit.

New fixed controls: dyadic real ReLU odd/even identity; abs Gaussian diagonal/
opposite/orthogonal/zero/scaled/constant moments; ternary group40 zero and0/80
sign extremes; I8 quant zero/sign/half-even. Independent analytic corner
proofs and exact rational controls before source main; no completed older
control replay. First fault/partials before numbered repair, no namespace
rerun. Actual sole executions/PID/create-time/Windows Event1000/final admission.
494 PASS grants only weight-informed Gaussian prior + physical byte/cost
eligibility. No supervised/ownstate/modelquality/rate/general-family promotion.
