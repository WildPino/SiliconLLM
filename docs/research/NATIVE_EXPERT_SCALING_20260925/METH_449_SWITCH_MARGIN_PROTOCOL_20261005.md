# METH-449: prospective saved-state margin and source-head decomposition

## Question and decision

Freeze source/math/protocol BEFORE imports/compile/numerical observation.
447 full-row I4 and448 block64-I4 are both retained qualified local FAIL under
unchanged predictive gates.448 lowers coefficient error and meanKL but changes
9winners versus4. Diagnose the mechanism before another precision/format trial.
No candidate model, fit, new data, blocksize/precision grid or gate relaxation.

Two arms on ALL336 consumed fixed teacher-prefix positions, books18..23:
saved original logits/prefixes,447 correct-ID state/logits and448 correct-ID
state/logits. Use original source128 tied I8 head rows and original scales.
No expert/FFN/full-model or complete-vocabulary head forward. Selected two-row
head integer replay is explicitly part of this diagnostic, not metadata only.
Source/core/input/probability remained matched in the frozen parent experiments.

This assay chooses the next mechanism:
- true state displacement across source decision planes -> next representation
  must reduce that directional error; average Frobenius/KL alone insufficient;
- selected shadow pair preserved but native pair flips -> quantify whether A16
  input quantization or F32 output cast/tie handling matters before changing FFNs;
- original native pair already disagrees/ties in smooth shadow -> document
  original readout sensitivity, no whole-source/semantic conclusion;
- any arithmetic/primal discrepancy -> retain FIRST failure before NEW repair.

Shadow comparisons concern ONLY native original winner versus declared paired
competitor, not global smooth-head argmax. A finding never relabels447/448 PASS.

## Fresh inputs and binding

Parents447 raw SHA3025bb7b383a631162c2eca803784e0c2cdc811c47a6abe1f73619a791e6dd29;
448 raw SHAc5d26a3e8192ae6c648567134290783192250ed7a92730cafb25db44440e516c.
All helper/protocol/output files and retained original/runtime binary hashes
freshly verified. Both parents ALL9apparatusPASS/4of5feasibilityPASS, failed
argmax counts4and9. Their source/prefix/keys/original archive identities must agree.
Own source/math/protocol physically equal committed filtered HEAD.

Original source128 google/switch-base-128 immutable revision
86c815ec05361a33a8b49fc717277da9c0a4e711. Fresh complete7,541,946,880B payload
SHA6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe and original
manifest SHA3a4892ad2bc846937a3ad7898048df64b9a26dc391a3d02c110fe75cd614511a.
Fresh canonical380 export record6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee.
Head descriptor32128x768/encoding1/count/offset/scales checked; original I8
codes no-128 and represented F32 row scales finite positive. Readonly payload.

CPU0, NumPy2.4.6/BLAS1, actual loaded binary hashes recorded. No Torch import,
GPU or concurrent modeljob. Exact approved pythonw daemon with2argv and argv[1]
D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py preserved; no management.

## Original/candidate selected-row qualification

At each position: original winner a=argmax(z0), native candidate winner c.
Source runner-up is original argmax excluding a, ties resolved by lowest ID.
Paired competitor b=c if changed, otherwise source runner-up. Analyze ALL336,
not only changed rows. Record identities and the raw paired native margins.

For original F32 head input h0, reproduce original native A16 scale/codes
byte-exact saved prefix. Same A16 definition for candidate F32 hc: absmax/32767
in F32, zero scale1, F32 division, nearest-even, clamp-32767..32767 to I16.
Two selected head rows a,b: independent I64 dot(q8,activationCodes), then
F64(integerSum)*F64(headRowScale)*F64(activationScale), F32 cast. All saved
original/candidate logits at those rows must be byte-exact.
2arms*336positions*2states*2head rows=2688 selected values replayed.

## Algebra, attribution and tolerances

For native logits z0,zc decoded F64:

    m=z0[a]-z0[b]
    d=(zc[b]-z0[b])-(zc[a]-z0[a])
    candidate_pair_gap=zc[a]-zc[b]=m-d.

Identity error<=1e-12. Actual flips need d>=m, candidate pair gap<=0, including
native equality/ID ordering. Original winner/top2 gap recorded separately.

Decode selected head rows wa,wb=q8*F64(F32 row scale), v=wb-wa. Decode original
and candidate A16 input reconstructions q0,qc in F64, dh=hc-h0:

    d_state=v dot dh
    d_A16=v dot ((qc-q0)-dh)
    d_unrounded=(candidate_unrounded[b]-candidate_unrounded[a])
                -(original_unrounded[b]-original_unrounded[a])
    d_cast=d-d_unrounded
    d_order=d_unrounded-d_state-d_A16.

Require |d_order|<=1e-10 and sum reconstruction error<=1e-10. The cast term is
specifically the independently replayed output F32 conversion; the dot/reduction
order remainder is reported separately, not attributed to quantization.
Smooth original/candidate pair gaps are -v dot h0/-v dot hc; their difference
matches d_state within1e-10. Record A16 unrounded pair gaps too.

Changed-case classification, tolerance1e-10 fixed before observation:
- source smooth pair gap<=tol: source_pair_tie_or_disagrees_shadow;
- otherwise candidate smooth pair gap<-tol: state_crosses_shadow_pair;
- otherwise candidate smooth pair gap>tol: shadow_candidate_pair_preserved;
- otherwise candidate_shadow_near_tie.
For all unchanged rows classification unchanged. Counts do not define new quality
gates; preserve every underlying value and both decomposition directions.

## Information and reference-required certificates

p=softmax(F64(z0)); q=softmax(F64(zc)); delta=zc-z0. KL=dot(p,logp-logq)
must equal parent per-position KL<=1e-10 and independent
log(sum p exp(delta))-E_p(delta)<=1e-10, KL>=-1e-10. Stable exp pivots used.
Record .5Var_p(delta) as LOCAL quadratic diagnostic, not assumed equal to KL.
Record entropy/pmax/TV and best constant-centered logit Linf=(maxdelta-mindelta)/2.
TV<=sqrt(maxKL/2)+1e-10 checked.

Original winner versus runner-up: logit gap g and posterior gap gp. Diagnostic
sufficient certificates, with conservative numerical allowance:
- g>2*centeredLinf+1e-12;
- gp>2*TV+1e-12;
- gp>sqrt(2*(maxKL+1e-10)).
Every certificate must exclude actual flips. They require original reference
logits and BOTH posteriors; no deployment/fallback shortcut follows.

Predeclared gap bins [0,.001),[.001,.01),[.01,.1),[.1,1),[1,infinity).
Count all rows and changed rows per bin, paired classifications, certificate
counts, row/block flip intersection/differences. No outcome-chosen thresholds.

Tiny independent qualification: exact margin crossing and ties, KL/cumulant/
constant shift variance/Pinsker, A16 zero/extrema, selected I64 native head dot
against manual integer sum. ALL eight final apparatus gates required.

## Resources and retention

No full heads/FFN/model, optimizer/new source download/corpus/GPU/C modification.
Expected about15..35s total. Hard admission<=300s, numeric/reporting<=60s,
total<=360s, peakprocess<=512MiB, retained directory<=16MiB, free disk>=1GiB.
Guards during hashes/each position. No extending failed budgets; retain FIRST
failure before NEW numbered repair. All447/448 scientific files/outputs immutable.

Retain per-position native/shadow paired gaps, decomposition terms, KL/TV/variance,
reference certificates/classifications and full IDs/keys in raw JSON. Keep NPZ
original/candidate head inputs/A16/scales and selected source head rows/scales;
reload saved archive byte-exact. Hash every current output, preserve engine/
source/original binaries and unrelated changes.

Outputdirectory results/native_expert_scaling/meth449_switch_margin/ and raw/
failure must not preexist. Command ONCE after freeze commit:

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth449_switch_margin_diagnostic.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth449_switch_margin_result.json

No new quality/rate/LUT/DRAM/useful-n/family100B evidence claimed. Full goal ACTIVE.
