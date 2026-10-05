# METH474: private stored-F32 output spans and complete source-function error floors

5 October2026. Prospective protocol, no474 fit or numerical output observation.
Full goal active/incomplete. Prior goal turn PROGRESS:473 spectra and independent
audit reject the70%/5% development preactivation rank budget. Original472/473
sources, outcomes and failures remain immutable. This is a new function metric
and private output representation, not a repetition of global bases444/445.

## Decision-changing question

For ALL107 fixed eligible experts, can one development-learned private rank32
output space preserve complete source FFN vectors within the original function
gates? An optimistic floor can rule out THIS fitted space before constructing
coordinates, WO coefficients or a candidate. It cannot rule out all spaces,
nonlinear formats, other ranks or budgets. No candidate rank grid is allowed.

Source f is the stored complete native-qualified F32 FFN output from472,
including original WI, ReLU, hidden A16, I8 WO/scale and finalF32. Fit only on
development functions with unchanged effective-input/book dedup/equal-book
weights. One full thin SVD of Y=(sqrt(w)*f)^T per107 experts supplies output
left axes. Fix rank32 for every factor; keep all21 original fallbacks unchanged.

## Stored space, null modes and primary error envelopes

Canonicalize EACH left axis using its first maxabs output coordinate positive;
flip corresponding right axis consistently. Keep the returned top32 axes even
if some are numerical-null modes. Zero/degenerate responses use that same
library-returned policy. No validation-selected axis, source mask or fallback.
Record numerical rank/null-mode count and all residuals. Null axes are not
identified by source response data; conclusions concern this specific basis.

Store P32=F32(U64_top32). The actual space is range(P32), not a silently retained
F64 U. Require ||P32^T P32-I||_F<=1e-6 and ||P32||_F^2<33. These conditions,
plus the F64 Gram forward-rounding bound, imply sigma_min(P32)>.99999. Thin F64
QR(P32) gives Q/R; orthogonality and P32-QR absolute residual<=1e-12. A numerical
distance guard1e-8*||f|| covers the projected-distance estimate for decisions.
This guard is explicit empirical numerical qualification, not a formal
interval-arithmetic certificate or statistical confidence interval.

For exact distance d=dist(f,range(P32)), any coordinate function g has
||f-P32*g||>=d. Coordinate generation may be arbitrarily powerful; no concrete
coordinate function is constructed here. Final native factor arithmetic can
leave the exact real span, so its allowed displacement is accounted for.

The scoped future format uses finite F64 arithmetic for the final32-term
P32/coordinate dot, with an optional common scale, followed by RNE F32 cast.
No componentwise nonlinear output stage, full-width residual or separate WO
row-scale rotation is included. Those would be different representations.

Define u64=2^-53, u32=2^-24, gamma64=64*u64/(1-64*u64),
eta=gamma64*sqrt(33)/.99999, lambda=eta+u32*(1+eta),
beta=sqrt(768)*2^-126. The beta permits either gradual underflow or flushing
at the FINAL cast; the finite F64 factor arithmetic remains within the scoped
forward envelope. Nonfinite intermediate/output candidates cannot qualify.
For t in the exact stored span and its rounded output y:

    ||y-t|| <= lambda*||t|| + beta,
    ||f-y|| >= max(0,(1-lambda)*d - lambda*||f|| - beta).

The latter follows by triangle inequalities and eliminating ||t||. The best
pointwise projection t has norm<=||f|| and constructive error upper envelope
d+lambda*||f||+beta. Actual numerical d is replaced by max(d_num-guard,0) for
the lower bound and d_num+guard for the upper. Source norms must be finite and
<maxF32/8, avoiding constructive final overflow. For exactly zero source
vectors coordinate0 constructs exact0, so all envelopes are0 without clamps.

Six fixed F32 tie-even/normal/subnormal/signed-zero cases and known-spectrum,
distance/envelope/null/zero/storage controls precede source fits. All SVD
reconstruction/eigen/energy residuals<=1e-10 and left/right orthogonality
<=1e-10*sqrt(m). Source data and actual runtime identities qualify assumptions.

## Frozen function gates and decisions

Same gates as472, applied to the optimistic private-output error range:

- EACH107 novel-validation expert: equal-book deduplicated RMS<=.05.
- ALL3065 natural-validation queries: pooled RMS<=.05, including87 original
  fallback queries; no health, probability or observed-error selection.
- EACH64 original natural-validation book: pooled RMS<=.10.
- Nominal complete rank32 format<=70% original bank. This is a local screen,
  not a new requirement on the full goal. No actual bank is exported here.

For each function gate: lowerRMS>threshold =>FAIL; upperRMS<=threshold =>PASS;
otherwise INDETERMINATE. A FAIL anywhere closes THIS learned stored private
rank32 output space under the scoped arithmetic/gates. ALLPASS plus independent
retention permits ONE separately frozen actual decoded-WO coefficient/full
native-function candidate inquiry. Any ambiguity is retained without rank grid.
PASS means the range floor does not rule out that space; it is not converted
model quality, an inexpensive coordinate map or a deployable candidate.

The nominal original-WI/decoded-WO format retains WI/scales and fixed fallbacks:

    B = 353188928 + 15360*sum_e(r_e),
    uniform rank32 =>405781568B <=70% of605945856B.

No separate WO scales are counted; they would be decoded into coefficients.
Future coefficients could use (P^T P)^-1 P^T*(S_O W_O), then actual F32 storage.
Those coefficients/arithmetic/layout are not computed or admitted here.

## Exact input binding and held-out separation

Original Switch128/7.415B pretrained I8/A16 donor, D768/F3072/top1, bank11. Full
7,541,946,880B payload/manifest and3320 descriptors unchanged. All6447 retained
469/471/472/473 input files7,604,388,190B freshly SHA/size/mtime checked before
new algebra. Exact original engine/native binaries/compiler/libomp, frozen
helpers/records/first failures and Python3.12.10/NumPy2.4.6/psutil7.2.2/
threadpoolctl3.7.0 assets bound. Binding prep reads standard-library bytes and
.npy shape headers only; no scientific compile/import/projection/SVD yet.

Reuse qualified19962 query/provenance rows and19962x768 source F32functions.
Original roles:0..63 development,64..127 validation,128..191 dev augmentation.
Only development enters fits. Exact(book,q bytes,alpha bytes) first-occurrence
dedup and equal-book weights1/(B*m_book), total11331 representatives/max308.
Novelty requires validation code SHA absent from ALLdevelopment codes for the
same expert, regardless of alpha. Reconstructed dev and novel-val indices/
weights equal old472 witnesses byte-exactly. All107 fixed IDs/21fallbacks remain.
Never shorten the32-byte SHA through np.bytes_ scalar conversion.

No new books, source replay, native/compiler child, framework, tokenizer, model,
GPU, downloads, adaptation or updates. Original complete source vectors and
native arithmetic qualification remain byte-bound reused evidence; this inquiry
does not claim a new native forward. No probability-weighted replacement metric.

## Resource and artifact contract

Hard180s CPU /2GiB OS peak /96MiB new outputs; admission<=120s. CPU0/BLAS1,
no other scientific Python jobs; preserve only the exact authorized publisher
pythonw daemon argv. Current-expert SVD/projection only; per-query batches64.

Witnesses retain full singular/right axes, U64_top32, actual P32, Q64/R64,
dev/novel-val indices/weights. Save/reload all bytes exactly. New19962x4F64
metrics retain reference/ideal-distance/lower/upper energies; no duplicate full
FFN vectors. Binding contains exact shape-derived witness/output upper bounds:
full right axes11907560B; fixed basis/QR payload53469184B; representative data
and8192Bper107file headers;638784Bmetrics plus4096Barray header reserve and6MiB
raw/progress/diagnostic reserve. Total must remain below96MiB before first fit.

ONE exclusive namespace/first command after source/math/protocol/binding freeze:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth474_switch_private_output_floor.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth474_switch_private_output_floor_result.json
```

All SIXapparatus gates mustPASS. First failures remain immutable before separately
numbered repairs. Never repeat a completed scientific main. Observation timeout
requires re-polling SAME confirmed live handle, not a restart. Actual tool exit/
PID+creation instance, fatal log, literal UTC Windows ApplicationError1000 query,
raw/progress/OUT bytes and resource measurements qualify terminal state.

Independent retention reconstructs dev/novelty/book weights, weighted source
functions and full right-Gram/eigen/energy/top-axis witnesses, storedP32/QR
relation, all19962distance/envelopes and every primary gate without SVD refit or
main/native/model replay. Actual source/runtime/input and helper identities,
Windows metadata and output inventory remain part of that retained evidence.

Full goal unchanged: real reusable conditional engine.c artifact, whole all-bank
composition/new states, fresh donor-relative prediction/generation/tasks AND>=50
accepted batch1 IDs/s on SAME artifact, causal useful-n, winner AND normalization
mass, CPU LUT/actual DRAM, multiple actual families/scales/~10B/~100B as resources
allow. A private function-space floor addresses one transfer prerequisite.
