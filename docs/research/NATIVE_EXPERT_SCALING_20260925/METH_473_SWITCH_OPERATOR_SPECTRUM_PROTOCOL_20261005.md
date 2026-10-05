# METH473: learned-WI response spectra and the complete linear-factor byte budget

5 October 2026. Prospective scientific protocol; no 473 response or spectrum has
been observed. The full goal remains active and incomplete. Previous goal turn
was PROGRESS: 472 produced and independently audited a complete stored candidate,
and rejected its fixed rank32 input-PCA recipe. Its failures stay immutable.

## Question and decision

Does the known learned WI admit a development preactivation approximation whose
ideal required ranks fit the complete 70% bank budget? Input PCA in 472 chose
directions using input energy and failed 105 of 107 novel-validation function
gates. This inquiry changes the metric to learned-operator action. It does not
retry 472, alter its gates or construct a new deployed candidate.

For each of the SAME 107 admitted IDs, A=S_I*W_I is the decoded real 3072x768
WI. X has weighted effective-input rows sqrt(w)*alpha*q. Compute ONE full thin
SVD of Y=A*X^T. For rank r, the ideal minimum of ||(A-B)X^T||_F^2 over rank(B)<=r
is the tail singular energy of Y. The optimum B_r=U_r U_r^T A defines an action
on the entire input space using the pretrained A; it is not identified only
by a data pseudoinverse. No U factors or new function bank will be exported.

Primary target: relative preactivation RMS<=0.05, hence squared tail ratio
<=0.0025 and energy retained>=99.75%. The computed F64 spectra require explicit
residual qualification. A fixed +/-1e-6 guard band on squared relative tail
ratio yields PASS, FAIL or INDETERMINATE. It is a numerical decision guard, not
a statistical confidence interval or a formal interval-arithmetic certificate.
Exactly zero response has zero residual and requires rank0; no denominator clamp.

Keep all 107 factors and all 21 original I8 fallbacks. Nominal stored F32 decoded
WI factors require 4*r*(3072+768) bytes; original I8 WO and WO scales remain.
Including fixed fallbacks and the 16,448-byte header/table:

    B = 352202816 + 15360 * sum_e(r_e).
    10*B <= 7*605945856  =>  sum_e(r_e)<=4684.

Uniform ranks are bounded by 43; this does not bound each expert's variable rank.
Keeping all WI row scales additionally changes the sum cap to 4599 and uniform
cap to 42. The authoritative inquiry uses the decoded format and cap 4684.
This is a nominal byte equation, not an admitted C ABI or a guarantee that F32
coefficients achieve the ideal real optimum.

For each ID report nominal minimum rank, and the minimum ranks under tail ratio
minus/plus the fixed guard band. Then:

- ALL107 rank32 PASS: separately freeze ONE rank32 operator-aware full-function
  candidate inquiry with actual F32 coefficients and the original nonlinear path.
- Sum of lower required ranks >4684: close this preactivation-preserving linear
  F32-factor family under this development metric and declared bank budget.
- Sum of upper required ranks <=4684: ideal rank allocation fits the nominal
  budget; review the full rank/work distribution before a separately frozen
  candidate. No validation-selected mask or fallback change is allowed.
- Otherwise: INDETERMINATE; retain the numerical ambiguity explicitly. No rank
  grid or candidate is justified by an ambiguous spectral screen alone.

Uniform43 failed IDs are descriptive. They do not determine the variable-budget
decision. None of these outcomes proves nonlinear FFN preservation or donor
quality; ReLU, second A16 and WO can suppress or amplify preactivation errors.

## Exact reused inputs and development separation

Original Switch128 donor: 7,415,217,408 distinct trained parameters, 128 experts,
D768/F3072/top1. Original 7,541,946,880-byte I8/A16 payload and manifest unchanged.
Freeze exact controller/math/protocol/binding identities BEFORE first scientific
compile/import/numerics. Binding preparation uses only standard-library metadata
and existing bytes; reading .npy shape headers is not a new fit or response.

Prospective bindings cover the actual Python3.12.10, NumPy2.4.6, psutil7.2.2 and
threadpoolctl3.7.0 assets, old frozen helper/record identities, original engine,
native binary/compiler/libomp/manifest, full payload and preserved unrelated work.
ALL 6224 prior469/471 outputs and ALL114 audited472 outputs are refreshed by full
SHA, size and recorded modification time before new algebra: 6338 input files.
Old source/first failures remain byte-exact. 472 raw/RET and parent bindings
are exact qualified prerequisites, not an inherited compression/quality PASS.

Use the qualified 472 query_inputs.npy: all19962 fixed bank11 queries, 13313 dev
queries. Book roles are 0..63 development, 64..127 validation, 128..191 development
augmentation. Only role0 enters any new response geometry. Reconstruct exact
(book, q bytes, alpha bytes) dedup and first occurrence, then equal-book weights
1/(B*m_book). Reconstructed representative indices/weights must equal the saved
472 witnesses byte-for-byte; no cached PCA axis or truncated subspace is used.

Existing metadata fixes 11331 development representatives, min34/max308 per ID,
sum(m_e^2)=1488445. Each spectrum includes all m_e singular values and the full
m_e x m_e right witness. No new sources, validation fit, filtering, held-out
fallback selection, loss/probability weighting, numerical-rank truncation or
fresh tokenizer/model replay. All 107 IDs are processed exactly once.

## Arithmetic qualification before new spectra

The SWI8A001 manifest must match the original export's 13 config integers,
F32 epsilon, flags and all3320 tensor descriptors, reaching exact EOF. Each
consumed original WI and row-scale section has its export SHA checked.

Known tiny native-order and known-spectrum/zero-response/storage controls run
first. The 13 cached I8/A16/I64-dot/F32 fixtures must reproduce byte-exact values,
including width4096 fixtures. Freshly verify all1344 cached native teacher WI
preactivations, including exact A16 input codes/scale and original order
integer dot -> row scale -> alpha -> F32. No ReLU/WO/head/model is replayed here.

For all new representative responses, q is I16 excluding -32768; integer dot
width*128*32767<2^53 (width768 for actual WIs). The native-order F64 dot is exact
before two scale multiplies. A's I8*F32 coefficients and q*F32 alpha are exactly
representable in F64. A*(alpha*q) may reassociate the later dot operations.

Let S be the sum of absolute scalar products, computed by positive integer
dot -> row scale -> alpha. With u64=eps64/2 and gamma_k=k*u64/(1-k*u64), require
elementwise |decoded_response-native64_response| <=4*gamma_(width+2)*S. Bound
qualification is explicitly scoped to normal finite F64 arithmetic. Products
of finite F32 source scales and integer codes remain within normal F64 range;
zero products are exact zeros. Source native-F32 response error energies are
reported separately. Final F32 arithmetic is not promoted to exact real algebra.

Directly form Y from these decoded responses and sqrt(book weights). Thin SVD
full_matrices=False, canonical sign of each right axis (first maxabs coordinate
positive). Reconstruction/eigen/energy relative residual<=1e-10; right
orthogonality Frobenius residual<=1e-10*sqrt(m). Finite descending nonnegative
singular values; full tail curve, no discarded numerical-null modes.

Save/reload every singular vector/value/index/weight/tail byte-exactly. Independent
retention will reconstruct development inputs/weights and Y from original WI,
verify right Gram/eigen/orthogonality/energy witnesses without SVD refit, and
rederive every primary rank/status/byte decision. Native cached qualification
remains source-recorded, with its input/output identities retained.

## Resources, terminal state and immutable first execution

CPU only, parent affinity[0], single actual BLAS worker. No native child, compiler,
GPU, downloads, framework or tokenizer imports/updates/training. No concurrent
Python/scientific job, except exact authorized pythonw publisher daemon argv.
Hard600 seconds /4GiB OS peak /128MiB new outputs; admission<=300 seconds.

Full right witnesses 11907560 bytes; singular values/indices/weights/tail add
318124 bytes; 8192 bytes metadata/header allowance per107 files gives exact
upper13102228 bytes. Reserve5MiB additionally for raw (<=4MiB), progress and
diagnostics. This is below128MiB. Actual bytes/time/OS peak are measured too.

Largest Y is 3072x308x8=7569408 bytes. At most the current expert's decoded/native/
magnitude/difference/bound responses and thin SVD factors are live; original WI
is read directly into compact I8/F32 sections, not mapped as resident full donor.
LAPACK/runtime/JSON/qualifier workspace has more than2GiB headroom under4GiB;
actual OS peak guard is authoritative. Query memmap94.46MB and cached1344 native
control rows70.25MB are bounded. No claim of physical DRAM traffic or model rate.

ONE exclusive namespace and ONE first command, after freeze and recorded resumption:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth473_switch_operator_spectrum.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth473_switch_operator_spectrum_result.json
```

No completed main rerun. Observation timeout means re-poll SAME confirmed live
handle; a fault is retained before any separately numbered repair. Five apparatus
gates must PASS. Raw/OUT/fatal/progress are retained unchanged; actual tool exit,
main PID plus creation time, peak resources, and literal invariant UTC Windows
ApplicationError1000 query qualify terminal state. Independent audit requires
these bytes and actual terminal evidence, not an apparent Python exit alone.

The full goal still requires a viable reusable conditional artifact, whole
all-bank composition/fresh donor-relative prediction/generation/tasks AND>=50
accepted batch1 IDs/s on SAME artifact, causal useful-n, winner AND mass, CPU
LUT and physical DRAM scaling, multiple actual families/scales/~10B/~100B as
resources permit. A spectral screen resolves one transfer prerequisite only.
