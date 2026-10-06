# METH497 — exact finite feature row-rank certificates

Prospective before ANY497 control/field/rank observation. Previous goal turn
PROGRESS:496 full unchanged-artifact error decomposition independently admitted.
Goal ACTIVE/INCOMPLETE; no scientific497 process or output exists at preparation.

## Question and fixed decision

Does the actual495 design H_e=[qphi*alpha,1] have full real row rank on ALL
development UIDs? Do consumed feature rows extend its span?496 found U error
above1% before quantization and every m_e<=308<513; a regularized residual
does not establish lack of finite-sample representation. Resolve this exact
question before changing width, precision or learner. No fitted alternative,
target fitting, validation selection, coefficient export, optimizer, source
FFN/native/model invocation or new resource. No prime/pivot/ID sweep.

Use unchanged qualified495 features.bin and493 UID/occurrence wires through
minimal496 qualification metadata. Source495/496 admissions and exact digests
remain dependencies; bank, coefficients, target arrays, compiler, original
source payloads and completed numerical controls are not current inputs.

## Exact representation and one field

Prime p=2147483647. Verify primality by trial division through isqrt(p) in the
new control; do not assume a field. For positive finite alpha F32 bits b:

```
if exponent==0: alpha=mantissa*2^-149
else: alpha=(2^23+fraction)*2^(exponent-150)
alpha_mod=(mantissa * 2^power) mod p,
negative power via the inverse of2 modulo p.
H_mod[j,:512]=(qphi[j,:]*alpha_mod[j]) mod p; H_mod[j,512]=1.
```

All denominators are powers of2 invertible in this odd prime field. A nonzero
minor modulo p proves that the same minor of the exact dyadic rational H is
nonzero, hence the same real row independence. No floating rank/cutoff or
interval assertion. qphi*F32alpha is exactly representable in F64 here, but
the certificate uses integer mantissa/exponent values, not a floating product.

Main: columns0..512, first remaining nonzero row, forward elimination, normalize
pivot row, eliminate rows below. Track original row indices, chosen columns,
raw pivot values and their product modulo p. The recorded minor uses original
rows IN PIVOT ORDER and columns in increasing order, so its determinant equals
the product without a row-permutation sign. Retain SHA256 of the original full
field matrix serialized I64 little endian, all input UID row orders, and minor
certificate. Audit independently maps alpha using Fraction.from_float exact
numerator/denominator, reconstructs ALL H matrices, verifies hashes, chosen
minor determinant with last nonzero row pivoting/sign, and verifies dimensions.
For a non-full main case, audit also independently recomputes field rank with
reversed columns and last-row pivots, so no unverified field upper bound leaks
into reports. Audit never imports main or its mathematical helper.

I64 safety: field values0..p-1; product<=(p-1)^2<2^62, subtraction stays within
signed I64. q times alpha_mod has absolute bound32768*(p-1)<2^46. Reduce modulo
p after each elimination. Scalar determinant/inverse uses Python arbitrary
integers. No BLAS multiplication, rounded determinant or probabilistic field.

## Complete cases and interpretation

128 source IDs and two cases each: development rows (ascending local UID),
ALL rows (development then consumed-validation, ascending within each split).
ALL17540UID/19962occurrences/11721development/5819consumed validation; actual
source acceptance1. Original wires and domain joins exact. Frozen eligible
counts development<=308, ALL<=498<513. Two empty cases for original ID0 are
explicit EMPTY_NO_EXPOSURE, with trivial rank0/null dimension513; no usefulness.

For r_mod=m, exact real rank=m/nullity=513-m certified; every finite target
admits an unconstrained real readout. This does not certify small coefficients,
I8 encodability, generalization, source information preservation or quality.
For r_mod<m, real rank lies in[r_mod,m]; rank deficiency over this prime alone
is INCONCLUSIVE over the rationals, never a class impossibility proof.

Consumed dimension gain is exact ALL_rows-development_rows only when both
real ranks are certified. If development full rank is certified but ALL not,
gain has lower bound max(0,r_mod_ALL-m_dev) and upper bound m_val. If development
rank is inconclusive, report gain lower0/upper m_val: do not subtract two modular
lower bounds. No consumed target is read or used. Per-UID consumed-novelty coverage
requires full development AND full ALL row-rank certificates; partial dimension
bounds do not identify particular independent consumed rows.

Decision fixed:
- All127 nonempty development and ALL cases full: FINITE_DEVELOPMENT_INTERPOLATION_AND_ALL_CONSUMED_FEATURE_NOVELTY_CERTIFIED.
- All nonempty development full, remaining ALL partial: DEVELOPMENT_INTERPOLATION_CERTIFIED_CONSUMED_NOVELTY_PARTIAL_OR_INCONCLUSIVE.
- Otherwise: FINITE_FEATURE_ROW_RANK_PARTLY_CERTIFIED_REMAINING_INCONCLUSIVE.

All256 cases retained, all1040 original UID/cell/rare/view/role-mode count groups
and384 source-role/expert exposure groups recounted. New coverage metrics are
certificate coverage, original split counts and unique UIDs, not source energy
or candidate ID fidelity. Compare original counts/source exposures to496 raw
metadata. No independent-ID permutation, exposure filter or new quality gate.

## New controls, limits and retention

One new three-row dyadic design with first three columns dependent but bias
restoring full rank; its pivot minor exact determinant -1/2. A duplicate-row
case checks field rank1. New normal/subnormal/max-finite F32 exponent fixtures
map independently through Fractions. Verify p prime and2 inverse; I64 bound
fixture. No completed495/496 control replay. Audit uses exact rational small
determinants/rank, not main control functions.

Metadata builder90s/256MiB; main600s/512MiB parent OS peak, audit same. CPU0,
NumPy integer arithmetic, BLAS1 configured although no BLAS calculation. No
scientific/native child. Metadata git duration charged; no separate metadata
git child OS-peak claim. Each output directory64MiB/all497 outputs128MiB/raw8MiB.
Reports/certificates JSON bounded by fixed256cases/513columns/domain counts.
Filesystem check once/sec plus forced terminal; deadline/memory watcher through
full record write. Builder includes post-write guards and actual PID/create-time.
All helper/protocol commit before builder, binding commit before sole main;
raw/receipts commit before independent audit. Actual tool/chunk/session/exit,
PID/create-time, Windows Event1000 and complete retained partial/fault records.
First faults precede numbered repairs; completed namespaces never rerun.
