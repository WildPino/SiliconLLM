# METH491: independently admitted necessary affine mass bounds

6 October 2026. Goal ACTIVE/INCOMPLETE. The sole main, independent verifier
and final metadata admission are complete. Four roots give necessary ideal
coefficient-norm bounds; eight fixed witnesses remain inconclusive. There is
no unrestricted affine impossibility certificate and no new model artifact.

## Question, configuration and retained inputs

The fixed 490 recipe failed its mass gate after 32 Adam steps. Its finite
trajectory did not decide the affine class. 491 asks a different question:
what coefficient norm must ANY ideal affine root mass have to satisfy the
existing uniform selected-root allowance on the development states?

Source: original Switch n128, 7,415,217,408 distinct parameters, 12 banks,
D768. Qualified 479 raw F32 inputs, ownership and occurrence links; source
root masses from admitted 490-R1 diagnostics. All 159,414 development,
79,458 consumed-validation, 238,872 unique-input and 387,036 occurrence
denominators remain. Validation is coverage data, not witness-construction
feedback or fresh held-out evidence. No model, native call or optimizer.

The [eligibility](METH_491_ROOT_MASS_INTERVAL_ELIGIBILITY_20261006.md),
[frozen protocol](METH_491_ROOT_MASS_INTERVAL_PROTOCOL_20261006.md) and
[operator record](METH_491_INTERVAL_OPERATOR_SOURCES_20261006.md) precede
observations. One deterministic 770-state development subset per root;
one 770x769 full SVD; final left basis vector quantized once to integer
weights on a 2^-40 grid. Both signs are retained; the sign with larger
outward C_lower is selected by the frozen rule. No subset/basis/grid sweep.

## Algebra establishing the result

The physically stored budget is eta=0.001421475836166869, inherited from
log(1.01)/7. For each selected source-side mass q and side sign t,
the allowed real raw logit z=a^T x+b obeys:

```
q exp(-eta) <= sigmoid(t*z) <= min(1,q exp(eta)).
```

Monotonicity supplies the oriented raw-logit interval [L_i,U_i]. For saved
integer weights V and phi_i=[x_i,1]:

```
C(V) = sum_{V_i>=0} V_i L_i + sum_{V_i<0} V_i U_i
r(V) = sum_i V_i phi_i
C_lower <= theta^T r(V) <= ||theta||1 ||r(V)||infinity.
```

Main endpoints and C use outward 100-digit Decimal intervals. Every original
finite F32 value is an exact integer divided by 2^149; all 769 signed
residual coordinates are saved as exact integer numerators. The common
quantization factor cancels. A positive C_lower with NONZERO residual yields
the necessary norm bound C_lower/||r||infinity. Only an EXACT zero residual
with positive C_lower would prove unrestricted ideal affine infeasibility.

A subset witness extends by zero weights to the complete constraint set.
Thus its lower bound constrains every head satisfying ALL development
targets. It supplies no primal feasibility proof. All twelve subsets have
770 states, with no missing two-sided-eligible exposed source ID. The 1,005
development targets not two-sided remain in the full source counts, with
zero witness weights; no upper endpoint was floored or silently discarded.
Their counts by bank 0..11 are 524,6,3,5,21,21,0,0,105,187,42,91.

## Outcomes and bias-elimination corollary

The table prints conservative integer floors of the certified lower bounds.
Full directed decimal endpoints and exact fractions remain in the records.
The retained 490 slope norms are descriptive comparisons, not optima.

| Bank | Necessary total L1 norm >= | Necessary slope L1 norm >= | Retained 490 raw slope L1 norm, approximately |
| --- | ---: | ---: | ---: |
| 0 | 42,501,342 | 98,901,154 | 23.404853 |
| 1 | 25,931,411 | 63,828,375 | 23.618437 |
| 3 | 15,610,756 | 19,233,036 | 12.182316 |
| 4 | 47,607,867 | 47,917,737 | 10.093047 |
| 2,5,6,7,8,9,10,11 | Fixed witness inconclusive | No positive bound derived | Retained individually in JSON |

ALL twelve residuals are nonzero. Four chosen C intervals are positive;
the other eight have nonpositive C_lower. These eight outcomes neither
establish feasibility nor exclude their affine classes.

The slope-only column is a **post-hoc algebraic corollary of the SAME admitted
witnesses**, not another independent repetition. Its
[plan](METH_491_ADMITTED_NORM_ALGEBRA_PLAN_20261006.md) and
[exact output](meth491_admitted_norm_algebra.json) eliminate b using the
first already-selected UID and its allowed z0 in [L0,U0]:

```
b = z0-a^T x0
theta^T r = a^T(rx-rb*x0)+rb*z0
||a||1 >= (C_lower-max_{z0 in [L0,U0]} rb*z0)/||rx-rb*x0||infinity.
```

Both the numerator and denominator are positive on these four banks.
Fraction arithmetic verifies the transformed residual by two equivalent
expressions; the same 100-digit endpoint encloses an alternate 200-digit
formula. No weights, subsets or SVD change. Necessary/current slope-norm
ratios are approximately 4.226M, 2.702M, 1.579M and 4.748M respectively.

These bounds concern ideal raw affine functions on fixed development states.
They depend on the specified raw input coordinates. A huge necessary norm
alone does not prove F32 execution impossible, a minimum rounding error,
unseen-state instability, loss of task quality, or failure of every hierarchy.
An upper bound on numerical rounding is not a lower bound on actual error.
The 490 drift applies to its observed heads, not hypothetical large heads.

## Actual execution, independent checks and costs

| Stage | Actual execution record | Outcome and charged resources |
| --- | --- | --- |
| Source freeze | b80b7b7 | Helpers, protocol and operator record frozen |
| Sole binding builder | 41e1a7/session29575/b822ee | exit0; 20.531s |
| Binding freeze | af71fae | Fresh source/runtime/input binding committed |
| Sole main | a50982/session76971/069042 | exit0; 38.938s; parent 180,375,552B; native 0 |
| Sole independent verifier | e723c8/session64074/3caea5 | exit0; 45.531s; parent 154,775,552B; native 0 |
| Main/audit Windows queries | 1930ad/c18f3b | exit0; both available, zero application events |
| Sole finalizer | 9e146c | exit0; ALL5 metadata admission gates |
| Corollary source freeze | 68df056 | Admission and bias-elimination helper/plan committed |
| Sole post-hoc corollary | 92b63f | exit0; no session; .234s; peak 56,684,544B |

Main PID11116/create1791264721.5174208; verifier PID29520/create1791264792.62497.
ALL5 main and ALL8 independent verifier gates PASS. The independent code
imports no main/math helper, fits nothing and replays no SVD or native command.
It reconstructs all 12 subsets, source/role/ID joins and the saved inputs;
checks ALL9,228 signed residual coordinates exactly through an alternate
float.as_integer_ratio decoder; recomputes mass/C intervals at 200 digits
through log(q) +/- eta - log(1-q exp(+/-eta)); and verifies norm division with
exact fractions. Stored SVD singular values remain constructor diagnostics,
not an independently established rank/decomposition certificate.

Main cap180s/512MiB, verifier300s/256MiB; new output cap64MiB, main output
29,012,896B. Main/audit hash 9,875,650,956/9,876,345,330B. Main canonical
input scans total 1,777,207,728B, separate from hashed bytes. CPU0/BLAS1;
no foreign-work waits. Hashing, I/O and guards belong to these totals.
These are inquiry costs, not model throughput or physical DRAM measurements.

The first metadata patch context fault remains in
[its receipt](meth491_first_metadata_patch_fault.json); it changed no files
and was corrected before numerical work. No numerical main/audit/finalizer
restart, new resources or replay of an old scientific namespace occurred.

## Reproducible records and decision

- [RAW](meth491_mass_interval_result.json), SHA256 `0bf90890796dfe7df62cd6adac4dbc8814b2538fd50c164a745a2bfb4e05a28f`.
- [RETENTION](RETENTION_491_20261006.json), SHA256 `a8377ca820f81a89984929a4756721870ad2050f81e8becb29246477b4871d3c`.
- [ADMISSION](ADMISSION_491_20261006.json), SHA256 `4810c933c4a2a6de94653fb75935ac625c66e66e24a5ffbdb330e8dbe7158338`.
- [Binding](meth491_binding.json), SHA256 `a2e567a26d1583fb2295fbc5c16d5f0a9e7c6fecc8f645b0efbbc310576bdf80`.
- [Corollary](meth491_admitted_norm_algebra.json), SHA256 `743baeed9e4d900adbded4833f83c9704bf8874362c9d62e5d4ed250651fa15c`.
- [Completion registration](meth491_completion_registration.json) links actual builder/finalizer/corollary handles and preserved work.

Executable apparatus: `benchmarks/native_expert_scaling/meth491_mass_interval.py`,
`meth491_mass_interval_math.py`, `meth491_operations.py`,
`meth491_prepare_binding.py`, `meth491_retention_audit.py`,
`meth491_finalize_admission.py`, the two Windows helpers and
`meth491_admitted_norm_algebra.py`. Exact arguments and sole invocation handles
are retained in RAW/RETENTION/registrations. Completed namespaces must not
be rerun; a replication would need a separately registered namespace.

**Decision:** retain these necessary bounds as a qualified analysis tool.
Do not extend the finite 490 optimization, repeat the same witness construction,
or promote a nonzero residual into generic impossibility. No passing root
head has been produced. [Whole-project algebra and next](METH_491_WHOLE_ALGEBRA_AND_NEXT_20261006.md)
selects a transfer inquiry on the selected conditional function.
Convenient wholeartifact, fresh quality/SAME>=50, useful large-n, CPU LUT,
physical DRAM and multiple actual families/scales remain open.
