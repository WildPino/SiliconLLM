# METH497 — exact finite feature rank and information, independently admitted

6 October2026. Goal ACTIVE/INCOMPLETE. All5 main/5 independent audit/5 admission
gates PASS. Fixed decision:
`FINITE_DEVELOPMENT_INTERPOLATION_AND_ALL_CONSUMED_FEATURE_NOVELTY_CERTIFIED`.
This qualifies exact finite-design geometry, not a fitted candidate or model.

## Complete result

All128 original source IDs,256 development/ALL cases,17,540 UID/19,962 occurrences,
11,721 development/5,819 consumed-validation UID,1,040 original count groups and
384 source-role/expert exposure groups retained and independently recounted.
No exposure filter. The127 exposed experts have full real row rank in BOTH
cases. ID0 is unexposed in both splits; its two empty rank0 cases are explicit.
All5,819 consumed feature rows extend their own expert's development row span
independently; no consumed target was read, fitted or used for prediction.

| Exact statement | Development | Development plus consumed features |
| --- | ---: | ---: |
| exposed expert cases certified full real row rank |127/127|127/127|
| total input rows /sum of real ranks |11721|17540|
| largest per-expert row count |308|498|
| feature columns per expert |513|513|
| sum of real null dimensions across128experts |53943|48124|
| unconstrained768-output coefficient null dimensions |41428224|36959232|

These are exact dimensions of the fixed real linear design defined by the
serialized qphi/alpha/bias. They are neither approximate spectral ranks nor
parameter-count claims about other architectures or constrained coefficient
sets. This dictionary's real readout can interpolate arbitrary development
targets. It can also interpolate arbitrary targets on the finite ALL set if
those targets were supplied; this is an existence statement, not an authorized
validation fit or a preservation/generalization guarantee.

## Certificate and mathematical implication

For each UID, H=[qphi*alpha,1]. I16 qphi and positive finite F32 alpha define
exact dyadic rational entries. Main maps their mantissa/exponent values into
the fixed prime field p=2147483647. Primality and denominator invertibility are
explicitly checked. All selected square minors have nonzero determinant modulo
p. Since denominators are powers of2, the corresponding rational determinants
are nonzero; rank over the reals is at least the row count and at most that
same count. No condition-number or floating cutoff is involved.

The complete certificates retain each UID order, field matrix I64 hash, chosen
original row indices in pivot order, pivot columns/raw values and determinant.
Main uses first-row forward normalized elimination. Audit reconstructs every
field entry through exact Fraction numerator/denominator arithmetic, then
checks each selected minor using last-row pivots and determinant permutation
signs. All256 minor/hash/case checks pass. No deficient modular case remains
for a nonempty expert. Empty minors have conventional determinant1 and rank0.

Let D be the development design and V its consumed-feature design for an
expert. The certificate proves rank([D;V])-rank(D)=rows(V). Thus the map

```
ker(D) -> R^(rows(V)),  delta_c -> V delta_c
```

is surjective. For every finite desired change of consumed outputs there is a
coefficient change leaving ALL development outputs identical, considered in
the unconstrained real class. This holds independently for768 output rows.
Development loss therefore cannot determine those consumed predictions. A prior,
coefficient restrictions or additional source information must supply them.
The result does NOT imply that arbitrary changes have bounded norm, stable
arithmetic, small I8 coefficients, correct donor behavior or low active cost.

## What changes relative to496

496 measured fitted U RMS2.38%..6.98% on the six occurrence groups; quantization
Q-U1.27%..1.32%, tiny P-Q.497 now proves finite development representability
in the same unconstrained real feature class, so its nonzero development
residual is not a finite-data class impossibility. The regularized solve,
serialization, prior constraints and conditioning remain relevant. General
representation outside these finite states, paired odd/even weighting and
source preservation remain open. Validation feature independence explains
missing identification, not the numerical magnitude of the observed errors.

Routing/coupled failure from495 is unchanged: canonical ID56.0362%/22.7015%,
coupled RMS71.0824%/75.8041%. No new keys, probabilities or candidate IDs were
computed. The384 groups here contain source exposures only. Original candidate
choice/energy reports remain in495/496 rather than being relabeled as497 work.

## Actual retained first fault and numbered metadata repair

Original metadata builder tool/chunk d6e4c8 exits1 before field/control/rank
observations: the historical495 admission SHA literal was61 rather than64
characters. Original frozen code/failure/registration retained before repair.
No source495 artifact changed. Repair1 corrects that digest, guards digest
lengths and links the first fault; unchanged497 science then runs once.
No prime, pivot, threshold, data or decision change and no numerical rerun.
Original failed builder PID/create-time were not emitted and remain unknown.

| Stage | Actual tool /terminal chunk /session | Actual exit | Late wall s | Parent OS peak B |
| --- | --- | ---: | ---: | ---: |
| original metadata builder |d6e4c8 /d6e4c8 /none|1|0.218|28356608|
| repair1 metadata builder |53c023 /53c023 /none|0|2.187|30744576|
| first main |6639f5 /c59608 /80008|0|17.641|177123328|
| first independent audit |f33c76 /f33c76 /none|0|7.656|175005696|
| metadata finalizer |49aebe /49aebe /none|0|metadata only|not numerical|

Repair builder PID24684/create1791280680.2074099; main PID28996/create
1791280701.3502254; audit PID25176/create1791280790.8974702. Builder90s/256MiB,
main/audit600s/512MiB parent budgets satisfied. Hashing, reads, controls,
integer arithmetic, reports and full record writes charged. Metadata git
duration is charged; transient metadata git OS peaks are not separately claimed.
Native/scientific child peak0. Both Windows Event1000 queries available and
return zero events, matched to actual process/time instances.

No coefficient/target/bank/compiler/source FFN payload is a current numerical
input. No source FFN/model/native call, readout solve, optimizer update,
new resource, candidate export, quality/rate/useful-n/family/goal promotion.
Engine and the three foreign tracked files preserve exact hashes. The complete
main and audit numerical namespaces are now terminal; do not rerun them.

Scientific freeze95c5239; original fault retention8911de1; repair source484b3cc,
repair document correction48d2ec2; binding62e8c4a; main artifacts a86ca15.

- Binding `89a7a5829cd15f933ed63f2378f7b5a7cc9f8498877da85625f47836291e2864`.
- Main `d0d5a4694b3b43e7d5ec2ae970653b0bf408991440c8dd181dcf9db4e0e4c7d3`.
- Audit `0d574c0d2226f0a7dfd208d6643462f529204154371f640cc6b53fcea73826b4`.
- Admission `6121e58ce83bb53fa12eaf3cda554c3d33288027c5ab7957ee35f97d780a35b5`.
- Original builder fault `ea3c230138818d8b4f522730bc86ff57fb98717e98146e7753b1e60f500c529b`.

[Protocol](METH_497_EXACT_FEATURE_RANK_PROTOCOL_20261006.md),
[metadata repair](METH_497_R1_BINDING_METADATA_REPAIR_20261006.md),
[main](meth497_exact_feature_rank_result.json),
[audit](RETENTION_497_20261006.json),
[admission](ADMISSION_497_20261006.json),
[whole algebra and next](METH_497_WHOLE_ALGEBRA_AND_NEXT_20261006.md).
