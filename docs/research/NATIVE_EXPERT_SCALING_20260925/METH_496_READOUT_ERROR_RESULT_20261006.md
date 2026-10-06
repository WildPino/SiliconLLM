# METH496 — unchanged-artifact readout error, independently admitted

6 October2026. Goal ACTIVE/INCOMPLETE. Fixed495 artifact; no new candidate.
All5 main /5 independent audit /5 final admission gates PASS. The frozen
diagnostic selects `UNQUANTIZED_FITTED_FUNCTION_REQUIRES_ANALYSIS_BEFORE_PRECISION_CHANGE`.
All six U/Q/P one-percent outcomes FAIL; parameter-error RMS exceeds10 times
arithmetic-error RMS on all six groups. Both function fitting and coefficient
quantization remain obstacles. Correct source IDs are used at all three levels;
the separate495 routing/coupled quality failure is unchanged.

## Actual complete domain

ALL17,540 original bank11 UIDs,19,962 occurrences and128 original ID slots;
11,721 development /5,819 consumed-validation UIDs. All1,040 metric groups,
384 source/candidate exposure groups, rare experts and empty cells retained.
No readiness/exposure filter. ID0 remains unexposed in both splits.
Unchanged ID fidelity:56.03617438785% development /22.70149510225% consumed
validation. No actual useful-n conclusion follows from128 distinct blocks.

## Frozen numerical levels

```
H = [F64(qphi)*F64(alpha);1]
U = F64(saved_physical_L) + H C32_e^T
d = exact_integer_dot(qphi,B_I8_e)
Q = (F64(saved_physical_L) + ((F64(d)*F64(Bscale))*F64(alpha))) + F64(bias)
P = F64(saved_native_oracle)

ef = U-Y; eq = Q-U; ea = P-Q; et = P-Y = ef+eq+ea
||et||² = ||ef||²+||eq||²+||ea||²
           +2<ef,eq>+2<ef,ea>+2<eq,ea>.
```

U is a computed F64 level using serialized C32. It is neither an exact-real
evaluation nor an unrestricted best fit. Q isolates the saved I8 coefficients
and ordered F64 scale products; physical L is held fixed, so its errors against
the donor belong to ef. P includes the previously qualified native F32 output
operations. No new feature, source FFN, native or model evaluation is used.

RMS below is sqrt(sum(error²)/sum(Y²)), expressed in percent. It is not a mean
of per-state relative errors. The original UID and occurrence weights differ.
Do not add these RMS columns or interpret component energies as orthogonal
percentages of total error explained.

| Canonical UID split | n | Fitted U error % | Coefficient Q-U % | Arithmetic P-Q % | Physical P error % |
| --- | ---: | ---: | ---: | ---: | ---: |
| development |11721|2.4801648996|1.2963749761|0.0000047527|2.7985992480|
| consumed validation |5819|8.8280495689|1.2761713251|0.0000048094|8.9191529329|

| Occurrence role/mode | n | Fitted U error % | Coefficient Q-U % | Arithmetic P-Q % | Physical P error % |
| --- | ---: | ---: | ---: | ---: | ---: |
|0/0|3584|2.5783470887|1.3187304691|0.0000045369|2.8976522245|
|0/1|3091|2.5781616403|1.3180025118|0.0000045072|2.8970743429|
|1/0|3584|6.9847949316|1.2719556709|0.0000045935|7.1004908638|
|1/1|3065|6.7951396652|1.2709995806|0.0000045601|6.9134217306|
|2/0|3584|2.3836894372|1.2659514503|0.0000045174|2.6974337290|
|2/1|3054|2.3836630444|1.2652813496|0.0000044915|2.6971080876|

The main aggregate fit/parameter cross terms have both signs. The complete raw
reports retain all three inner products and13 energy/closure columns per UID.
Changing coefficient precision alone would leave this saved U above1% in all
six groups. This does not close the fixed-feature class or a different learner.

## Rare experts and information

| Development exposure class / UID split | n | Fitted U error % | Coefficient Q-U % | Physical P error % |
| --- | ---: | ---: | ---: | ---: |
|1..4 /development|7|0.0191624139|9.2710373955|9.2704030571|
|1..4 /consumed validation|3|182.9019573035|8.6068679804|182.8715309537|
|5..15 /development|36|0.1467484478|8.1096677353|8.1111250042|
|5..15 /consumed validation|19|168.6312472104|4.6593582930|168.7595225156|

These rare groups contain few UIDs. Excellent training interpolation and severe
coefficient quantization can coexist with poor generalization. Their arithmetic
RMS ratios are6.45e-7..1.19e-6; complete denominators and energies are retained.
No rare-only selection or extrapolation is justified.

From the already qualified128 cell counts: every development m_e<513;
maximum308, minimum0; even development+consumed-validation counts have maximum498.
For H_e with513 columns, rank(H_e)<=m_e exactly, independent of any floating
SVD. Thus nullity(H_e)>=513-m_e and sum_e nullity>=128*513-11721=53943.
With768 unconstrained output rows this gives at least41,428,224 coefficient
directions not identified by these development equations. This is a lower bound
on data-only ambiguity, not a parameter-count or generalization theorem for
all possible constrained/weight-informed models. The fixed regularizer resolves
ambiguity computationally; it supplies prior assumptions rather than observations.

## Closure and independent verification

Main uses blocks256 and H*C32 GEMM; audit uses127 and qphi*C32 with alpha factored
outside and bias added separately. Audit independently maps integer dot through
I64 and recomputes every energy, cross term, domain and report. It imports no
main mathematics. New dyadic controls use exact Fraction expectations; signed
512 dot uses a separate literal integer sum. All required comparisons PASS.

Main maximum vector identity absolute residual5.684341886080802e-14,
envelope ratio7.325732284020778e-5. Maximum energy closure absolute residual
1.4901161193847656e-8, envelope ratio5.009204359831751e-6. Audit envelope maxima
7.27825183082399e-5 /2.3623150102537795e-5, both<=1. Maximum absolute energy/inner
difference2.6702880859375e-5 satisfies the frozen combined relative1e-9,
absolute1e-5 comparison. It must not be compared against absolute1e-5 alone.

## Actual execution and retained resources

| Stage | Actual tool /terminal chunk | Actual exit | Full late wall s | OS peak B |
| --- | --- | ---: | ---: | ---: |
| metadata builder |52b8cf|0|5.546|42254336|
| main |b44034|0|6.016|477896704|
| independent audit |df21a2|0|6.781|479539200|
| audit Windows /finalizer |output-only wrapper; chunk/exit not retained|not retained|metadata only|not a numerical run|

Main PID22556/create1791279202.259388; audit PID23868/create1791279259.925454.
Both finish within300s/768MiB; native peak0. Builder within90s/256MiB, hashes
656,571,383B using actual-input/runtime dependencies. Main/audit budgets charge
hashing, admission, reads, all arithmetic, reports and full record writes.
Both actual Windows Event1000 queries are available, return zero events, and
identify the real process PID/create-time/time window. No first fault, repair,
namespace rerun, GPU/new resource, fit, optimizer update, projection solve,
source FFN/model/native invocation. Engine and the three foreign files preserve
their exact hashes. Retained result-directory bytes1,828,589 before this report.
The finalizer saved the complete admission with all5 gates PASS and emitted
its digest. Its terminal wrapper retained output text, not the command chunk
or exit field; no actual finalizer exit code is asserted and no rerun is made.

Minimal current binding avoids inherited ancestor scientific configuration fields
and unused compiler/source payload hashing. Compiler remains historical495
qualification. No claimed speedup in model inference follows from faster
metadata/decomposition. Completed496 numerical namespaces are terminal.

Hashes:

- Binding `830c69075edcbd8a1055351ef10f68e737a0be29e56f5fa2770e5de76e26c9ab`.
- Main `a633e57526de03d6bf17848504e0b2318be6e80b4109f10c0dbbe3143e346456`.
- Audit `984679b08f896483ec29381c972c8afef8c21f87bbf85ed49ad9aefe080c095b`.
- Admission `fa7a535528f0e913466ec83e163147ba2d31a5f95327b8b68dfb3fb7870a5e12`.

Scientific freeze4f8ced8; binding commitaf0f9a4; raw/registration7a3a852;
per-UID energies/terminal receiptsa3417aa. Audit execution HEADa3417aa.

No composed/own-state/fresh generation/task quality, SAME>=50, physical DRAM,
large useful n, additional-family/actual100B or goal promotion.

[Protocol](METH_496_READOUT_ERROR_PROTOCOL_20261006.md),
[main](meth496_readout_error_result.json),
[audit](RETENTION_496_20261006.json),
[admission](ADMISSION_496_20261006.json),
[whole algebra and next](METH_496_WHOLE_ALGEBRA_AND_NEXT_20261006.md).
