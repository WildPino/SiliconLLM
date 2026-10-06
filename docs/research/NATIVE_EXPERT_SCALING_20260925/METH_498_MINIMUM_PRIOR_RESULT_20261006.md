# METH498: minimum-prior interpolation admitted; fixed local recipe FAIL

6 October 2026. Goal ACTIVE/INCOMPLETE. [Prospective protocol](METH_498_MINIMUM_PRIOR_PROTOCOL_20261006.md).
One change from495: equality-constrained minimum displacement from the SAME
source prior, replacing its .01 penalized fit. A/L,512features+bias, prior
metric/calibration, product keys and I8 codec remain fixed.

## Complete evidence and decision

ALL9 main /6 independent audit /5 final admission gates PASS. All128 original
coefficient cases,127 development-only solves,17540UID,19962occurrences,
1040metric groups and384source/candidate exposure groups qualify. ID0 retains
the complete existing prior and has no observed UID; no expert filtering.

All five frozen local outcomes FAIL. Decision: NEXT_SOURCE_PRIOR_TRANSPORT.
The interpolant is numerically feasible and stationary in its fixed metric;
it still fails consumed function quality before serialization/quantization.
No lambda/width/prior/checkpoint/ID/optimizer sweep or validation-target fit.

| Canonical UID split | N | Fitted F64 RMS | Fitted F32 RMS | Native correct-ID RMS | Native coupled RMS | ID fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development |11721 |3.5240572e-15 |3.3299150e-7 |11.435345% |79.123372% |56.036174% |
| Consumed validation |5819 |9.518808% |9.518808% |14.759822% |98.679417% |22.701495% |

RMS ratios are relative to the source weighted-function energy. Tiny F64
development residuals are numerical measurements, not exact arithmetic claims.
Consumed validation has been used diagnostically in prior work; it is not fresh.

| Occurrence role/mode | N | Fitted F64 RMS | Native correct-ID RMS | Native coupled RMS | ID fidelity |
| --- | ---: | ---: | ---: | ---: | ---: |
|0/0 |3584 |3.472262e-15 |11.735577% |74.225851% |53.627232% |
|0/1 |3091 |3.472388e-15 |11.735042% |73.266577% |58.589453% |
|1/0 |3584 |7.528140% |13.571779% |94.674896% |22.767857% |
|1/1 |3065 |7.348061% |13.471386% |81.718080% |24.274062% |
|2/0 |3584 |3.573682e-15 |11.153110% |79.043530% |52.873884% |
|2/1 |3054 |3.573906e-15 |11.153616% |75.582001% |57.105435% |

F64 and F32 pass the1% function criterion on the four development role/modes
but fail on both consumed role/modes. Native correct-ID, coupled and ID gates
fail on every role/mode. Aggregate ALL-six gates are consequently false.

## Algebra, numerical qualification and physical accounting

Let H=[dequantized_qphi,1], R=Y-physical_L, Kreg=T T^T and Cprior the SAME
F32 calibrated source prior widened to F64. Using Z^T=T^-1 H^T=Q Rqr:

    C64=Cprior+(T^-T Q Rqr^-T (R-H Cprior^T))^T.

The real minimum is unique by497 exact development row rank and positive
Kreg. Main uses triangular substitutions/QR; independent audit reverses
feature order, factors the permuted metric and verifies equality and whitened
stationarity without another coefficient fit. All127 solves qualify.

Main maximum per-expert development RMS1.7905451e-14; independent audit
1.6636704e-14. Main feasibility-envelope ratio<=1.248405e-6,
stationarity<=5.500853e-5; independent<=1.259381e-6/0.002551210.
All are well inside the fixed numerical envelopes. These are numerical checks,
not outward-rounded optimality certificates. QR absolute diagonals range
0.1235481..37.584104 across cases; these are not singular values or a condition
number. No cutoff, extra ridge, inverse or numerical fallback.

F32(C64)==saved_C32 BYTE on all 50,429,952 coefficients
(128*768*513). Independent I8 RNE/max-scale encoding verifies every new
B/scale/bias byte. A/keys and all private L bytes remain unchanged, as does
the empty expert's entire physical block. All17540 new native ID/probability/
logit bytes equal495; independent integer/F64/F32 reconstruction agrees BYTE
with every correct-ID and coupled output. One native invocation; no controls,
Adam, old candidate prediction or source/model program was replayed.

Four nonorthogonal errors are retained perUID:

    ef=U64-Y; es=U32-U64; eq=Qphysical-U32; ea=Pnative-Qphysical
    Pnative-Y=ef+es+eq+ea.

| Canonical split | F32 serialization RMS ratio | I8 parameter RMS ratio | Arithmetic RMS ratio |
| --- | ---: | ---: | ---: |
| Development |3.329915e-7 |0.114353418 |4.832201e-8 |
| Consumed validation |3.261688e-7 |0.112104546 |4.920232e-8 |

All six inner products and vector/energy closure qualify. Maximum main
identity-envelope ratio0.0001421074, energy0.000006087646. RMS values must not
be added or treated as orthogonal energy shares. The increase from the
496 coefficient ratio~1.28% to~11.2% concerns a changed fitted readout, not
an arithmetic regression or a universal I8 floor.

Rare development classes1..4 and5..15 have7/36UIDs; their F64 fit is near
numerical zero. Consumed N3/N19 remain182.9018%/168.6291% RMS, compared with
182.902%/168.631% on496. Their tiny source-energy denominators are retained;
these small groups neither define a full-distribution floor nor disappear.

## Actual executions, resources and provenance

| Stage | Actual tool / session / terminal; exit | Charged wall seconds | OS peak bytes |
| --- | --- | ---: | ---: |
| Sole metadata builder |b68507;0 |5.562 |55,291,904 |
| Sole full main |326641 /93177 /637ea8;0 |36.953 |parent768,958,464 +native131,522,560 |
| Sole independent audit |66210d /54656 /4ffd01;0 |19.859 |841,637,888 |
| Sole metadata finalizer |9f7c4c;0 |tool wall0.312 |not a model measurement |

Main PID27200/create1791282887.4474812; native PID19208/create1791282910.9312344;
audit PID27492/create1791282973.3617656. Native elapsed5.608813s is a complete
17540-row diagnostic batch, NOT token rate. One observer exit race is recorded;
actual child exit0, qualified observed modules and complete bytes retained.
Windows queries1d15dc/main and a0548e/audit are available and return zero events,
with exact parent/native instance/time matching in final admission.

Main and audit900s/1536MiB, builder90s/256MiB, native120s; CPU0/BLAS1.
No GPU, download, installation, source FFN/model call or new resource. Full
hash/read/solve/native/verification/report-write costs and terminal receipts
are charged; transient metadata git subprocess peaks are not separately measured.
No numerical/binding/control/native/audit fault or namespace rerun.
Output-list display truncation and metadata path probes are not numerical reruns.

Core local conversion evidence totals844,787,976B:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| bank.bin |127232136 |3d069c2aaccb18ef58e0675a5423f127133f7cd97b557c476645259ef9a73439 |
| coefficients_F64.bin |403439640 |15580737e9e4ab9f988af230406a4cc40af76012b3d589e64c5b30b93ae8f702 |
| coefficients_F32.bin |201719832 |60fc7f4e02d41ab4565df4487d904479283b47bc5e0fe0fc7e1aade78eb868a0 |
| predictions.bin |109589944 |b7dee577041406cf744935dd2e530c516a7e7e1f879df52851e74cbeda0999ec |
| energy_by_uid.bin |2806424 |d1b9a3345aa8efcf4bc41ec0d9a688dcb1c63e3edb339bd6fc07d68a3074a908 |

Large assets stay local in results/native_expert_scaling/meth498_minimum_prior_fit,
hash-qualified by RAW/RET; helpers, reports, small controls/resources/receipts are
in Git. These conversion coefficients are not deployment bytes. The physical
bank stays127,232,136B; old logical12-bank weight reads14,587,008B/token exclude
core/head/state/LUT/other traffic and do not establish DRAM or speed.

Freeze a43a318; binding1939927; main3008d85; independent audit6ff4c3c.

    BIND b85fd8a136e546f7d940f518811081b2b9e993554f11c84adb6039990524bf37
    RAW  da4f297a8370dddc7e34039d6c0e763750f24789b407f11bf0c7a6cd323ce957
    RET  e6e57f7f94b712512a782c98888eeb456f4f3e08bc58d78fd86e6cfd39970384
    ADM  bff9a2ed8533bc28a0fc1892151cbedd7fb1274da62103bddea09cc4f032bac6

[RAW](meth498_minimum_prior_fit_result.json), [RET](RETENTION_498_20261006.json),
[ADMISSION](ADMISSION_498_20261006.json), [binding](meth498_binding.json);
sole main/audit/finalizer registrations retain actual tool/chunk/session/exit.

## Whole-project decision and resumption

This fixed interpolating pF recipe is CLOSED in its quality criteria.
It is not a conversion, feature-class, I8 or source-prior impossibility proof.
[Whole algebra/next](METH_498_WHOLE_ALGEBRA_AND_NEXT_20261006.md) selects an
explicit variable-mass factorization using unweighted donor priors and actual
saved F/p, before new teacher information or a precision-only change.
No499 implementation/control/capture/fit/native/audit has occurred.

All composed/all-bank/own-state fresh prediction/generation/task quality,
SAME artifact>=50 accepted batch1, useful n/RAM, CPU LUT/winner/mass, physical
DRAM and actual additional families/scales remain joint open gates.
Original489 CPU128 rate is a separate positive artifact. Engine/three unrelated
tracked hashes preserve; completed495/496/497/498 namespaces are terminal.
