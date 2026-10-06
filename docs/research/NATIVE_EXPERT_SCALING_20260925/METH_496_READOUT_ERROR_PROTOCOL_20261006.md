# METH496 — fixed-artifact readout error decomposition

Prospective before any496 control/decomposition observation. Previous goal
turn PROGRESS:495 full local fit/native/audit admitted FAIL. Goal ACTIVE/INCOMPLETE.

## Decision

Separate empirical error of the saved unquantized fitted readout from physical
coefficient and output arithmetic errors, on ALL original bank11 source UIDs
and occurrences. No new fit/update/width/prior/checkpoint/ID sweep, alternative
candidate/export, native/source FFN/model replay or new resources.

Use the unchanged495 bank, serialized fitted C32, saved feature qphi/alpha and
physical L, saved native oracle P, source493 Y and UID/occurrence identity.
Current dependencies are only these bytes, source495 raw/retention/admission/
binding metadata, actual Python/NumPy/psutil/threadpoolctl runtime, new scientific
files and operational receipt helpers. The compiler and unused source payloads
remain historical qualifications, not current inputs. Build a minimal new
binding; inherited ancestor configuration fields do not define496 arithmetic.

## Fixed levels and accounting

For each UID and its original source ID e, form H=[F64(qphi)*F64(alpha);1].

```
U = F64(physical_L) + H C32_e^T
d = exact integer dot(qphi,B_I8_e)
Q = (F64(physical_L) + ((F64(d)*F64(Bscale))*F64(alpha))) + F64(bias)
P = F64(saved_native_oracle)

Efit = U-F64(Y)
Eparameters = Q-U
Earithmetic = P-Q
Etotal = P-F64(Y) = Efit+Eparameters+Earithmetic.
```

Q uses the prescribed ordered F64 products/additions BEFORE495's F32 output
operations. U is a computed F64 GEMM level, not a formal exact-real result;
small F64 evaluation differences belong to this numerical definition/limitation.
Bias is the saved C32/physical bias; no new calibration. Integer products and
all intermediate integer sums are below2^53, so F64 GEMM of integer operands
gives exact d; all actual B totals are also within I32 bounds.

Save source/fit/parameter/arithmetic/total and Q-Y energies, all three pairwise
inner products and both vector/energy closure residuals. Do not sum RMS values
or describe nonorthogonal energies as percentages of total error explained.
Vector identity tolerance3e-12*max(1,abs(Y)+abs(U)+abs(Q)+abs(P)) per coordinate.
Energy closure tolerance1e-10*max(1,source_energy+sum(error_energies)+
2*sum(abs(pairwise_inner_products))). Both envelope ratios must≤1.

Independent audit uses blocks127 instead of256 and factors alpha outside
qphi*C32 GEMM for U, with C32 bias added separately. It does not import main
or its mathematics. Q's integer dot is independently converted through I64
before the frozen scale products. Compare energies/inner products relative1e-9,
absolute1e-5; tiny closure residuals are independently checked by their own
envelopes and compared as residual limits, not bit equality. Reports compare
relative1e-9/absolute1e-5; counts/identities exact. No rank or interval theorem.

## Complete domain and reports

ALL17540UID/19962occurrences/128ID slots.11721development roleflags&5,
5819consumed-validation roleflags&2, disjoint. All source controls accepted.
Report UID splits,128*2ID/split cells,4rare-exposure classes*2splits,
192*2modes*2controls=768views,3roles*2modes=6occurrence aggregates,
384source/candidate exposure groups and unchanged ID fidelity. No filter to
ready experts; the empty ID remains explicitly unexposed. Consumed validation
is diagnostic and never selects another predictor.

Diagnostic outcomes: all-six unquantized-U RMS≤1%, decoded-Q RMS≤1%, saved-P
RMS≤1%, and all-six parameter-error RMS≥10*arithmetic-error RMS. Compare P
and ID/denominators against qualified495 raw metrics. If U misses any fixed
role/mode threshold, investigate fitted-function/projection/regularization
before a precision change. If U passes all and P fails, use parameter versus
arithmetic evidence to select the physical readout variable. These outcomes
do not qualify a new candidate or remove the observed routing failure.

## Wire and budgets

energy_by_uid.bin <8sIIQ: M496ENG1,width104,columns13,count17540; rows13F64:
sourceE,fitE,parametersE,arithmeticE,totalE,fit_parameters_inner,
fit_arithmetic_inner,parameters_arithmetic_inner,Q_Y_E,identity_maxabs,
identity_envelope_ratio,energy_closure_abs,energy_closure_envelope_ratio.
Total1824184B. Other outputs are small controls and raw complete reports.

Metadata builder90s/256MiB; main300s/768MiB and audit300s/768MiB, CPU0/BLAS1.
Each directory64MiB/all496 outputs128MiB/raw8MiB. All hashes, metadata admission,
reads, matrix operations, reports/output writes charged. Stream C32 per expert
and state blocks; no all-expert F64 matrices or saved alternative predictions.
No child/native/model calls or optimizer/projection solves. Memory/deadline
observer runs through full record write. Filesystem-size checks once per second
plus forced terminal checks; file writes are bounded by these fixed schemas.
Actual sole tool/session/PID/create-time/exit/Windows Event1000 admission.
First faults and partials precede numbered repairs; no namespace rerun.

New controls: dyadic four-coordinate U/Q/F32P example with signed cross terms,
independent exact Fraction identity/energies; a new signed512 integer dot pattern
q_i=(17*i)%32768,w_i=(3*i)%255-127, checked by literal Python integer sum.
No completed old control replay. Commit all helpers/protocol before builder;
commit binding before sole main, then independent audit/finalizer.
