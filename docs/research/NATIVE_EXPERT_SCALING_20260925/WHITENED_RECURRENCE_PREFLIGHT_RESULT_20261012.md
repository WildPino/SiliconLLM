# Exact recurrent whitening and matched-output direction comparison

12 October2026. **COMPLETE / INDEPENDENT_STORED_PASS. Goal ACTIVE/INCOMPLETE.**

## Exact invariant component

Site12 adaptive384,previous accepted LOCAL finite update point. FIT-only source
recurrent covariance,PSD/eigenfloor1e-6*maxeigenvalue,condition<=1e6. Symmetric
S and inverse transform Rbar=R Si,Wbar=S W,Vbar=Si V. Reconstruct physical
channels before fullskip/gate/jointRMS;do not move coordinates through SiLU.

Physical projector and every output on all48 FIT/DEV histories preserved.
Maximum projector difference1.40472e-14,output7.54309e-15. Full gradient chain
rule G_Vbar=S^T G_Vphys,G_Wbar=Si^T G_Wphys qualified;maxchain1.83e-13. This is
an invertible optimization-coordinate change,not new capacity or information.
Actual model/generator/gate/conv/state256/skip remain unchanged/injected.

## Comparison controls and result

Both coordinate tangents are mapped into physical channels and normalized by
FULL current-reconstruction RMS-Jacobian output change on all24 FIT histories.
Alpha1 has first-order normalized output RMS1 for both. Raw parameter-Frobenius
units would be incomparable:raw output RMS122.579 for reference versus1,654,669
for whitened. Both final directions are normalized to the same output measure.

| Full FIT loss,matched output direction | Initial | Best curve alpha | Best loss | Reduction |
|---|---:|---:|---:|---:|
| Physical-reference tangent | .533004978 | .03 | .530465558 | .476% |
| Whitened-coordinate tangent | .533004978 | .1 | .520251987 | 2.393% |

Normalized directional slopes-.114456/reference and-.225383/whitened. Each
feasible curve point uses all24 FIT histories and original Armijo/minimum
improvement policy. No DEV curve fitting/choice. All fine FD probes at1e-5/
1e-6/1e-7 pass1e-6 relative;maxreference4.18e-9,whitened1.07e-9.

Recommended stored physical point:whitened alpha.1. This is a preflight
recommendation,not an installed update,actual new scan or held-out candidate
quality admission. Original coarse FD nonacceptance remains historical and
does not disappear because these different normalized directions qualify.

## Full independent audit and resources

Every92,352,512 physical/whitened output words,34,246,656 savedGY words,
45,662,208 rawJ direction words and1,572,864 parameter-replay words verified.
Independent CPU autograd BOTH graphs on all24 FIT cases,full48 output invariance,
Kronecker tangents,radial/tangent direction normalization,all FD/20 curve points/
FIT-only recommendation/SPD/inverse/projection/chain rule. All source/output
hashes before/after exact. Largest scalar difference5.55e-10;rawJ3.08e-15;
normalized-direction max3.35e-14. Parameter replay uses frozen retraction helper;
independent numerical equations are a distinct qualification.

Producer166.67569s,OSpeak1,041,506,304B,CUDAallocated693,219,840B/reserved
1,333,788,672B,new arrays1,418,700,684B on E. Audit198.05586s,
OSpeak1,517,088,768B. Timing admission false;all owned runs closed.
Source/protocol freeze4538f72,binding SHA256
`372f0e284d0cd926380f5f48400c4f284bb2f590549b338c7f180f768372cc20`.
Audit freeze267e49a,launch022be7f.

- [Producer](../../../benchmarks/native_expert_scaling/whitened_recurrence_preflight.py), [protocol](WHITENED_RECURRENCE_PREFLIGHT_PROTOCOL_20261012.md), [binding](whitened_recurrence_binding_20261012.json), [launch](whitened_recurrence_launch_20261012.json)
- [Result](whitened_recurrence_result_20261012.json), [terminal](whitened_recurrence_result_20261012.terminal.json), [E storage](whitened_recurrence_storage_20261012.json)
- [Audit](../../../benchmarks/native_expert_scaling/whitened_recurrence_audit.py), [audit protocol](WHITENED_RECURRENCE_AUDIT_PROTOCOL_20261012.md), [adjudication](whitened_recurrence_adjudication_20261012.json), [audit terminal](whitened_recurrence_adjudication_20261012.terminal.json), [closure](whitened_recurrence_session_closure_20261012.json)

## Decision

Conditioning changes the useful direction under an unchanged finite objective
and representation. Best whitened curve improves more;that does not prove a
solution of the remaining representation deficit or guarantee held-out benefit.
Next [fixed candidate actual-scan/held-out qualification](WHITENED_RECURRENCE_CANDIDATE_NEXT_20261012.md)
PROPOSED/UNIMPLEMENTED/UNBOUND. No long unchanged epochs selected. Compact causal
history,whole chatbot quality,source-state96/native cost,conditional functions/
RAM-driven n/CPU IDs+masses/DRAM,multiple families/scales and same useful
artifact>=50 remain unproven. Original engine.c unchanged.
