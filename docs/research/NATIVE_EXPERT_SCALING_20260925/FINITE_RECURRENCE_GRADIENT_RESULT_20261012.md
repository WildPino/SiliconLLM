# Full finite recurrence backward and one constrained FIT update

12 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / LOCAL QUALITY FAIL.**
Goal ACTIVE/INCOMPLETE. Site12 adaptive384,fixed ranks,full source injection.

## Implemented mathematical component

Finite equal-case FIT loss between reconstructed-combined-RMS source output and
fixed unroundedF64 source teacher. Full backward includes every RMS term,all
V/W paths and fixed teacher denominators. QR gauge preserves the initial
projector;biorthogonal tangent projection solves a Sylvester equation. The
projected direction,retraction,FD controls and FIT-only candidate selection are
fully qualified. Original source weights,generator/gate/conv/state256/fullskip
remain retained;no complete compact chatbot or native speed admission.

## Actual update and held-out results

One alpha.0003 chosen from the frozen ladder,using FIT only:

| Quantity | Initial | Selected |
|---|---:|---:|
| FIT equal-case squared normalized loss | .537005642 | .533004978 |
| FIT mean ideal operator RMS | .730772777 | .728068570 |
| DEV mean ideal operator RMS | .792716631 | .789260029 |

FIT loss reduction.745%. All24 FIT and24 DEV cases improve,however all original
local budgets still FAIL. ActualF32 compact scans,controlledF64/BF16 output:
DEV ownnorm mean.789261,worst.937549,centeredmean.778653,worstdomainmean.874015.
Source-denominator oracle mean.755688,worst.862594. Original local thresholds
.1mean/.2worstcase/.2worstdomain/.1centered unchanged. No model-quality admission.

The loss curve is sharply sensitive to these parameter units:alpha.003 gives
.631504,alpha.01 gives1.669421 versus baseline.537006. This is evidence to test
conditioning,not a proof that more dimension is unnecessary or impossible.

## Faults retained; no false gate repaired after observation

V1 stopped after all24 backward records:normal W projection missed Lambda^T,
breaking tangent/slope identities. V2 corrects that real algebraic defect,
preserving all V1 arrays. V2 then stopped at the first coarse FD gate. Full
independent saved-prefix audit qualifies both complete backwards with CPU
autograd/radial-tangent action and independent Kronecker Sylvester projection.
Two partial-audit metadata faults (missing field,NumPy bool JSON) are retained;
the third partial audit completes with every hash verified.

Original FD errors at1e-3/1e-4/1e-5:.02245284/.000225185/2.251916e-6;all original
flags FALSE. Error decreases quadratically. Added1e-6 probe gives2.252001e-8.
ProducerV3 records all original false flags and separately qualifies the exact
same direction at1e-6/1e-7/1e-8 with ALL relative<=1e-6;observed2.25200e-8/
2.04584e-10/1.15737e-10. Same most stringent original tolerance. No original
failure is retrospectively admitted. See [scale protocol](FINITE_RECURRENCE_GRADIENT_CONTROL_SCALE_20261012.md).

## Full independent audit and custody

All24 full CPU autograd backwards/34,246,656 GY words;all per-case/total V/W
gradients,independent tangent,original/fine FD case losses,ten candidate points,
FIT-only selection and1,130,496 parameter-replay words verified. All48 ideal
baseline/selected outputs and all48 actual compressed scans/BF16 outputs:
184,705,024 ideal+machine output words and8,658,048 sequential scan words.
Stored source-projection max3.84e-7/full and2.33e-7/recurrent agreement.
Independent full sequential scan max4.02e-7;machine output relative difference0.
Gradient differences~1e-15,largest scalar difference2.96e-11,all before/after
hashes exact. Parameter-bit replay uses frozen retraction helper on stored
qualified direction;independent numerical equations are a separate qualification.

V3 producer209.91286s,OSpeak1,363,656,704B,CUDA allocated856,805,376B/reserved
1,803,550,720B,new output1,470,735,198B on E. Full audit186.32147s,
OSpeak1,624,571,904B. Timing admission false. V3 sourcefreeze88aaf10,bindingSHA
`583002ffd2d661e901cd76ba2843eabd97308eaae3ca8342570498929d73c71f`.
Full audit source/protocol freeze072c252,launched fromedeb82c.

- [ProducerV3](../../../benchmarks/native_expert_scaling/finite_recurrence_gradient_v3.py), [original protocol](FINITE_RECURRENCE_GRADIENT_PROTOCOL_20261012.md), [binding](finite_recurrence_gradient_v3_binding_20261012.json), [launch](finite_recurrence_gradient_v3_launch_20261012.json)
- [Original transpose fault](FINITE_RECURRENCE_GRADIENT_TRANSPOSE_FAULT_20261012.md), [FD prefix fault](FINITE_RECURRENCE_GRADIENT_FD_PREFIX_FAULT_20261012.md), [full failed-prefix qualification](finite_recurrence_gradient_partial_adjudication_v3_20261012.json)
- [Result](finite_recurrence_gradient_v3_result_20261012.json), [terminal](finite_recurrence_gradient_v3_result_20261012.terminal.json), [E storage](finite_recurrence_gradient_storage_v3_20261012.json)
- [Full audit](../../../benchmarks/native_expert_scaling/finite_recurrence_gradient_audit.py), [audit protocol](FINITE_RECURRENCE_GRADIENT_AUDIT_PROTOCOL_20261012.md), [adjudication](finite_recurrence_gradient_adjudication_20261012.json), [audit terminal](finite_recurrence_gradient_adjudication_20261012.terminal.json), [closure](finite_recurrence_gradient_session_closure_20261012.json)

## Decision

Finite backward/update/causal scan qualification now exists;the small gain does
not justify long unchanged gradient epochs. Next [exact whitening/precondition
qualification](RECURRENT_WHITENED_COORDINATES_NEXT_20261012.md),PROPOSED/UNIMPLEMENTED.
Same physical projector/capacity/objective initially,changed optimization metric;
check comparable output perturbations rather than parameter units. No state96
composition/complete own-history model/conditional bank/n/family/same useful
artifact>=50 admission. All owned runs closed;goal active and incomplete.
