# Finite recurrence output, full RMS sensitivity and head/time coupling

12 October2026. **COMPLETE / INDEPENDENT_STORED_PASS.** Goal ACTIVE/INCOMPLETE.

## Scope and exact quantities

All48 source histories/sites0/12/23,two frozen recurrent-only384 channel layouts.
Ideal projection residual e=(VW^T-I)(y-D*x);actual-scan discrepancy measured
separately. No new fit,scan,source forward,optimizer or native call.
F=O diag(gamma),g=SiLU(source gate),u=g*y,r=sqrt(mean(u²)+sourceepsilon).
Unrounded F64 output o=F u/r;full action
`J_y e=F(g*e)/r-o*[u^T(g*e)/(3072*r²)]`.
Finite delta is the actual unrounded output difference for y+e versus y.
This smooth operator is distinct from the discontinuous BF16 rounded operator.

Full finite delta/J vectors,per-position raw energies,all head energies/rho and
FIT-only corrected/no-radial mean Grams are retained. Independent-time proxies
use verified prior FIT GY_rec and the existing projectors. DEV diagnostics never
select a fitted Gram,basis or action. The selected action uses predeclared FIT
heuristics,not a capacity lower bound or new quality gate.

## Diagnosis that changes the next action

Adaptive384,equal-case means:

| Site | FIT remainder / finite delta | DEV remainder / finite delta | FIT paired/no-radial trace ratio | FIT change in summed head energy from radial term |
|---|---:|---:|---:|---:|
| 0 | .168424 | .177222 | 1.067474 | .021420 |
| 12 | .300442 | .330782 | 1.207911 | .031338 |
| 23 | .142074 | .148231 | 1.184819 | .014695 |

At12 the tangent approximation misses30% FIT/33% DEV of the finite output
difference in relative RMS. The FIT remainder flag>.25 selects
COUPLED_FINITE_OBJECTIVE_NEXT for both layouts. A corrected averaged Gram alone
is not selected. Cross-head energy fraction is-.234141 FIT/-.289748 DEV for
adaptive12;the FIT fraction is below the predefined.25 flag while DEV shows
stronger cancellation. Do not silently use DEV to change the action threshold.

Radial VECTOR magnitude can still be substantial:adaptive12 correction norm
relative to no-radial action .402108 FIT/.420898 DEV. The3.13% figure above is
the change in summed per-head ENERGY,not a claim that the RMS derivative is
negligible. These quantities describe different interactions.

At23 fixed8,the paired/trace ratio2.059584 and radial head-energy effect.271824
also trigger the coupled finite objective. Adaptive23 avoids these flags and
retains its previous locally accepted representation;no additional fit there
is selected. Site0 has no heuristic sensitivity flag,while local quality still
fails:absence of a flag proves neither optimality nor sufficient capacity.

Ideal finite output RMS agrees closely with preceding actual-scan results:
adaptive DEV .273068/.792717/.063484 at0/12/23. Mean actual-scan-versus-ideal
residual discrepancies are3.39e-7/2.83e-7/2.86e-6 relative to ideal residual.
This separates the representation deficit from the qualified scan arithmetic.

## Qualification and resources

432 explicit2048x64 head-Jacobian Gram controls at three frozen positions of the
shortest FIT case plus18 central-difference controls (three step sizes/two
layouts/three sites). All original control tolerances pass. Producer Gram max
relative1.19118e-15. Largest FD relative1.35230e-6 occurs under the appropriate
step-dependent threshold;do not compare it to the threshold of another step.

Independent CPU audit covers every554,115,072 finite/J vector word,1,488,102 raw
statistic words and19,480,608 head-energy/rho words. Uses radial/tangent action
and explicit head output vectors rather than producer energy contractions.
All FIT Grams/proxies/controls/aggregates/decisions and hashes are checked.

Independent audit PASS:full vector relative5.63409e-14,scalar1.32723e-14,
Gram3.35144e-16,control scalarabsolute6.29592e-12;all bound hashes before/after
exact. Audit406.64650s,OSpeak1,107,787,776B. All owned sessions closed.
FD max by step1e-3/1e-4/1e-5:1.35230e-6/1.35232e-8/1.40195e-10.

Producer212.59050s,OSpeak1,128,161,280B,CUDA allocated932,720,128B/reserved
1,086,324,736B.147 NPZ artifacts total4,610,587,410B on authorized E storage.
No clean timing/native speed admission. Source/protocol freeze74bedd2,binding
SHA256`59ad6f9a1712cd618433784e8b31e645e323ffd6cbcc69b0429997a9fa10a931`.
Audit source/protocol freezea1dfa23,launch4835bf2.

- [Producer](../../../benchmarks/native_expert_scaling/coupled_recurrence_sensitivity.py), [protocol](COUPLED_RECURRENCE_SENSITIVITY_PROTOCOL_20261012.md), [binding](coupled_recurrence_sensitivity_binding_20261012.json), [launch](coupled_recurrence_sensitivity_launch_20261012.json)
- [External storage receipt](coupled_recurrence_sensitivity_storage_20261012.json), [result](coupled_recurrence_sensitivity_result_20261012.json), [terminal](coupled_recurrence_sensitivity_result_20261012.terminal.json)
- [Audit](../../../benchmarks/native_expert_scaling/coupled_recurrence_sensitivity_audit.py), [audit protocol](COUPLED_RECURRENCE_SENSITIVITY_AUDIT_PROTOCOL_20261012.md), [audit launch](coupled_recurrence_sensitivity_audit_launch_20261012.json)
- [Adjudication](coupled_recurrence_sensitivity_adjudication_20261012.json), [audit terminal](coupled_recurrence_sensitivity_adjudication_20261012.terminal.json), [closure](coupled_recurrence_sensitivity_session_closure_20261012.json)

## Selected correction and remaining goal

[Finite-output gradient/one bounded FIT update](COUPLED_FINITE_RECURRENCE_GRADIENT_NEXT_20261012.md),
PROPOSED/UNIMPLEMENTED/UNBOUND. Exact backward of reconstructed combined RMS,
equal-case finite source output loss,fixed adaptive384 ranks,explicit projector
retraction/conditioning/derivative audit before a bounded FIT-only step. New W
then requires new actual compressed scans;no source-label substitute.
No hours-long unchanged epochs/corrected-mean-Gram rerun selected.

Local accepted site23 remains a candidate for source-derived native layout/cost
qualification with retained generators/gates/conv/skip included. Source-state96
composition,history acquisition,whole-model conversation,n/DRAM/IDs+masses,
multifamily/scales and same useful artifact>=50 remain unproven. No original
engine change or complete chatbot admission from this diagnostic.
