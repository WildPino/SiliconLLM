# Complete source-directional diagnosis: the transfer objective must constrain structure

8 October2026. Goal ACTIVE/INCOMPLETE. This is progress on the actual
CHATBOT->compact conditional functions->C pipeline: new source information and
a verified decision about its conversion objective. It is not a converted
whole checkpoint, semantic-quality result, native arithmetic or accepted50.

## Pipeline decision

Canonical Qwen interaction, original source calibration, complete necessary
target cost and existing archive/C loading are available. The transition that
fails is source function->compact functions: previous E16 value fit has21.68%
FIT/61.39% novel DEVELOPMENT response RMS. Its16 functions are supported; uniform
E160 has28 unsupported leaves. Native assembly cannot repair these responses.

The NEW complete Jacobian diagnosis confirms missing source-directional
structure at both FIT and novel anchors. All four prewritten diagnostic gates
pass, and independent saved-matrix reconstruction agrees. Select ONE separately
bound value+derivative joint learner at the SAME geometry/active cost. Do not
add epochs/width/precision to the closed value-only recipe or qualify more slots
from RAM/hash uniqueness. [Next implementation](CHATBOT_STRUCTURAL_TRANSFER_NEXT_20261008.md).

## Measured result

Exactly16 FIT+16 novel DEVELOPMENT original BF16 states,32 exact-distinct x.
FIT anchors come from14 original conversations. Anchors were selected BEFORE
derivative values by original selected-parent mass, original row index, distinct
x and fixed interior/shadow prerequisites. They are representatives, not a
random/complete domain sample. Source and fitted functions read all896 inputs.

| Equal-anchor pooled Frobenius metric | FIT | Novel DEVELOPMENT |
|---|---:|---:|
| Source full-J energy |887.3960064101537|917.7525122819903|
| Error full-J energy |607.4837722351004|628.2413147525938|
| Total relative J RMS |82.73868336%|82.73713071%|
| Selector-rowspace relative J RMS |73.24956152%|73.79896978%|
| Selector-nullspace relative J RMS |83.27645927%|83.23385603%|
| Source nullspace energy share |94.31152418%|94.43556126%|
| Anchors with total relative J RMS>10% |16/16|16/16|

DEV total>10%, source null share>=50%, null relative RMS>10% and>=8 anchors
above10% all PASS. Decision:
`DIRECTIONAL_DEFICIT_SUPPORTED_NEW_STRUCTURAL_FIT_JUSTIFIED`.
This is the diagnostic's prewritten decision, not proof that the next learner
will work or that derivatives cause the whole chatbot's errors.

The actual32x896 F32 selector has **exact real rank32/nullspace864**. Nonzero
minor columns0..31, determinant163870988 modulo1000000007 after exact common
2^-149 scaling proves independence. Independent exact-integer-ratio decode/
minor elimination agrees. QR orthogonality5.1733e-15, projection relative
roundoff1.0893e-15; these numerical observations are separate from the rank proof.

Within fixed top4 interior cells, the student derivative includes BOTH changes
of each function and changes of selected softmax mass. All320 source/student
directional pairs (32 anchors*5 directions*2 models) pass independent central
finite differences. Maximum discrepancy/tolerance3.94965e-5. Independent NumPy
Jv differs by at most6.4463e-17; saved FD reconstruction gap0. Direction set:
e0/e31/e447/e895/normalized selector-null e895. All retained perturbed points
remain inside their original selected cell. This does not test boundary jumps.

Mass-J null-energy shares~1.0..1.33e-30 are acquisition scalars consistent with
the rowspace formula. Full mass-J matrices were not separately saved; those
component scalars are explicitly NOT independently reconstructed. Complete
source/error J metrics, full-J/probe frame SHA and all decision predicates are.

## Algebra, information and limits

Write P for the selector and v in ker(P). In a fixed cell, P(x+t*v)=Px and
the selected masses stay constant. Therefore the shared and four selected
full-input functions must carry the source's variation in v. More leaves chosen
from the same P do not alone distinguish states in this fibre. This says where
information must live; it does not prove the target's width/capacity insufficient.

The observed null energy is high, but the nullspace also spans864/896 input
dimensions. Frobenius energy uses isotropic mathematical directions, not the
frequency/covariance of reachable chatbot states. Do not interpret94.4% as a
percentage of lost semantic information or prove that the selector itself is
the cause. Derivative errors also occur in its visible rowspace.

For a smooth error e along an interior segment from anchor a to x,

    e(x) = e(a) + integral_0^1 J_e(a+t*(x-a))*(x-a) dt.

Thus value and derivative preservation are linked. A bound on a FULL SEGMENT
would control propagated local response error; one anchor derivative does not
provide that bound. Fitting sampled values alone constrains neither this integral
nor all input directions. New derivative supervision is a source-informed
constraint, not a guarantee of novel response or own-history fidelity.

All acquired derivatives are F64 calculations of smooth SwiGLU/softmax formulas
with original BF16 source and retained F32 student coefficients exactly promoted.
They are NOT derivatives of the rounded BF16/F32 inference program. Individual
leaf values and NEW x+/-h*v surrogate responses are derivative operands;
new original BF16 full forwards0, PCA/clustering/control replay0, optimizer0.
Original source capture/response predictions/a-squared features are not repeated.

## Reproducibility, resources and first provenance fault

[Protocol](CHATBOT_DIRECTIONAL_PROTOCOL_20261008.md),
[implementation amendment](CHATBOT_DIRECTIONAL_IMPLEMENTATION_20261008.md),
[independent audit protocol](CHATBOT_DIRECTIONAL_AUDIT_PROTOCOL_20261008.md).
Workers: `chatbot_directional_plan.py`, `chatbot_directional_jacobian.py`,
`chatbot_directional_audit.py`; common monitored launcher/binders in the same
benchmark directory. Commands, used-file/runtime/actual resolved-path SHA,
process creation identities, exact output manifests and actual exit are in
the corresponding raw/terminal JSON. Restricted runtime/local3060 only.

| Job | Source freeze | Worker s | Family s | Worker OS through exit B | Conservative family B |
|---|---|---:|---:|---:|---:|
| First anchor plan |ec197f9|2.172|4.640|49393664|see raw terminal|
| First full-J acquisition |c0aba67|31.156|126.188|859705344|892026880|
| Independent retained audit |1c2fdb3|3.219|6.282|507547648|537878528|

Acquisition GPU peaks277238272 allocated/304087040 reservedB. Source/student
J each[32,896,896] C-order F64,205521024B INCLUDING128B NPY header. Full probes
[32,5,2,4,896], Q, directions and fsynced per-anchor journal; ALL420609465 output
bytes preserved under `results/native_expert_scaling/chatbot_directional_jacobian_20261008/`.
No times are native rate or conversion throughput. Family includes hash/runtime
checks:126.188s is the actual whole job, not just31.156s compute. Launcher final
receipt/stdout tail remain outside its last peak snapshot, explicitly scoped.

First plan command mistakenly passed `9bbeef-placeholder` as revision. That
RAW FIELD is INVALID; raw and terminal remain unchanged. Separate administrative
`chatbot_directional_plan_provenance_20261008.json` verifies the actual pre-value
ec197f9 commit against every committed bound file using SHA and only LF/CRLF
conversion; raw/terminal SHA pinned. No anchor replay or silent replacement.
Generic launcher now requires40 lowercase hex revision characters. Phase2 bound
the correction and committed actual anchor bytes before derivative acquisition.

Raw SHA:

- plan `bcbb9a5921ece474870375c438c42a474b0f115b85f9e3a35366cf1916b165dc`;
- full-J `576f0c5a8495456fe6d4dac9b0055e8a32d964546029d00ddc4f89c337356b22`;
- audit `3885e228dc7022d7f962edf07cb74d445870320626192c8e4c06f107e61564d2`.

Administrative `chatbot_directional_terminal_20261008.json` closes all6 known
instances, no matching Windows Event1000 faults, typedUTC/exact worker FILETIME/
launcher<=16ticks float tolerance; controls179810/179791. Foreign three tracked
SHA unchanged. No live acquisition/audit. Goal remains INCOMPLETE: eligible local
converter, ALL24/native integration, fresh dialogue/task/own-history quality AND
accepted50 SAME artifact, useful n/LUT winner+mass/actual DRAM/other scales open.
