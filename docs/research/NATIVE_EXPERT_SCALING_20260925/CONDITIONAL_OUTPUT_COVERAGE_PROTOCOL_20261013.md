# Fixed-bank coverage versus selector diagnosis

13 October2026. Previous input-only affine bank/full independent auditPASS,
all additional1%/all donor gatesFAIL. No bank, basis, offset or route changes.
Sites0/12/23,both frozen banks,all48 histories. Response-aware diagnostic,
never an executable cheap selector/model-quality admission.

For each target raw native F32 Y2048 (promoted F64), every expert's fixed
affine image is mu_Y,j+span(U_j). Compute every per-token squared projection
residual explicitly: e_tj=||Y_t-mu_j-U_j U_j.T(Y_t-mu_j)||^2. Exact minimum
over stored numerical e_tj; lower ID wins exact ties. Save full error matrices,
oracle IDs and actual qualified input-only IDs (uint32). True targets and an
all-expert scan are REQUIRED; these IDs are an inadmissible oracle, no route
replacement/DEV candidate selection. Finite F64 proxy, not interval-certified
real lower bound. Roundoff may affect near-tied IDs; coverage ENERGY is primary.

Per-case oracle RMS=sqrt(sum_t min_j e_tj/sum_t||Y_t||^2), actual-plane RMS,
nonnegative selector excess squared loss, label-position error, ID differences,
existing .01 extra-output gate for oracle, split/site means/worst/pass counts.
All labels retained; no fit on DEV. If even oracle misses .01, no selector
of these unchanged affine response spaces reaches that finite proxy budget
on that case. This does not exclude other trained banks/nonlinear functions.

Independent auditor does not import producer. For each basis/target32-row
block, p=(Y-mu)U and Gram=U.TU, derive e=||Y-mu||^2-2||p||^2+p Gram p.T,
including actual finite orthogonality correction. Check EVERY token/expert
error with abs(delta)/max(||Y-mu||^2,1e-30)<=1e-10; retain negative numerical
identity values/count, no silent clipping of primary residual. Metadata
abs(delta)/max(abs(reference),1)<=1e-10. Verify saved oracle IDs are exact
argmin of the fully numerically verified saved matrix; retain independent
identity argmin differences explicitly when near-ties/cancellation change IDs.
Do not claim bit-stable mathematical oracle assignment from energy tolerance.
All actual IDs exact prior packets. Recompute minima/gaps/all statistics and
before/after hashes regardless of poor oracle quality.

Prepare/run900s/audit1200s/OS6GiB/new orderedE64MiB,CPU0..5/BLAS6. Pin scripts/
helper/protocol/storage/principal actual Python/NumPy/BLAS/psutil, prior bank
report/adjudication/binding,all6 bank files/all144 native and route packets,
engine/foreign files. Source weights are provenance through qualified parent,
not read again. Before/after hashes and terminal/PID+creation records mandatory.
No source/native/GPU/T4/RESERVED/teacher/basis/router update. No new expensive
full prediction bank data; per-token80 errors all3 sites ~43.3MB.

If oracle substantially succeeds but actual route fails, study task-aware
FIT route objective/information and held-out coverage. If oracle also fails,
repair bank representation/coverage before router-only optimization. Original
source CHATBOT->compact shared core/ternary functions->engine.c, RAM-driven
useful n/multiple families/scales/OWN histories/SAME useful50 all remain open.
