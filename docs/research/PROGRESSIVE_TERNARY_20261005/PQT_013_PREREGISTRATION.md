# PQT013: matched row alphabet and compensated rounding

6 October 2026. Prospective protocol, before new original candidate fitting.
PQT012 is completed negatively, with exact source/raw proofa9ddda48. This test
addresses the distinct alphabet/granularity mechanism identified in the primary
literature. All reused calibration/development/held-out inputs are consumed
diagnostic evidence. A positive result requires fresh final confirmation,
whole-model quality and actual native execution before research promotion.

## Question, fixed inputs and arms

Compare symmetric and asymmetric three-symbol row quantizers under matched
curvature, compensation and natural original expert contexts. Use original float
Switch-base-128 revision86c815ec05361a33a8b49fc717277da9c0a4e711, encoder11
experts81/87/18/91 and the exact original PQT012 capsule117,362,605B,
SHA654535bbed88a8a525af351b6052810fff6c495aa7845b17759db8e059a26f83.
Source and split geometry remain `pqt012_inputs.py` as frozen in31aecfa9;
no new extraction, routing, capacity change or favorable expert selection.

Six fixed arms: G64-D/G64-PR, SYM-D/SYM-PR, ASYM-D/ASYM-PR. Group64 baselines
reuse the eight actual packed D/PR artifacts from PQT012, proved ata9ddda48.
Do not repeat their fits. Re-export identical codes/scales into the new explicit
format and prove identical reconstructed coefficients. New headers count in
the actual deployed file size; baseline artifact bytes need not equal PQT012.
SYM and ASYM use row endpoints, so group64 comparisons cannot isolate asymmetry.
Only matched row SYM versus ASYM comparisons support an alphabet-effect claim.
There is no optimizer, residual, rotation, clipping search or learned scale.

Before numerical fitting, perform a metadata-only eligibility qualification.
If every calibration row maps unambiguously through retained original encoder
position/document metadata to the original input token ID, exclude IDs0 and1
(padding/end-of-sequence) from curvature contexts for an optional additional
SYM-PR-F/ASYM-PR-F pair. Each expert must retain at least64 calibration rows
from at least8 original documents. If either metadata or coverage fails for any
expert, omit the whole pair prospectively. No alternative filter or expert
selection is allowed. Retain counts, exact selected row indices and mapping
proof. Evaluation keeps all original rows and documents, including specials.
The qualification decision must be committed before fitting. The optional pair
tests filtering under fixed original contexts, not QMoE's propagated full model.
Metadata qualification001 now passes all1,024 row-token/position identities:
expert18 keeps171/236 rows from90 documents; experts81/87/91 retain all321/238/229
rows. Include the pair. For selectors identical to the complete original
calibration row list, reuse the matching unfiltered codes/levels and their
provenance instead of repeating an identical fit. Still export/seal all32
labelled artifacts, count their actual evaluations, and independently require
equal coefficients/predictions for those predetermined equivalent functions.
The filtering contrast changes context only for expert18. This prospective
deduplication precedes every original candidate fit and observes no outputs.

## Exact quantizer and compensation

F32 row endpoints: `xmin=min(min(W_row),0)`, `xmax=max(max(W_row),0)`.
ASYM levels are `{-a,0,+b}` with `a=-xmin`, `b=xmax`; all-zero rows usea=b=1.
SYM uses `a=b=max(abs(xmin),xmax)`; all-zero rows likewise use1. One-sided rows
remain fully symmetric in SYM. This differs from the author's one-sided SYM
endpoint special case, deliberately preserving the conventional symmetric
alphabet in the matched comparison. Only positive magnitude levels may emit
a nonzero code. Strict half-level comparisons assign exact ties to zero:
negative code if `a>0 and w<-a/2`, positive if `b>0 and w>b/2`, else zero.
No grid fitting or endpoint updates after the original row statistics.

For matched row PR, F32 `H=(X.T@X)*(n/2)`, matching the pinned QMoE helper's
`H/=2/n` normalization. Add `max(0.1*mean(diag(H)),1e-6)` to the diagonal.
Compute lower Cholesky, its inverse and the upper Cholesky of the inverse;
all finite with positive diagonal. No activation ordering. Commit columns in
original order, panels128; compensate remaining panel/tail weights using the
quantization error divided by the corresponding upper-factor diagonal.
Endpoints remain fixed. Cholesky failure is a counted hard stop; do not insert
an identity-Hessian fallback. Direct row arms use the same endpoints.

WI curvature uses original calibration X (or qualified filtered X). WO uses
post-ReLU output of that arm's actual quantized WI on the same selected X.
Never replace candidate post-ReLU contexts with teacher WO inputs. All natural
rows remain in calibration reporting. Dead input features get the same declared
damping floor; all-zero curvature is admitted only through that floor. Save
actual damping/curvature diagonals, attempts/completions and per-panel events.

These choices are informed by the pinned official
[QMoE quantizer](https://github.com/IST-DASLab/qmoe/blob/9110baa9466f2a7d8590e3c5dc3a5e11f7446604/quant.py)
and [rounding](https://github.com/IST-DASLab/qmoe/blob/9110baa9466f2a7d8590e3c5dc3a5e11f7446604/gptq.py).
The original routing/capacity64, natural data and F32 representation differ
from that publication. This is not a whole-model QMoE reproduction.

## Packed artifact and scientific gates

Codes are fixed-width two-bit pairs:00 zero,01 positive,10 negative,11 forbidden;
least significant pair first, exactly four coefficients per byte. Store F32
levels and NPY headers: one positive row magnitude for SYM, two row magnitudes
for ASYM, existing group64 scales for baselines. ZIP_STORED deployment files
contain actual WI/WO codes, level arrays, geometry, method metadata and hashes.
No original dense coefficients or float correction path are deployed. Count
the whole deployment file, not its codes or a transport archive, against storage.
Independently decode and execute the actual packed function at F64 in the audit.

Seal all24 or32 complete artifacts and their hashes before any numerical
development/held-out read. Qualified parent baselines are fixed, but are sealed
with all new candidates. Reuse original1% relative output-RMS and35% FP16 pair
cap9,437,184B. Every expert must pass development/held-out point RMS and the
97.5th-percentile upper bound of2,000 document-cluster bootstrap resamples.
The gate is identical for all arms; no average or task score substitutes for it.

Secondary paired RMS ratios use identical original document resamples: G64-PR
against G64-D; SYM-PR against SYM-D; ASYM-D against SYM-D; ASYM-PR against SYM-PR.
If filtering is admitted, compare SYM-PR-F against SYM-PR and ASYM-PR-F against
SYM-PR-F. Retain reference identity and samples, using seeds20261006+expert+
split_offset as in the qualified parent metric helper. An improvement is a
mechanism signal, never promotion of a candidate failing absolute gates.
Report per-expert/split values, original-weight coefficient SSE, sparsity,
endpoint magnitudes, full bytes and calibration-to-evaluation gap.

Without filtering:24 artifact seals,72 artifact/split predictions,12 original
F64 witnesses,16 new progressive matrix fits/30,720 committed columns,
32 Cholesky calls/16 inverses and16 new direct matrix quantizations. With the
pair the maximum is32 seals,96 predictions,12 witnesses,32 progressive fits/
61,440 columns,64 Cholesky calls/32 inverses; direct fits unchanged. The admitted
metadata decision above reduces actual new progressive fits to20/38,400 columns,
40 Cholesky calls/20 inverses, with12 explicitly reused matrix fits. Count actual hidden-context
and function calls separately. No source download or baseline refitting.

## Qualification, execution and retention

Before fitting, freeze source/binding and pass scalar independent endpoint/tie/
compensation controls, golden codec tests, all-zero/one-sided/dead-feature
controls, malformed header/code/level/hash rejection and all-artifact phase
controls. Compare compensation on correlated small matrices against an
independent scalar inverse/Cholesky implementation, including panel boundaries.
Local numerical qualification requires the live shared-process guard; docs and
stdlib byte/metadata inspection must not interfere with the active main checkout.

Pinned Kaggle image and Python/Torch/NumPy versions remain PQT012's qualified
runtime. Verify both physical T4 identities, F32/noAMP/TF32 off, one CPU thread,
deterministic math. Server5400s; install300/worker2400/audit1200/transport600s,
clamped to remaining time. RSS8GiB, each GPU allocator4GiB, scientific output
1GiB. Independent audit verifies every source/input/artifact identity,24/32
seals before evaluation,72/96 actual packed forwards,12 original functions,
actual counters, metric/cluster/bootstrap equations and fixed gates.
The audit uses independent F64 forwards and scalar-qualified shared helpers;
it does not claim an independent full large-matrix progressive implementation.

Reuse the already private original input dataset after fresh owner/privacy/
version verification; a new private auxiliary capsule contains only the eight
qualified baseline artifacts, calibration filter metadata if admitted and their
binding. New kernel/dataset refs must be frozen before fresh in-action account
owner/quota/reservation/all-addressable-terminal admission. No overlap or unknown
dispatch retry. Keep>=maximum1.5h+0.5h quota margin. Native timing requires the
separate owner window and is conditional on an admissible quality artifact.

Archive complete scientific raw/source files server-side into bounded lossless
transport packets after worker/auditor finish, with exact member hashes before
removing only verified original output files. Transport cannot change scientific
values or deployment storage accounting. Declare the exact packet namespace,
postpack logs/costs and temporary two-copy disk cap2GiB in the execution binding
before dispatch. Final output<=1GiB+1MiB framing; one terminal fetch. Save first
failures and costs, prove original raw/source and actual committed packet bytes.
Delegate status monitoring during long T4 execution; coordinator dormant.

If all admitted arms fail, close this matched-alphabet hypothesis at the fixed
coverage/cost. Justify any remaining materially distinct bounded method or close
the investigated scope negatively. Do not weaken gates or infer universal
impossibility. No numerical original candidate execution is authorized by a
protocol alone without its qualification, frozen binding and resource admission.
