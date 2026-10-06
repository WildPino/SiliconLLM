# Next method decision: calibration breadth and transfer

6 October 2026. **Planning decision, not an executable preregistration.** No
PQT-009 original-model function, new corpus selection, fit or dispatch has run.
The authoritative research goal is unchanged.

PQT-008 establishes that global hard-code adaptation works computationally:
all matrix counters advance, final codes change and exported hard states replay.
It also establishes severe preservation failure. Final calibration agreement
near 69% drops below 4% on new news identities. The data used for adaptation
contain only 2,048 unique prediction positions repeated sixteen times. That
gap does not isolate data coverage, domain, precision, scale geometry or the
surrogate gradient. Increasing updates on those same windows without an
identifiable question would be weak evidence.

The next question is whether **broader calibration at the same update and
position-exposure budget** improves independent complete-model fidelity, and
whether gains occur on fresh same-domain contexts as well as new news contexts.
Retain the existing one-shot/staged comparison so breadth is not silently
confounded with a new optimizer, precision exception or final representation.

Proposed controlled design: narrow calibration with sixteen consumed windows,
versus 256 distinct WikiText train windows containing that narrow subset and
a deterministic spread over the remaining train corpus. Each one-shot/staged
arm gets exactly 256 applied updates and 32,768 prediction-position exposures;
the narrow arm repeats samples, the broad arm sees 32,768 unique positions
once. All arms start at the original checkpoint; every unique matrix counter,
fixed scale, vector policy, hard-forward surrogate and stage exposure stays
explicit. A different window sequence is an intended part of breadth; it is
not a matched numerical optimization trajectory.

The executable protocol must fix exact selection/windows/token identities,
calibration overlap rules, schedules, seeds, frozen inputs/source commits,
quality and relative criteria, costs, finite/time/memory stops and separate
audit before any new original-model/data access. Repeated narrow arms should
reproduce the already frozen PQT-008 hard-code identities before new evaluation;
if reproduction fails, preserve the first failure and resolve the apparatus
instead of assigning it to calibration breadth. New evaluations exclude all
consumed research token contexts, all 32 PQT-007/008 news rows and their prefixes.
Prospective same-domain and news metrics remain separate; no pooled average
can hide a failed domain or substitute for the existing absolute gates.

The broad teacher targets must be generated per window or represented with
bounded memory. Saving 256 complete F32 vocabulary-logit windows would approach
20 GB and exceed a T4's memory; avoid an unbounded copy of PQT-008's sixteen-
window cache. Retain exact token/state/target hash identities and have the
independent process reconstruct every teacher target. Count fitting memory,
source generation, exports and audit costs. Final hard archives alone must
still contain only codes/scales, counted original vectors and descriptors.

This comparison can test transfer at a fixed small adaptation budget; it cannot
prove that calibration breadth is the only cause, establish semantic/factual
capacity or establish expert CPU usefulness. A failure remains scoped. Native
timing awaits the explicit CPU window independently; do not treat that pending
measurement as blocking all GPU research or as permission to occupy shared CPU.

Resume by reading PQT_008_RESULTS.md and TERNARY_INDEX.md, binding all consumed
selection artifacts and freezing the executable PQT-009 protocol. Then qualify
synthetic selection/gradient/archive controls, source-free checks and exact
committed source before fresh private Kaggle admission/dispatch. Use the existing
delegated monitor for long T4 runs and wait for its terminal/failure notice.
