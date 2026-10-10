# Next attribution: does the auxiliary target agree with the learned decoder?

10 October2026. PROPOSED, UNEXECUTED. Finish/audit the matched actual51 family
first. No source inference/new labels/optimizer/RESERVED/T4 or native replay.

Implementation preparation: [executable protocol](ORIGINAL_LATENT_READOUT_ATTRIBUTION_PROTOCOL_20261010.md)
and worker/held launcher/stored auditor/audit launcher are prepared and AST
parsed, still UNEXECUTED/unbound. The protocol corrects only storage pricing:
lossless F64 full-V scores require4,599,124,512B and a5GiB output cap; the
original2.3GB estimate corresponded to F32. Same two heads/labels/decisions.

## Why this uncertainty matters

After24 new updates A native DEV KL8.76437->6.51931, tasks0/16; B6.51611.
A centered-relative boundary error0.90222->0.90468. Final-output recovery has
not recovered this fixed source-coordinate trajectory. B's first measured
auxiliary core gradients are nonzero, but small relative to KL gradients.
The proposed test separates compatibility of the auxiliary target with the
current decoder from recovery of that target by the actual internal path.
Do not infer that weight1 is optimal, that longer fitting is useless, or that
all compact state is impossible. No unchanged long campaign is selected.

## Algebra and a limit of simply adding experts

For the current artifact, logits are `W_C a_C(history)` with W_C shape65537x256;
RMS normalization and final gamma only change the256-vector a_C. Increasing n
can enrich the nonlinear map to a_C; it does not enlarge the current fixed
head's column space. In the quotient by constant logit shifts (softmax invariant),
the available linear dimension is at most256. The donor head can have a larger
span, but its observed probability manifold may be much smaller. This is a
matrix-rank statement, NOT a chatbot-quality lower bound or a proof D256 fails.
Conditional decoder subspaces could enlarge the total available span with
bounded selected work, but usefulness, normalized selection and native cost
would need demonstration. Do not add such an operator on rank arithmetic alone.

## Deterministic stored intervention, no decoder fitting

Reuse all24 DEV projected final source residuals Z=h24P, all4386 complete label
distributions, actual A51/B51 packed F32 head/gamma and native final metrics.
At the already fixed label positions compute in F64, separately for each head:

`a_Z = (Z / sqrt(mean(Z^2)+1e-5)) * final_norm_C`

`logits_Z = a_Z @ head_C.T`.

Compare the complete65537-way distribution with the donor and with retained
native outputs: per-label/case/domain KL, disagreement, teacher entropy and
uniform KL. Save the full new label score arrays plus independent stable F64
log-partition checks. Bind all consumed extents/protocol/code before contracting.
This is an idealized target-injection consistency control, not an optimum over
all decoders, source state forcing in a chatbot, an engine numerical result,
fresh behavior, physical timing or speed admission. A learned head may have
adapted to a different coordinate system, making injection worse; that outcome
would expose loss/decoder inconsistency, not prove an information ceiling.

## Prospective interpretation

- If BOTH injected controls reduce native DEV KL by>=50% and disagreement by>=20%
  relatively, the existing decoder can read this target substantially better:
  investigate functional history/boundary recovery before changing readout span.
- If BOTH retain>=80% of native KL and still fail absolute/domain screens,
  teacher-coordinate matching has not made the current decoder sufficient.
  Investigate task-sensitive latent/decoder coupling or conditional readout;
  distinguish coordinate mismatch from rank loss before any width change.
- Otherwise retain the mixed per-case/domain result; no single-stage explanation.

These choose a diagnostic direction; they are not quality admission gates.
Do not refit on DEV, sweep maps, repeat the old source-state decoder studies,
or claim causal recovery from an injected control. Actual own-history task
quality and same-artifact accepted50 remain decisive for a converted model.

## Cost before a possible implementation

Pure stored CPU matrix contractions, two label-only head controls/no history
learner. Reuse2x67.1MB F32 heads,24 projected histories and BF16 labels; expected
lossless F64 score outputs4,599,124,512B. Price~2–5min plus IO,
hard900s/4GiBOS/5GiBoutputs/log8MiB,
six CPU threads/no owned timing overlap, no GPU/source/optimizer. Freeze a new
worker/binding/held launcher before executing. Preserve first faults and adopt
only durable outputs. This file is a plan; no such controls have run yet.
