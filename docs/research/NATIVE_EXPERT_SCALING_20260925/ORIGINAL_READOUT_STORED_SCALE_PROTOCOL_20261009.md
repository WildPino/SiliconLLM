# Post-hoc actual-direction scale interpretation

Selected AFTER observing the frozen72-label convex readout result; not part of
that prospective experiment or a held-out decision. Freeze this code before
execution. No source/model/native replay, GPU or parameter updates.

The freely chosen features/control sqrt8 achieved a much lower sampled feasible
loss. That does not imply scaling the current model's wrongly directed features
will improve its actual distribution. Reuse all256 saved source rows and their
byte-matched original1507-history native logits. Convert BF16 source exactly;
F64 entropy, uniform KL and KL to softmax(s*native_logits) for s=1 and sqrt8.
All inputs hash pre/post. Argmax is invariant for positive scale.

If full-label mean KL decreases, amplitude change remains a candidate for a
separately frozen/exported native variant. If it increases, retain current
initializer/Adam1 for the first broader whole recovery; more confidence on these
actual features is harmful. This concerns one FIT trajectory and F64 scaling
of saved native F32 logits, not exact F32 gamma re-export, fresh chatbot quality,
DEV/family/generalization or capacity. Do not alter the historical native FAIL.

Script [original_readout_stored_scale.py](../../../benchmarks/native_expert_scaling/original_readout_stored_scale.py),
local CPU-only stored-file audit, expected seconds, no large output. First-stage
limits/results and input provenance remain immutable. The next recovery must
measure actual native outputs, not these freely optimized per-label features.
