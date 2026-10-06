# Next method decision after complete-model conversion

6 October 2026. Planning record, **not a preregistered executable experiment**.
No PQT-008 fitting, data selection or GPU dispatch has occurred.

PQT-007 gives a stronger scoped negative result than earlier projections:
source-guided independent fitting improves all 169 local objectives but does
not preserve the complete model. Retain that result instead of selecting
different thresholds, excluding matrices silently or tuning on its news rows.

The next useful question is whether **bounded global behavior adaptation
during progressive conversion** can repair interactions that independent
matrix fitting does not address, while exporting actual ternary codes and
counted scales. Compare a staged conversion with an otherwise matched
one-shot hard conversion using the same number of applied optimizer steps,
calibration contexts, end-to-end source-distribution objective and final
representation. Initialization is the existing pretrained checkpoint, not
pretraining from scratch. Trainable latent float weights/optimizer buffers are
temporary fitting costs; they must not survive as an uncounted deployed path.

Before implementation/dispatch, decide and freeze the exact conversion order,
stage boundaries, hard-forward gradient surrogate, fixed/adapted scale policy,
optimizer/counters, loss geometry, batching, seeds, evaluation exclusions and
finite/resource stops. Qualify gradients/actual hard code changes on synthetic
controls and audit complete model states independently of optimization. Do not
claim that the earlier single-projection STE trial already tests this global
progressive schedule. Do not claim the new schedule will succeed.

Use current archives as descriptive fixed baselines only. Evaluation must use
new token identities; exclude all sixteen PQT-007 AG News rows and prior
research windows. Reusing the same provider split is independent of fitting
only with explicit exclusions and prospective selection; it is not untouched
project-wide holdout. Consider whether the narrow strict behavior gates and
a separate capability diagnostic answer distinct questions, before the next
results. Keep any prospective gate change visible and avoid reinterpretation
of the existing failed gates as passes.

Meanwhile, the bounded same-artifact native expert timing is ready. It depends
on the owner's explicit uncontended CPU window, independently of GPU work.
Preserve that pending dependency and do not infer permission from silence.
