# PQT-005-R1: output namespace collision repair

Prospective repair, 5 October 2026. Inherit the entire question, original input
dataset version 1 / hashes, experts, arms, fitting methods and actual update
counts, seed/splits/probes, metrics, gates, budgets and independent audit from
[PQT-005](PQT_005_PROTOCOL.md). No numerical method or tolerance changes.

The first private run, kernel 137234255/version 1, failed at 34.712 seconds
inside the executor (`experiment` process 47.203s, install 20.324s). It had
frozen all expert/arm representations and the original native pair files but
had not evaluated development/probe functions. Its first failure is retained:
`ValueError: array already exists` at the evaluation loop's unconditional
`base.save(e8_calibration_inputs,x)`. The file was already intentionally saved
before fitting. The exclusive writer correctly rejected the duplicate.

R1 omits that second calibration-input write, reusing the existing immutable
array. Other groups still receive new exclusive files. Preserve the original
run/artifacts/source `f468117` and fresh R1 output/job namespaces. R1 repeats
the complete deterministic fit rather than resume from an undocumented state.
After retrieval, compare every original final representation, original pair,
calibration input/reference and FF schedule against R1; exclude time-dependent
training/event records from byte reproducibility. No first-run quality outcome
exists and no fitting choice is selected from its numbers.

Run ID `PQT-005-R1`, private acct2
`giggio253/pqt-005-r1-original-experts-20261005-002`. Same pinned image,
5400-second server budget and 900/3300/900 install/experiment/audit stops.
Keep the embedded input manifest's original PQT-005 identity: it names the
unchanged version-1 private source dataset, not the repaired executor output.
Freeze source and fresh bundle before live admission. Monitor through the
delegated Luna agent as requested by the owner; the coordinating process waits
and resumes only after terminal state, failure or a meaningful admission issue.
No local CPU-native timing or whole-model claim. Research goal incomplete.
