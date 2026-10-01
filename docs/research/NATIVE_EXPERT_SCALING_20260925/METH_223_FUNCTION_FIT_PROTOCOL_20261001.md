# METH-223: saved fit gap before changing the learned geometry

Freeze before execution. METH-222 fails actual donor function accuracy
and useful E16->E160 gain. It does not yet distinguish fitting limitation,
larger-bank generalization, coefficient precision, or sparse support.
Read only its exact saved capture, coefficient artifact and result;
no new source-model inference or tuning. Reproduce original fit cell
counts and every validation sequence's common/E16/E160 SSE exactly.
Stop on reconciliation failure, preserve it, and do not reinterpret
diagnostic as a passing new model.

Measure fit and validation SSE of the unchanged stored BF16-effective
functions. Replay the exact fixed common FP64 ridge regression only;
require rounded weight and recomputed bias to equal saved tensors.
Report unrounded FP32 common as a diagnostic control, never a selected
replacement or new quality result. Record validation loss by selected
cell's original fit state count: <65,65–255,>=256; counts are not
independent samples and correlations do not prove a causal data effect.

Fixed interpretation:

- Common-representation limitation if unrounded common fit normalized
  SSE>=0.05 and removing BF16 coefficient rounding improves validation
  normalized SSE by <=0.005. Specify a nonlinear common; no precision retry.
- Larger-bank generalization gap if E160 fit SSE<=90% E16 fit SSE, but
  E160 validation SSE exceeds E16. This is the measured fit/validation
  relationship on these rows, not proof of a universal cause or n limit.
- Small-support concentration only if <65-count cells cover >=1% of
  validation states and account for >=50% of the positive E160–E16
  validation SSE gap. Otherwise do not blame only rare cells.

These diagnoses can hold simultaneously. Use them to freeze the next
changed nonlinear common/regularized specialist mechanism, retaining
real pretrained-output targets and independent quality/native gates.
Do not repeat METH-222 or replace its failed result. Five minutes local,
six threads/RTX3060,20GiB RSS,10.5GiB GPU,<1MB additional report. No T4,
new corpus, model training, generation/task or large-n CPU measurement.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth223_function_fit_diagnostic.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth223_function_fit_diagnostic_result.json
```
