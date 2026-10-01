# METH-249 exact aggregation recovery

Retention-only producer5240e53/session76007 exits1 after23.515s,again
before any validation target. All160 fits and physical checkpoint complete.
All16 retained parent SSEs equal METH-247 exactly;the pooled `+=` result
is4.5870074608943345e-5 rather than4.587007460894334e-5. The original
METH-247 producer uses Python `sum()`,whose floating accumulation differs
from sequential `+=`. The separately frozen read-only diagnostic also
uses `sum()` and is exact. This is aggregation apparatus,not changed
predictions. Preserve first failure SHA
`b2765ea5c35de4d02fd8c7e2e18917435c1d4d8d3776c80b8ad0042c79a46d00`,
diagnostic SHA
`21825bbbff61fa58dd4c3f7bfa169690197af14fc96e90b5b4f6bbbc2cc0c36e`,
retention producer failure SHA
`105535b560e1e0eee2ed7a04eb3750515bbd94d99755c45d0e9c808cad8a42fb`.

Completed checkpoint298,727,124bytes SHA
`5b95ecc272b257929388ac6520cfcbdf3643a9e4aa4aa21c0acd1e69482df7a0`.
Recover it without fitting. Verify failure stage/error/no validation rows,
all160 IDs/counts/normal/mean/fallback controls,all16 exact parent SSEs.
Pool with original `sum()` over common recorded fit energy,require exact
old parent ratio. Read snapshot,verify all retained old fields unchanged,
finite tensors/exact key set,all inverse/growth and176 decoded coefficient
distinctness controls. Same absolute/nonincrease fit prerequisites.

Only after all gates,execute the original frozen128-window consumed
validation once with all original exact FP32 controls and FP64 comparison
gates,including10% count/rotated/prior and paired-bootstrap thresholds.
No tolerances or scientific choices change. Future original producer uses
`sum()` for exact parent aggregate too. Source fallback was applied only
to zero-support156;do not count this as fitted slope evidence.

Local3060/six threads,5min recovery,20GiB RSS/10.5GiB GPU,no T4/download.
Original fit budget checks passed every row;its failure does not preserve
peak RSS/GPU values,so report recovery peaks separately. Freeze before run:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth249_recover_aggregation.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth249_parent_anchored_latent_result.json
```
