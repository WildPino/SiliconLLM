# METH-249 apparatus retention addendum

Frozen318da16/session48737 exits1 after20.187s. All160 child solves,
source fallback and coefficients complete,but the exact pooled parent
score assertion fails before checkpoint/validation. Only audit rows survive;
the latent coefficient arrays were not saved and cannot be recovered.
This is an incomplete apparatus attempt,not a count-quality conclusion.

Read-only parent diagnostic frozen8e27e39 reproduces all16 METH-247
fit SSE values and pooled ratio **exactly**,2.109s,no child fitting or
validation targets. It does not resolve why the preceding stateful runner
failed. Preserve both raw records without relaxing any gate.

Amend only retention: record per-parent SSE/relative discrepancy,save/read
back the full completed snapshot before the same exact score assertion,
and retain all parent controls in failure JSON. All scientific choices,
controls and thresholds remain frozen. Rerunning completed fit systems is
necessary because the first invocation lost its weight arrays. Use new
repair1 paths,never overwrite the first failure. If the same exact score
fails,the snapshot and controls permit a separate offline/numerical
diagnosis without another fit. Same20min/3060/six-thread resource budget.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth249_parent_anchored_latent_pilot.py --checkpoint results/native_expert_scaling/meth249_layer12_parent_anchored_continuous_repair1.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth249_parent_anchored_latent_repair1_result.json
```
