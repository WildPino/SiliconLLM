# Exact combined FFN remainder subset: execution state

13 October2026. Goal ACTIVE/INCOMPLETE. Producer RUNNING,not yet adjudicated.
Source freeze18e7da7;binding commite9a9fa5. Preparation terminal0/12.828s/
OS42,618,880B. Binding1,940,238B/SHA256
16acd77970ce6cbe6ceb00a23cc0b37a2723d4bb8bc95ed2bb9e7df4117393cb.

Owned worker6120 (exact creation time in terminal on exit),
started21:25:15 Europe/Rome. Foreground tool handle76408. Re-poll the same
handle or verify PID/create-time/command after an observation timeout;never
restart because a poll expired. No source model/GPU/T4/delegation.

Command (PowerShell,from repository root):

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/ffn_exact_subset.py --run --freeze 18e7da7 --binding docs/research/NATIVE_EXPERT_SCALING_20260925/ffn_exact_subset_binding_20261013.json --binding-sha 16acd77970ce6cbe6ceb00a23cc0b37a2723d4bb8bc95ed2bb9e7df4117393cb --directory E:/SiliconLLM/results/native_expert_scaling/ffn_exact_subset_20261013 --out docs/research/NATIVE_EXPERT_SCALING_20260925/ffn_exact_subset_result_20261013.json
```

Exact argument vector is also preserved by the worker's terminal receipt.

CPU6/1800s/OS8GiB/output3GiB. Array payload2,785,362,780B;720 full predicted
trajectories and3 FIT-score/ranking models. Immutable48 histories/sites0/12/23,
64 pre-existing FIT cells/budgets32/128/512/2048 plus all4608 identity control.
Actual output remains full2048;flat IDs/mass1 is not scalable routing.

After producer terminal0,run frozen ffn_exact_subset_audit.py (no arguments)
for complete720-prediction/FIT selection replay. Auditor CPU6/1800s/OS8GiB.
Then ffn_exact_subset_close.py checks preparation+producer+auditor terminal0,
owned PID absence/all pins/exact723-file E family/protected hashes. Preserve
any fault;do not waive gates or sample the audit. Interpret local coverage
only after full audit/closure. No physical ternary bank or whole-model/TPS
admission. Physical compilation and own-model quality follow only if justified.
