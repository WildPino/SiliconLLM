# METH-288: preserve pre-execution path-binding failure and freeze repair

Original freeze `3b6dc9e`; session24758 exits1 before any new CPU operator
observations. Script excluded changed engine personality from original285
source verification using a POSIX string, but original result serialized
Windows backslashes. It erroneously compared the legitimately changed
engine to old285 engine SHA. Original C/header hashes remain unchanged;
only the new engine personality intentionally changes the entry point.

Preserve [original failure](meth288_norm_reduction_qualification_result.failure.json)
and original script at freeze3b6dc9e. Original input bundle is retained.
Repair ONLY uses Path-to-Path equality for the intended engine exception.
No archive/code arithmetic/data/rows/gates/reference/resource change.
The original failure provides zero new native numerical observations.
New output/bundle/native paths prevent overwrite. Freeze this repair before
rerun; no live job remains. Original prospective288 protocol still applies.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth288_norm_reduction_qualification.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth288_norm_reduction_qualification_repair1_result.json --bundle results/native_expert_scaling/meth288_prefix_ids_repair1.bin --native results/native_expert_scaling/meth288_native_prefixes_repair1.bin
```
