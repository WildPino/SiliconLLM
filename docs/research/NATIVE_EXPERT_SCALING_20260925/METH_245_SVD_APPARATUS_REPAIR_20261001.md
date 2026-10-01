# METH-245: factor-construction precision apparatus repair

Original source/operator/protocol freeze `3b5706b`,session6823 exits1
after4.547s at the first layer's SVD energy gate. **Zero factors,zero
source-function screens,zero native executions**. Preserve the13,237,280
byte incomplete fixture and failure JSON SHA256
`fb8b0f9c6971a0a1d33ffb6a4dbdf30422093424f4cf0ad80e1834c4fba1262a`.
This is an apparatus failure,not rejection of the rank32 C representation.

Bound first-layer-only precision diagnostic session75669 completes,exit0.
FP32 CUDA gesvdj has energy relative discrepancy **2.3733183e-4**,above
the unchanged1e-5 gate;FP64 gesvdj gives **3.7037040e-13**. Both are finite
and sorted. Residual squared norm.06332713037727014. Top32 energy fraction
.10753069/.10754163 respectively. Diagnostic JSON SHA256
`6333489c06187263ba9308f56a70bd172dccf15ff5a589bf03e549cb6af9b452`.
No new source/validation/native timing is performed by this diagnostic.

Repair only the construction solver to **FP64 CUDA gesvdj on the same
FP32 residual**,balanced factors then rounded BF16 exactly as before.
Keep rank32,all base bytes,canonical sign,codec/C/layout,numerical/source/
cost gates and budget unchanged. Native operator consumes the same BF16
layout;no threshold relaxation or native retiming. Freeze repair before
rerunning,using new physical/report filenames so failed artifacts survive.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth245_separate_residual_feasibility.py --binary results/native_expert_scaling/meth245_separate_residual_fixture_repair1.bin --exe benchmarks/native_expert_scaling/meth245_separate_residual_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth245_separate_residual_native_repair1_result.json
```
