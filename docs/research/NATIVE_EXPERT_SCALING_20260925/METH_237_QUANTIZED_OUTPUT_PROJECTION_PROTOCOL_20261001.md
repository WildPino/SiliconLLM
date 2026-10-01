# METH-237: frozen quantized-feedback output projection

## Changed mechanism and decision

METH-235's one-shot continuous projection passes but every rounded prior
fails1% source-derivative fidelity. METH-236 proves row scales alone cannot
repair those fixed codes. Implement **quantization-aware alternating
projection**,changing actual code assignment while keeping the qualified
row-Q8/FP32/LUT format and active shape. This is a changed compiler procedure;
the old one-shot recipe and its failed gates remain stopped.

Bind METH-235 result/snapshot, METH-236 result SHA256
`d6febbb46d8387935a6c0653ad3df171588a8b0a6ae294b1a457a3b72d478a23`,
METH-234 physical input/table and original source. All METH-235 gates other
than stored-gradient fidelity must have passed; all16 scale-only bounds
must exceed1%. Reuse same16 source centers and fit-input route/hash/counts.
No captured output-label fitting or validation/new-source access.

## Fixed algorithm

For each parent,start at its **stored METH-235 row-Q8 prior**,decoded as
FP32 code-times-scale then FP64. Fixed full-feature Jacobian A and original
source J are unchanged. Reuse thin QR A=Q*T and qualified conditioning.
Exactly **four** cycles:

1. R=J-B_stored*A.
2. deltaB=solve(T^T,R^T)^T*Q^T; raw B=B_stored+deltaB.
3. Cast raw B to FP32 and encode with the unchanged row-max-Q8 codec.
4. Decode actual new coefficients,report source-gradient residual and
   feed **those** into the next cycle.

No damping,optimized scale search,validation stopping,best-cycle choice,
extra iterations,width/precision changes or noisy distinctness perturbation.
Only the fourth cycle is judged. Record all coefficient hashes/residuals
and repeated encodings so stalls/cycles are visible. A failure stops this
fixed alternating-projection recipe; no increase of iteration count.

After cycle4,FP32 bias conserves source value at c under actual row-scaled
output arithmetic. Keep all11 input/table/source-down/router control
tensors unchanged; save and read back every actual tensor. Coefficient
hashes include codes+scales; source information must create16 distinct
encoded readouts,not bias/noise alone.

## Frozen gates, costs and limitations

- All source/control/route bindings exact; replay each original METH-235
  fit-source SSE within1e-12 relative (absolute floor1e-12),same512-state
  chunks. All snapshot tensors read back exactly.
- Every unrounded FP64 cycle reconstructs source J within1e-5 relative
  Frobenius. At parents0,15 final unrounded FP32 output autograd has
  relative Frobenius<=1e-5 and peak-relative max<=1e-4. Feature-derivative
  checks are inherited from the unchanged,hash-bound METH-235 basis.
- All final stored source-center relative output L2<=1e-5; finite coefficients,
  scales,bias. Final decoded coefficient distance from original source-down
  baseline/base norm<=.25 for every parent.
- All final encoded source-gradient errors<=.01,**unchanged** from METH-235.
- Pooled fit-input original-source FFN SSE/energy<=.01; report old/new
  pooled and all16 cell errors,not output-label/held-out quality. No
  baseline-relative function improvement requirement.
- All16 final coefficient pairs distinct.

All pass licenses separately frozen actual conditional readout fitting.
No full model,independent quality,n-scaled route/LUT/DRAM,native trained-bank
cost or>=50 accepted token/s claim follows. Shared full-feature operator
remains METH-234's8.522ms prerequisite,not a measured trained-bank rate.

One local3060 run,six host threads,20min after imports,20GiB RSS,10.5GiB GPU,
2GiB free disk,<100MB snapshot+reports,no T4. Freeze code/protocol before
execution; completed parents/iteration histories survive exceptions.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth237_quantized_output_projection.py --checkpoint results/native_expert_scaling/meth237_layer12_quantized_output_priors.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth237_quantized_output_projection_result.json
```
