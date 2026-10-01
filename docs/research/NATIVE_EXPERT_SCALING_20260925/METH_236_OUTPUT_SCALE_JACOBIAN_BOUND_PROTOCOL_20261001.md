# METH-236: frozen fixed-code output-scale Jacobian lower bound

METH-235's continuous source projection passes, but all16 actually encoded
row-Q8 priors have>1% derivative distortion. Determine whether changing
only their FP32 row scales could ever meet the same1% bound. Reuse original
source, actual full-feature gate/up/LUT and METH-235 snapshot/prior result
SHA256 `132013d3991622143d268090015900abfcbb88e22d5799eb4df1073b4a97b140`.
No fit/validation inputs, labels, fitting, new source or native timing.

At each original parent center, construct same A=J_phi and source J.
Keep every integer output code fixed. For output row i, v=q_i*A. The
positive row scale minimizing ||s_i*v-J_i||^2 is
`s_i=(v dot J_i)/(v dot v)`. Compute in FP64,require nonzero v and positive
finite optimum. Check normal-equation relative residual<=1e-10. Replay
original stored gradient error within1e-7 (accounting for decoded-FP32
weight multiplication versus FP64 scalar products). Report FP64 optimum
and FP32-rounded-scale distortion. No new physical prior is produced.

If **any** parent's FP64 lower bound exceeds.01, exclude scale-only repair
for these fixed codes before function fitting. If all bounds pass, license
a separately frozen actual calibrated-prior variant with stored/point/
source-function checks. A bound pass is not a prior pass or quality gain.
No METH-235 threshold relaxation or ignored worst prior.

This metric uses source Jacobian response over a conditional full nonlinear
basis; it is not the failed METH-195/196 grouped global weight-MSE scale
optimization. The bound can rule out this narrower mechanism without
repeating its quality screens.

One local3060 run,six host threads,10min after imports,20GiB RSS,10.5GiB GPU;
small JSON only,no T4/new download. Source/snapshot/control hashes fixed;
failure records stage and completed rows. Freeze before execution.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth236_output_scale_jacobian_bound.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth236_output_scale_jacobian_bound_result.json
```
