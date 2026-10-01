# METH-228: optimistic bias-only bound on the rejected tangent functions

Freeze before diagnosis. METH-227 all storage/source/occupancy controls
pass and E160 lowers E16 SSE24.3%, but .5970 normalized error rejects
the geometry. Do not refit its coefficients or calibrate a candidate.
Determine whether a constant local output offset could even resolve
the error, or whether changing the function itself is necessary.

## Immutable replay and diagnostic

Hash-bind METH-227 result/snapshot and saved METH-222 capture. Replay
exact full-input routing and FP32 matrix products/one bias addition
with the same deterministic GPU settings/batch shapes. Require every
128 original window energy and E16/E160/rotated SSE to match exactly.
No fit states, new source inference, training or validation-driven
candidate export. No independent-quality claim.

For E16/E160, decompose recorded FP32 output-minus-target residuals in
FP64 into within-cell centered SSE and count times cell-mean residual
energy. Require their sum to reconcile original SSE within relative1e-12.
Separately retain each fixed FP32 **prebias** product minus exact teacher
target in FP64. For each occupied validation cell, its optimal arbitrary
real constant bias is minus the mean prebias residual. The centered
sum of squares is the minimum error for those recorded products with
any real per-cell bias. This oracle uses validation targets: it is
nondeployable, not a learned/accepted model or a proposal to use these
values during inference. Do not save its bias vectors as a checkpoint.

## Rounding-aware decision

The reported gate uses one FP32 bias addition and FP32 subtraction before
FP64 squared summation. Let t=.01,u=2^-24, and
`a=sqrt(number_of_output_entries)*float32.tiny/sqrt(teacher_energy)`.
This tiny term conservatively covers underflow/flush-to-zero. Any finite
bias candidate with reported normalized SSE<=t must have unrounded
matrix-product-plus-bias normalized error norm at most

`r=(sqrt(t)+a)/(1-u)` and `b=r+(u*(1+r)+a)/(1-u)`.

By the triangle inequality and FP32 rounding bounds, the real-bias oracle
normalized SSE must therefore be<=b*b. Add a conservative numerical
guard, `b*b*(1+1e-8)+1e-12`, for FP64 reductions/means; it does not change
the candidate .01 gate. If E160's oracle exceeds this guarded bound,
exclude bias-only repair for the frozen products followed by this bias
addition/subtraction. This is a narrow mathematical bound, not a claim
about altered routing, Jacobians, nonlinear/higher-order functions,
different matrix-product arithmetic, or all affine fitting methods.
If it does not exceed the bound, only freeze a separate fit-only bias
method before any candidate test; no such fitting occurs here.

Report per-cell support, mean-error fraction, centered error and oracle;
retain exact replay rows/hashes. Both outcomes leave final quality/rate,
useful large-RAM n, engine.c integration and multi-family/10B open.

Budget15min after imports,20GiB RSS,10.5GiB GPU,<2MB output,local3060/
six threads,noT4/external source. No native retiming.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth228_tangent_bias_bound.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth228_tangent_bias_bound_result.json
```
