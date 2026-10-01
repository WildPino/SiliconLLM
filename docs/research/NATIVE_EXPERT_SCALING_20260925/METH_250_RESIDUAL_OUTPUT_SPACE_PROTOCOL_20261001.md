# METH-250: frozen fit-only residual output-space diagnosis

METH-249 full parent is unchanged and count gain is0.0829%, below10%.
Test a specific bottleneck before another child fit: does the inherited
rank32 left space capture the error remaining in the actual parent?

Bind METH-249 raw result
`d4bd08d5e5c0773ef33a5ddbd1fa3aa2292e585bbc3f198697ba543c88a78c0e`,
original METH-247 result/checkpoint, capture/router and METH-238 operator.
Use only512 fit windows, never read validation target values. Replay all
parent labels/counts and every actual parent FP32 SSE exactly, including
the original `sum()` aggregate. No child fits, parent refit, strength/key
change or codec/native execution.

For each actual parent, compute FP64 residual `r=y-parent_prediction_F32`,
center it, and form output covariance `r_centered' r_centered` (896x896).
Extract its top32 eigenvectors, descending, canonical signs via largest
absolute component (lowest-index ties). Eigen minimum >=-1e-10 maximum,
trace/centered energy relative error<=1e-8; eigen residual and basis
orthogonality Frobenius<=1e-8. Round this unit basis once to BF16. Its
Gram condition<=1e8 and inverse identity Frobenius<=1e-8. Preserve unrounded
FP64 and actual BF16 basis tensors, physical exact readback.

Project centered residuals orthogonally onto the actual stored old and
new left spaces through their FP64 pseudoinverses. Record captured energy,
mean residual energy, and optimistic remaining error when arbitrary
within-space state corrections and a free full mean intercept are allowed.
Unrounded projected energy equals top32 eigenvalue sum within1e-8;
stored remaining error cannot beat that optimal bound (1e-10 energy
rounding allowance). This bound does not require a deployable feature
mapping and is not a trained child-function score.

License a separately frozen full-parent/private-residual-basis pilot only
if all numerical/identity controls pass and the stored new space captures
>=10% of centered residual energy AND exceeds old-space captured energy
by>=10% of centered residual energy. Otherwise stop this fixed candidate
before fitting. These are prospective fit-space feasibility gates, not
consumed count acceptance. New pilot must retain absolute1%, count/rotated/
prior10% and paired-window requirements, then a separate actual codec and
native bank if eligible. No large-n/full-model quality/rate follows.

One local RTX3060 run, six threads,10min after imports,20GiB RSS/10.5GiB
GPU,small basis checkpoint/JSON. No T4,download or fresh data. Freeze first:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth250_residual_output_space.py --checkpoint results/native_expert_scaling/meth250_layer12_actual_parent_residual_basis.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth250_residual_output_space_result.json
```
