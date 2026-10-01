# METH-235: frozen output-only source derivative prior qualification

## Uncertainty and changed mechanism

Can source values/derivatives transfer into the complete4864-feature
row-Q8/LUT output basis without an896x896 affine branch, while the actually
encoded readouts conserve the source function? METH-234 qualifies the
source operator at8.522ms with.01944% scoped source-function error. It does
not qualify projected priors, fitted readouts or useful expert-count gain.
METH-231's selected-feature/affine geometry stays rejected.

Reuse original pinned Qwen0.5B layer12 matrices; METH-234 actual binary/
table/source controls and result SHA256
`972a451f89fdbcc567b6f1f52df0e22bd2185f28d52cf3179b560b8ba8fba815`;
METH-227 actual16 parent/16x10 child keys with result SHA256
`85f5b41e99850a2eeb32d0c071f32060f35d0e958db4729fffa115681a406fd2`;
METH-222 capture SHA256
`2ba4b548f189f548a954bbaf595e192e982da6498c0d230fbce1f11597773ef5`.
Only first512x128 fit inputs are used, not captured output labels,
validation inputs/labels or any new quality source. Require original
parent fit labels/hash/counts and all source/binary segment hashes exact.

## Output-only projection

For each of16 frozen parent centers c, use the actual fixed row-Q8 gate/up
and shared LUT to obtain phi(c),H4864. Analytic feature Jacobian A=J_phi(c)
uses interval slope16*(table[i+1]-table[i]),zero for gate<=-16,one for
gate>=16 or rounded position>=512,with the declared FP32 boundary policy.
Source smooth SiLU supplies original f(c),J(c). B0 is decoded actual
row-Q8 source down weights,not original BF16 down.

FP64 thin QR A=Q*T. R=J-B0*A; solve triangular T^T*X=R^T and set
deltaB=X^T*Q^T. Raw B=B0+deltaB is the minimum-Frobenius-norm correction
when A is full column rank. No damping, rank truncation, fit-selected
strength or alternate solver after outcome. Require finite singular
values and condition(T)<=1e8 before solve; fail immediately if violated.
This gate bounds conditioning,not capacity or a universal QR guarantee.

Cast raw B to FP32,encode with the existing symmetric row-Q8 codec,FP32
scales. Set stored FP32 bias=f(c)-row_linear(phi(c),codes,scales),with scale
after reduction. No affine term/input change. Raw FP32 bias separately
checks unrounded value. Save all16 actual readouts,shared inputs/table,
source-down baseline,router keys in safetensors; read every tensor exactly.
Hash coefficient pairs using codes plus scales,not bias/noise alone.

## Frozen gates and reporting

- All bindings/source/fit route controls and every snapshot tensor exact.
- All16 projection condition numbers<=1e8; finite coefficients/bias/outputs.
- Correction Frobenius norm/base norm<=.25 in every prior; no hidden
  coefficient explosion accepted merely because the center matches.
- Analytic feature derivative checks at parents0,15: fixed16 original
  rows `[0,1,127,255,511,767,1023,1535,2047,2559,3071,3583,4095,4351,4607,4863]`
  versus reverse autograd and JVP along input axes0,127,511,895 and c/||c||;
  every relative Frobenius/L2<=1e-5. This is scoped derivative qualification,
  not an all-coordinate/all-center autograd sweep.
- Every FP64 raw B*A versus source J relative Frobenius<=1e-5. Parents0,15
  full unrounded FP32 output autograd:relative Frobenius<=1e-5,peak-relative
  max error<=1e-4.
- Every raw/stored source-center relative output L2<=1e-5.
- Every stored effective-Jacobian relative Frobenius distortion<=.01;
  report the unrounded and stored derivatives separately. Stored int8
  derivatives are not called exact.
- Pool all65536 fit-input source-output SSE/energy<=.01,original
  BF16-weight/FP32 FFN target. Report all16 routed-cell errors and unchanged
  source-down baseline; no baseline-relative gain gate and no separate
  cell-error gate. Evaluate in512-state chunks, no output-label fitting.
- All16 encoded coefficient pairs distinct. This prerequisite does not
  establish useful n gain or replace the future E16/E160 comparison.

All pass licenses separately frozen full-feature conditional fitting.
Any failed gate stops this prior mechanism before fitting/validation.
No METH-231 gate is relaxed; this is a new output-only representation.
No complete LLM/held-out generation/tasks,routed C bank,large-RAM route/LUT,
accepted-token rate or sparse-GigaChat applicability follows from a pass.

## Budget, command and persistence

One local RTX3060 run,six host threads;20min after imports,20GiB RSS,
10.5GiB GPU,2GiB disk free before launch,<100MB snapshot plus small JSON.
No T4/external source. QR conditioning failure records stage/progress and
stops; numerical/fidelity gate failures record all completed measurements.
Commit protocol and runner before execution; keep checkpoint ignored and
report byte hashes reproducible with pinned CRLF. Partial records preserve
each completed parent; never restart an unconfirmed-live job after timeout.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth235_full_feature_output_priors.py --checkpoint results/native_expert_scaling/meth235_layer12_full_feature_output_priors.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth235_full_feature_output_prior_result.json
```
