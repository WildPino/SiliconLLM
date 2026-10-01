# METH-242: frozen source-value-only precision correction

## Question and changed information

METH-240's actual full-feature mixed E160 functions lose22.39% against E16.
METH-241 reproduces all58,720,256 fit target elements exactly using the
original BF16 FFN at captured (4,128,896) geometry. FP32 source values
differ, and constant-input child156 resets its better fitted parent to a
less accurate FP32 source value. This supports testing the value mismatch,
not attributing the complete count failure to it.

Change only child source-point values to canonical BF16 source execution.
Retain all actual METH-240 parent/child weight arrays, physical fixed32
codec, fitted E16 biases, source-FP32 derivative projections, input maps,
lookup, keys, labels, feature variances and tau1024. BF16 centroids are
rounded inputs; source-FP32 sensitivities remain approximations, not
mathematical derivatives of the discrete BF16 function. No slope refit,
strength tuning, new exception count or modified acceptance thresholds.

Bind METH-240 result SHA256
`3c02724b1194f305036aae6896a489e1d19b477d7d8fe1d3ddc45fa9579fc505`,
snapshot `23f4a92b8092ccfbf79575ecd2dd59bd300b50fd6124e818cfbf862327def611`,
METH-241 result `33568b9dc99a1ba93d43346db58a97865815e35cb8c669ea8656d44dd70b1980`,
and their original source/capture/native/route bindings.

## Exact correction and gates

At each160 child center c, repeat c to(4,128,896), cast input BF16 and
execute original BF16 gate/up/SiLU/product/down, take first output f16(c).
Using the actual separate row-scaled-int8 plus BF16-escape accumulation,
set `bp_new=FP32(f16(c)-mixed_readout_without_bias(phi(c),Wp))`.

The centered ridge slope system has centered residuals, so changing a
constant prior intercept does not change its solution. With actual old
stored prior/intercept arrays, let `delta=FP64(bp_new)-FP64(bp_old)` and
`gamma=n/(n+1024)`. Transport the fitted intercept exactly as
`bf_new=FP32(FP64(bf_old)+(1-gamma)*delta)`. No new label-fitting solve.
Verify the defining FP64 centered-fit mean/intercept equation independently
on every child fit cell, relative to mean-target norm <=1e-5. Actual mixed
center value error relative to BF16 source <=1e-5 for every child.

Before validation require exact label/count hashes, all physical snapshot
tensors read back, every tensor except `child_prior.bias`/`e160.bias`
byte-unchanged, all176 effective weight hashes distinct and equal METH-240,
unchanged parent-prior/E16 fit scores exactly equal METH-240. Save a complete
new physical checkpoint, do not silently replace the rejected artifact.

Then score the same128 already consumed development windows once for
parent_prior/E16/child_prior/E160/rotated-child E160 with actual mixed
operator. Require per-window energies and unchanged parent scores exact.
Retain complete function SSE/energy<=.01 and E160 SSE<=90% of each E16,
rotated E160, child prior and parent prior. Paired10,000 window-bootstrap
gain P05>0, seed242243. Report fit and validation metrics, every correction
and per-window arm. These data are consumed, not fresh independent quality.

Any prerequisite failure stops before validation. Count/function failure
closes this source-value-only correction as a count solution; no strength,
input rounding or intercept retry. Pass licenses a separately frozen
stored native bank test, not full-model/10B or >=50tok/s acceptance.

## Budget and command

One local3060 run,six host threads,10min after imports,20GiB RSS,10.5GiB GPU.
Require4GiB free; one checkpoint <1.7GB plus JSON. No T4/download/new data.
Freeze runner/protocol before execution. Preserve failure stage/audits.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth242_bf16_value_prior_pair.py --checkpoint results/native_expert_scaling/meth242_layer12_bf16_value_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth242_bf16_value_prior_result.json
```
