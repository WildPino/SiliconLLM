# METH-287: propagated residual drift and remaining same-input operators

## Prospective control card

Nearest cell [286 same-input diagnosis](METH_286_SAME_INPUT_RESULT_20261002.md),
raw SHA `1c73b9fa032318f820de796b3e7743708846fcd4888bb414d4e75fd1750202bc`;
original [285 whole failure](METH_285_WHOLE_PREFIX_RESULT_20261002.md).
Changed coordinate: whole-prefix per-layer residual propagation and isolated
o projection/post norm/source FFN/addition estimands.286 does not measure
those or propagated layer drift. Same276 archive/725 fields, same278 source0
all147 tokens/all24 layers, compact-core-only arm (not donor). No new
weights/routes, tuning, data selection or arithmetic modification.

Trace original285 forward with five memcpy observations per layer:
input x, post-attention residual, post norm, source FFN FP32, final residual.
Generated trace function must equal original285 byte for byte after removing
the five diagnostic copies and renaming the function. Source-only proof and
compilation pass before freeze. Actual final CPU hidden/full-head bytes must
equal saved285 first-source/first-arm final row; otherwise apparatus stops.
GPU own loader/no cache/hooks must reproduce saved285 first-source tail8
hidden exactly. GPU FP32 source FFN recomputation rounded BF16 must equal
actual hooked base output; otherwise apparatus stops.

Isolate original CPU o projection on GPU attention input, post norm on GPU
post-attention residual, source row_forward on GPU post norm. Compare CPU
source FP32 and BF16-rounded output separately. On GPU operands test the
two original BF16 residual additions against actual hooked GPU residuals.
These oracles are diagnostic only and cannot serve as a deployable artifact.

## Reporting, interpretation and stops

Report every propagated and isolated field over all3528 states: finite,
median/max relativeL2, per-layer median/max and unequal-coordinate count,
denominator floor1e-12. No adaptive exclusions/thresholds/variants. Pin
original285 CPU/GPU raw hashes and archive/helper/local-source provenance.
Retain all raw trace/reference/fixture/isolated bytes and hashes. The result
localizes drift; it does not by itself establish a causal repair. Changed
execution needs a subsequent frozen recipe and unchanged285 smoke limits.
Original285 failure and286 rejected BF16 probability variant stay preserved.

Quality/rank/rollout/rate acceptance is not measured; fixed native recipe
still stopped. Same-artifact donor-quality/>=50, useful new large-n capacity,
actual CPU routing/LUT/DRAM and cross-family10B/100B remain open.
Expected GPU<2minutes then CPU<3minutes; hard GPU720s after imports,
20GiB RSS/10.5GiB CUDA, CPU compilation/execution180s after GPU completion/
synchronization. Raw fixture/native/trace/reference<=384MiB. LocalRTX3060/
six threads,no download/T4/no CPU rate overlap or timing interpretation.
Preserve binding/resource/apparatus failures; narrow repair requires new
freeze before rerun without altering scientific choices or limits.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth287_layer_trace.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth287_layer_trace_result.json
```

Freeze code/protocol before observations; preserve live handle/no duplicate
launch on observation timeout. Canonical result in this directory must
record terminal status and update INDEX/METHOD.
