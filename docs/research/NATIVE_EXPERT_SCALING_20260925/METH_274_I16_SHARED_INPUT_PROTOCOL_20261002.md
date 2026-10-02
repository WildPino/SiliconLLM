# METH-274: prospective I16 input/Q8 shared projection operator

## Decision and changed variable

273 diagnoses actual private128 shared/down-right phases at92.14% of summed
phase medians.271's output-aware source coverage passes fidelity but its
kernel costs>10ms;272's FP32 input-dot fusion does not improve that cost.
Test dynamic I16 inputs only for shared gate/up Q8 projections. No count,
selector,source/private rows,stored weights,LUT,down escapes,rank32/bias,
conditional bank/router/map changes. This is an approximate input operator,
not an exact scheduling transformation or new expert-n result.

Bind271 raw SHA
`638594a650189ab45efa62c08be04a3e42b21f7eff00c52d372d5991077120c3`,
273 raw SHA
`d8918584b3c7d9ac323347e979600a15099559cacc3016c92c132f1144225643`,
unchanged271 fixture337,596,452bytes/all361 segments,269 original BF16
baseline/source/archive/helper hashes,and all125256 states/layer.
Code must be implemented and committed before274 observation; no274 code,
executable or result exists at this protocol freeze.

## Fixed arithmetic

Decode original BF16 x to FP32. For each state choose FP32
`inverse=32767.0f/max(abs(x))`, `scale=1.0f/inverse`; for zero input use
inverse=scale=1. Quantize `round_ties_even_FP32(x*inverse)` to signed I16,
clip[-32767,32767]. Retain original x for all128 private BF16 input rows.
Shared gate/up each compute an exact integer dot with their stored signed
Q8 row,then `FP32(FP32(integer_sum)*scale)*stored_FP32_row_scale`.
LUT/product and every later stage stay unchanged.

Use AVX2 signed I16/Q8 products: sign-extend16 Q8 weights,`madd_epi16` into
eight I32 lanes,accumulate without saturation,then sum lanes in I64. D896
and max codes127/input32767 bound each lane well below I32 overflow; assert
the bound. No saturating `maddubs`,integer down projection,fused kernels,
activation8-bit or alternate scales/precision sweep. All rows/ranks fixed.

GPU oracle uses the same I16 quantizer,FP32 dot of integer-valued codes,
then same scale/row-scale order; record its rounding limitation. On CPU
check all6144 shared gate/up projections against a scalar exact I64 oracle
using identical I16 codes and FP32 scaling before interpreting output/times.
Require zero integer-dot/quantizer discrepancies; account setup cost in
component timing. Also verify all384 final native outputs versus GPU oracle
with original medianrelativeL2<=1e-4/max<=5e-4. GPU reference alone is not an
integer exactness certificate.

## Source/cost gates and stop

Require all immutable fields/source/old32 subset/fixture guards and finite
outputs. All6144 actual source-BF16 controls must meet the original271
mean<=.85*.0001972897928984215 and maximum-layerenergy error
<=.00030542892636731267; odd-reserve mean<=.85 its paired original32 and
maximum-layer energy<=paired original32. Selection remains271's saved IDs,
no fitting/reserve tuning. Report drift versus unchanged271 separately.

Only after source fidelity passes run original three256-token24-layer
six-thread component sweeps,including dynamic input setup. No GPU/model
work overlaps timing; explicit synchronization. Require median<=10ms,
original numeric limits and integer exactness guards,not a best pass or
relative/historical timing substitute. Preserve failures; no changed gate,
source filtering,count/scale alternative or repeat to rescue the recipe.

Local3060/sixthreads10min source/oracle scoring,20GiB RSS/10.5GiB GPU,1GiB
new output; native compile/check/timing separately5min,timeout180s,no T4/
download. Pass licenses separately frozen full model with this actual
arithmetic plus consumed regression then fresh excluded-source generation/
task/anonymous review.259 remains closed by267; all earlier cost stops
stand. Native cache/full quality/accepted>=50,useful learned RAM-scale n,
route/LUT/real DRAM cost and family/10B/100B transfer remain mandatory.

## Apparatus freeze and execution order

`meth274_i16_shared_input.py` reads the unchanged271 fixture,verifies all
segments/source rows and repeats old269/271 controls exactly. GPU reference
parameters use CPU IEEE FP32 divisions followed by GPU multiplication/round
so the reference does not silently substitute CUDA reciprocal arithmetic.
Export all5,505,024 GPU I16 codes and12,288 FP32 inverse/scale parameters in
an11,059,220-byte check fixture; these are validation data,not deployable
cached inputs. Actual C recomputes them from x inside each forward/timing.

The C qualifier checks all59,768,832 gate/up integer dots against a scalar
I64 expression (compiler may vectorize it),plus CPU/GPU codes/scales and
zero/ties/max-positive-negative I64 reduction edge guards. It then emits
ALL6144 final FP32 output vectors. Before timing,verify actual CPU BF16-
cast source/reserve fidelity against the same gates,all6144 GPU/native
errors under the original numeric limits (including original384 separately).
This additional full CPU component check strengthens the apparatus without
changing source/selector/gates. Timing mode only runs after all qualification
gates and repeats the original384 check before three sweeps; outputs must
be bitwise equal the qualifier's first384. Native modes run after GPU sync,
combined compiler/qualifier/timing<=5min,individual timeout<=180s. Qualification
cost is conversion/check overhead and is reported separately from rate.

No274 observation at this code/protocol freeze. Source and dependencies,
quantizer/native outputs,all per-layer/source/numeric/command/resource hashes
and failures are preserved. Frozen command:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth274_i16_shared_input.py --exe benchmarks/native_expert_scaling/meth274_i16_shared_input_cpu.exe --quantizer results/native_expert_scaling/meth274_i16_quantizer.bin --check results/native_expert_scaling/meth274_i16_shared_input.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth274_i16_shared_input_result.json
```
