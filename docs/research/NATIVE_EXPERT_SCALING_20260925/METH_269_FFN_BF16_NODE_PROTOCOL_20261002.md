# METH-269: isolate BF16 nodes,LUT and existing FFN representation

## Decision and reused evidence

268 source-route/score oracle recovers15/40 next-choice differences but
leaves25 and increases overall logit squared error. Same-input route and
selected archived BF16 values match exactly. Actual BF16 FFN error at
generated prefixes is larger than the old FP32-source component metric,
but both inputs and arithmetic changed; no causal comparison follows yet.
Before a changed core or route-robustness recipe, separate these effects
on *the same* fixed consumed component states.267 native stop remains.

This assay changes no stored parameter or fit. It is distinct from241's
local child/readout fit: no child function training or intercept retry.

## Fixed inputs and numerical arms

Bind268 repair1 raw SHA
`ed3d8289d5e729d59d4c545bff88ab0d294056d777e0b5689766ccb012fd1abe`,
same259 archive/export/helper hashes and original source/tokenizer config.
Use all6144 original125 BF16 states (256/layer), fixed file SHA
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
These are reused source component inputs, not new held-out quality or
the generated-prefix states used in268. No data-dependent filtering.

Reference: original source BF16 gate/up projections, exact original BF16
SiLU/product and BF16 down projection. Require bitwise equality of explicit
equation with the installed HuggingFace Qwen2MLP using exact source weights.
Require the current FFN equation to equal259 StoredFFN bitwise.

Freeze these eight diagnostic arms before observing any new component error:

1. Exact source weights with all internal FP32 operations and final BF16 cast.
2. Same FP32 source equation with existing513-entry SiLU LUT.
3. Actual BF16 source nodes with LUT substituted for SiLU, activation BF16.
4. Unchanged existing259 FP32-internal FFN equation/final BF16 cast.
5. Same259 encoded gate/up/private weights, round gate/up to BF16, LUT
   activation rounded BF16, BF16 feature product; unchanged fixed readout.
6. Same as5 with exact BF16 SiLU instead of LUT.
7. Oracle exact source BF16 feature vector with unchanged259 readout.
8. Oracle exact source FP32 feature vector with unchanged259 readout.

Roundings are explicit; no rank/scale/private-row changes. Oracle features
require original full source input matrices and are not an accepted compact
candidate. Measure every arm against actual BF16 source and,descriptively,
source FP32 output: total/normalized squared error,state/layer tails,and
different BF16 output values. No new acceptance threshold or quality pass:
the measurements select which mechanism merits the next prospectively
qualified representation, not post-hoc promotion of259 or a fit on these
consumed inputs. A node-only improvement still needs full independent
model/generation/task/semantic quality and actual CPU cost.

## Budget and command

Local RTX3060/six threads,deterministic/highest/TF32off,5min after imports,
20GiB RSS/10.5GiB GPU. Standalone layer/state equations; no whole model
generation,training,T4,download or concurrent CPU speed measurement.
Stop on reference/current-equation mismatch,nonfinite or resource breach.
Preserve per-layer partial/error records. Goal n/RAM/router-LUT/DRAM,
same-artifact accepted>=50tok/s,multiple real donor families/scales unchanged.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth269_ffn_bf16_node_assay.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth269_ffn_bf16_node_assay_result.json
```
