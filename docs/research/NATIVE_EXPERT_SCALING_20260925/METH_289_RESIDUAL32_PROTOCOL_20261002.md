# METH-289: explicit FP32 residual stream, unchanged285 whole smoke guards

## Prospective control card

Nearest cells: [287 trace](METH_287_LAYER_TRACE_RESULT_20261002.md) local
differences amplified through24-layer residuals; [288 norm64 failure](METH_288_NORM_REDUCTION_RESULT_20261002.md)
does not repair ablation maximum. Changed coordinate: execution precision
of the residual stream ONLY. Remove BF16 rounding after attention residual
addition and FFN residual addition. Keep original285 FP32 norm reduction,
BF16 normalization/weight products, BF16 qkv/o/attention/FFN returns,
BF16 source-plus-conditional addition and final BF16 norm/head exactly.
The running x vector may now contain FP32 values. This deliberately changes
the executed equation; it is not a claim of exact original BF16 arithmetic.

Fewer residual roundings may reduce amplification while storing identical
weights/conditional functions. Neither previous frozen equation measures
this target precision; benefit/quality are unknown. No norm64 adoption,
weight/route/alias/data/threshold fitting or new learned capacity. Same276
archive/725 fields/all76 helper hashes; same278 prompts0,1,8,9,16,17 entirely,
last8 states/48 per compact-core-only and complete-bank arm. New289 model
header must equal original285 except TWO literal residual replacements;
assay main must equal original285 exactly. New phase60 RESIDUAL32 personality
selects it; all original code/failures retained.

Reuse pinned original285 full-prefix GPU reference (no new GPU job). Pin
285/287/288 results and original285 header/C hashes; verify referenceSHA.
Original GPU loader record explicitly labeled reused; actual CPU725 loader
executes new precision recipe. Consumed diagnostics, not independent new
held-out/fresh-source evidence.

## Frozen gates, cost and decision

EXACT285 gates: same archive/all725/no fallback; finite all; EACH arm hidden
relativeL2 maximum<=.05, full-head logit relativeL2 maximum<=.05 and top1
agreement>=.95 across48 positions. All12 fresh CPU cache rebuilds BYTE-exact
to original sequential run under THIS recipe, all12 erased-history faults
detected. No relaxed gate/adaptive rows/variants. Failure stops residual32
before native quality/K64/rate. Pass only licenses full native donor-relative
prediction/generation/task/semantic/K64/rate under their own frozen gates.
It does not prove useful n/DRAM/LUT/family10B/100B or>=50 on same artifact.

Keep full error/top1/cache/provenance arrays. Expected CPU<2minutes after
imports; compilation/execution hard600s, outputs+reused reference<=160MiB.
Local six threads/no GPU or competing model job; no accepted-rate inference.
Source-only py_compile/phase60 compilation before freeze. Preserve unexpected
apparatus/resource failures; only narrow repair under a new freeze. Freeze
protocol/code before observations and preserve live handle/no duplicate job.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth289_residual32_qualification.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth289_residual32_qualification_result.json
```

Canonical result/terminal state and INDEX/METHOD update are required.
