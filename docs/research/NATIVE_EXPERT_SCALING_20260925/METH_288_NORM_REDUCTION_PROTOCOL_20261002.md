# METH-288: one changed norm accumulation, unchanged whole285 gates

## Prospective control card

Nearest cell [287 trace](METH_287_LAYER_TRACE_RESULT_20261002.md) identifies
small local norm/projection/FFN differences amplified in the residual stream;
post norm isolated maximum .0020082. CPU original norms sum896 squares
sequentially in FP32.285 fails compact-core-only full logit5% guard.
Changed coordinate: RMSNorm sum of squares ONLY accumulates in F64, then
casts the mean to F32 before the original F32 inverse/epsilon and BF16
normalization/weight multiplication boundaries. Accurate sum may reduce
rounding-boundary drift; this is a hypothesis, not a known causal repair.

Same original276 archive/725 fields, all76 pinned helper hashes, same278
entire prompts0,1,8,9,16,17, last8 states per compact-core-only/complete-bank
arm. No weight/route/alias/FFN/matmul/RoPE/attention/head change. New288
header must equal frozen285 header except ONE literal norm-line replacement;
new288 assay main must equal285 main exactly. Bind original285 C/header
hashes and287 result hash. New phase60 NORM64 personality selects this
recipe; all historical recipes/files/results preserved.

Reuse original285 GPU reference NPZ by its full pinned SHA, verified original
result SHA `c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda`.
No new GPU execution/reference/data selection. Original GPU loader record
is explicitly reused and distinguished from current actual CPU loader.
Reusing consumed numerical controls tests the changed execution, not new
independent donor-quality/generalization evidence.

## Unchanged gates and cost/stop

EXACT285 smoke guards: same725/archive/no fallback; all finite; EACH arm
maximum hidden relativeL2<=.05 AND maximum full-logit relativeL2<=.05 AND
top1 agreement>=.95 across48 states. All12 independently rebuilt CPU cache
finals must be BYTE-exact to sequential; all12 erased-history faults must
be detected. No count/limit loosening or post-result tuning. Positive
failure stops this norm64 recipe before quality/K64/rate. Pass only licenses
full native donor-relative prediction/generation/task/semantic/K64/rate
qualification, never the final goal by itself.

Keep full errors/top1/cache flags/loader/file/source/executable hashes.
Expected CPU<2minutes plus imports; hard compile/execution600s, output plus
reused reference<=160MiB, local six threads/no GPU or competing model job.
Source-only py_compile and phase60 compilation pass before freeze. No
accepted-rate interpretation. Apparatus errors retained, narrow repair
requires new freeze. No new useful capacity, DRAM/LUT/routing scale or
family10B/100B claim; original285/259/264/274-275 stops stay preserved.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth288_norm_reduction_qualification.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth288_norm_reduction_qualification_result.json
```

Freeze code/protocol BEFORE observations. Preserve handle/no duplicate
launch on observation timeout; write canonical adjudication and INDEX/METHOD
when terminal. Six prompts are consumed diagnostics, not a new holdout.
