# METH-291: causal operator-group localization on actual intervened inputs

## Prospective control card

Nearest cells:285 CPU full-prefix failure,286/287 small same-input errors
amplified through residuals,288/289 rejected precision variants,290 stable
no-cache causal GPU reference. Changed coordinate: causal operator-group
interventions, not another speculative precision variant. Same-input local
controls do not identify which group seeds/amplifies the full-model error.
Pin290 rawSHA `797b8ef913cea4b8eab2754ab37e66c0c75026854eec05a3c0ca9f8bfa2abbcb`,
285 rawSHA `c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda`.

Same276 archive/725 fields/all76 helper hashes/own GPU loader and C direct
loader. Source0/all147 tokens/tail8 compact-core-only ablation (not donor).
Actual fixed285 C/header hashes retained; no weights/routes/data/precision
changes to the substituted CPU operators. Four groups: input/post/final
norms; qkv/o/full-head projections; causal attention; source FFN BF16 return.
Residual adds/RoPE/embedding remain GPU BF16; previous same-input controls
found these exact, but closure below is required before causal interpretation.

Build a DLL with original285 primitives, expose batched-row wrappers.
Original scalar/AVX norm/mat/FFN equations are called directly. Attention
uses original current-F32-probability causal equation, not rejected286 variant.
Python replaces selected GPU module forwards/SDPA using actual current
inputs converted BF16->F32 CPU representation, C execution, then BF16 GPU
return. There is NO injection of original teacher hidden states. All groups
receive actual intervened inputs, including when a group is restored to GPU.

## Fixed arms, mandatory closure and reporting

1. Unmodified GPU original full prefix/tail8/M8 head must repeat original285
   hidden/full-head bytes exactly; otherwise apparatus stops.
2. `all_cpu`: all four groups CPU. Its tail8 hidden/full-head bytes MUST
   equal original285 actual CPU output exactly. If closure fails, save
   scientific closure-failure result/raw, stop before remaining arms and
   make NO causal group interpretation. A changed assay needs new freeze.
3. In fixed order restore ONE group to GPU: norms, projections, attention,
   source FFN. All others still original C on actual intervened inputs.
   These are nondeployable oracles, not native performance/quality candidates.

Retain all five arms'8 hidden/full-head rows, per-position errors/top1 IDs,
finite/median/max relativeL2/unequal-coordinate counts and callback counts.
Denominator floor1e-12. Each restored group's prospective indicator requires
at least50% reduction in BOTH hidden/logit median AND maximum versus allCPU.
Report ALL indicators, including failures; no adaptive group combinations,
row exclusions, threshold changes or parameter fitting. A positive indicator
motivates source/arithmetic repair investigation of that group; it does not
itself establish a deployable correction or explain another family/context.

Original285/288/289 smoke stops unchanged. No native quality/rank/rollout/
K64/rate acceptance measured or licensed by this assay. New deployable
equation needs a prospective record and unchanged285 gates before complete
donor-relative quality and accepted>=50 on SAME artifact. Useful huge-n/RAM/
DRAM/LUT/router and cross-family10B/100B still open.

## Resource and failure handling

Source-only py_compile/DLL compilation pass before freeze. LocalRTX3060/
six CPU threads, same BF16/SDPA/TF32off/deterministic/CUBLAS4096:8. GPU and
CPU cooperate inside THIS numerical assay, not a performance measurement;
no competing model job or CPU rate benchmark. Expected<4minutes, hard720s
after imports,20GiB RSS/10.5GiB CUDA/newNPZ<32MiB, compilation60s.
No download/T4. Retain failures; narrow binding/apparatus repair requires
new freeze without changing scientific choices. Keep live handle and never
duplicate a launch after observation timeout. All hashes/commands/terminal
cost/scope and canonical adjudication in this directory; update INDEX/METHOD.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth291_operator_surgery.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth291_operator_surgery_result.json
```
