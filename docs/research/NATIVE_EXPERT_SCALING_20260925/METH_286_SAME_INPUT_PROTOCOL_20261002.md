# METH-286: isolate frozen285 numeric operators on identical GPU inputs

## Prospective question and control card

Nearest cell: [285 whole prefix failure](METH_285_WHOLE_PREFIX_RESULT_20261002.md),
raw SHA `c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda`.
Changed coordinate: diagnostic estimand, isolated operations supplied the
actual GPU inputs rather than accumulated CPU prefix states.285 cannot
identify which local operator introduces error. Same276 archive/725 fields,
unchanged285 header/source hashes, same278 manifest, consumed source0,
all147 tokens/all24 layers, compact-core-only arm (not original donor).
No fitting, weights, route, model configuration or source selection changes.

Capture norm input/output, actual BF16 q/k/v projection output, actual
post-RoPE SDPA q/k/v and actual pre-o_proj attention. Intercept SDPA without
changing its arguments/backend; require causal/maskNone/dropout0/scale.125,
14 query heads/two grouped KV heads (or exact repeated heads). Require SDPA
output reshaped BYTE-exact to actual o_proj input and v exact to projection.
CPU-activity profiler records actual ATen SDPA backend events; no assumption
that Flash is selected. Pin installed Qwen/SDPA source hashes in script.

CPU includes original285 model header. On SAME isolated inputs compare:

- Input RMSNorm via original cc_norm.
- q/k/v projections via original cc_mat using actual GPU norm input.
- RoPE via original cc_rope and original CPU frequency/cos/sin, using actual
  GPU pre-RoPE q/k rather than CPU projection outputs.
- Original causal F32 normalized-probability attention, using actual GPU
  post-RoPE q/k/v; exact equation from frozen285.
- One prospective diagnostic alternative: BF16 round normalized softmax
  probabilities before their F32 products/accumulation and BF16 output.
  This boundary comes from installed Qwen eager_attention_forward, not a
  fitted parameter. It does not establish equality to actual fused SDPA.
- GPU forced MATH SDPA on the SAME captured q/k/v/kwargs. This is an oracle
  numerical control, not a changed full-model reference or deployable model.

Torch documents fused SDPA numeric differences and F32 math-backend
intermediates for BF16/Half ([official2.6 docs](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)).
This motivates backend observation, not an attribution of285 failure.

## Frozen reporting/decision and stops

Report finite outputs, per-layer median/maximum relativeL2, overall
median/maximum and unequal coordinates for EVERY tested operation, all3528
states. Retain raw fixtures/reference/native outputs/hashes. Denominator
floor1e-12 fixed. Norm/projection/RoPE controls locate local drift; they do
not by themselves explain accumulated whole-model5.4224% error.
Record whether BF16 probability variant halves BOTH median and maximum
relativeL2 against actual default attention. This fixed indicator motivates
a subsequent execution recipe only if satisfied; it does not reopen285 or
license a native quality/rate run. If unsatisfied, preserve result and
localize further only as motivated by measured errors/backend. No adaptive
variants, post-hoc row exclusions or threshold changes in this experiment.

This is diagnostic only: no quality/rank/rollout/rate acceptance gates,
no increase in useful learned capacity. Original285 fixed recipe remains
stopped. A corrected full execution must be prospectively frozen and face
the unchanged285 smoke gates before native quality/K64/rate. Same-artifact
>=50/useful RAM-scale n/DRAM/LUT and cross-family10B/100B remain open.

Expected GPU1-2minutes then CPU<3minutes; hard GPU720s after imports,
20GiB RSS/10.5GiB CUDA, CPU compilation/execution180s. GPU synchronization
and completion precede CPU execution; no CPU rate interpretation. Total
fixture/native/reference<=256MiB, localRTX3060/six threads, no download/T4.
Source-only py_compile/C compilation succeeded before freeze/observations.
Unexpected binding/resource/apparatus error is preserved; narrow repair
requires a new freeze before rerun, without changing scientific choices.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth286_same_input_diagnosis.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth286_same_input_diagnosis_result.json
```

Freeze this protocol/code before observations. Preserve live handle; no
duplicate launches following an observation timeout. Record terminal result
and adjudication in this directory and update INDEX/METHOD.
