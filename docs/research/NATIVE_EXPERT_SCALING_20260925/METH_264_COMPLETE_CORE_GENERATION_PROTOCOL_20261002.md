# METH-264: frozen cached/full and independent generation health

Require all263 independent prediction gates, raw SHA
`7bcb8774bf335762e5e9b89dee29223bf1fa2bfea8d59480ed0621fa1c4d7821`,
same26124 sources/tokens and same259 complete artifact. Pin original
source/parent/child/tokenizer/helpers. No new learned function, candidate,
source selection or decoding adjustment. No model output was inspected
for this generation screen before code/protocol freeze.

Reuse unexecuted215 generation apparatus with259 loader and new bindings:
three arms original BF16 donor, original BF16 centered E1280, actual
saved259 source core/unique bank. Unpenalized greedy raw full BF16 tied
head,lowest ID ties,config EOS,at most128 new tokens. No temperature,
repetition penalty, prompt wrapper or exact-anchor enforcement introduced.

For each arm before its generation, first prompt in each of3 categories:
compare full-prefix versus cached next-token choices for16 steps, extending
the same prefix with full-reference choices. Record all choices and max
logit differences. Require zero mismatches; a failed cached-reference gate
holds generation/tasks/native promotion and diagnoses the cached arithmetic,
not automatically the transfer's prediction quality. No alternative sample
or widened mismatch threshold after observation.

Then generate all24 fixed prompts/arm with cache, recording raw continuation
IDs/text,EOS,early non-EOS termination<16 and repeated8gram3x/distinct2.
On every candidate generated state, full head still chooses actual token;
separately check fixedK64 proposal inclusion and exact-row reranking with
archived int8/FP16 proposal. Any miss/mismatch stops this fixed generated
shortlist path before task/native promotion. Preserve partial rows. This
finite generation check is not a universal shortlist certificate.

Once all72 continuations complete, compare candidate health against each
control: pooled EOS count no lower than control-2; early-non-EOS/repeated8gram
counts at most control+1 pooled and in every category. These are original215
health gates, not adjusted from new continuations. Health pass only licenses
separately frozen task and blind semantic assessment; health does not prove
factual/detail accuracy or source knowledge transfer. Failure closes this
fixed cached/health/shortlist variant, preserving other scoped evidence.

Local RTX3060/six threads,deterministic/highest/TF32 off,70min after imports,
20GiB RSS/10.5GiB GPU (cached/full3-arm autoregressive budget),no T4/download.
Record arm/document partial progress and failures. Source weights cached;
saved candidate loader has no fallback. No inference-speed benchmark or
CPU LUT/alias lookup/DRAM/real large-RAM n/other-family/10B/100B proof here.
Full same-artifact native>=50 accepted batch1tok/s remains required.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth264_complete_core_generation.py --prediction-sha 7bcb8774bf335762e5e9b89dee29223bf1fa2bfea8d59480ed0621fa1c4d7821 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth264_complete_core_generation_result.json
```
