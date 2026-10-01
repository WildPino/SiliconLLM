# METH-225: fixed learned nonlinear common on real pretrained outputs

Freeze before fitting/validation. METH-224 H768 BF16/FP32 operator passes
numerical/native cost. METH-223's fixed affine common fits poorly, with
negligible coefficient-rounding effect. Learn a nonlinear shared function
before regularized conditional descendants; do not repeat affine leaves,
global weight-SVD correction or a post-hoc hard-carve quality gate.

## Fixed data, representation and training

Exact saved METH-222 layer12 source states,512 fit raw windows and128
separate validation windows,128 tokens each. This reused validation is
consumed development data, not independent/document-held-out quality.
No new source inference or candidate/source replacement. Actual teacher
target is the full original BF16 donor FFN output.

Common: SwiGLU gate/up896->768, down768->896 and output bias896.
Store all weights BF16, bias FP32; all forward activations/control FP32,
matching the METH-224 native operator. FP32 training masters use BF16
rounding in forward with straight-through gradients. Initialize gate/up
from predetermined first768 source rows; fit down once by centered FP64
ridge on resulting fit hidden activations, lambda=.001 mean covariance
diagonal. Round down BF16 and compute bias from rounded weights/fit means.
This is a nonlinear feature initialization; every matrix remains trainable.

Fixed4096 AdamW updates,batch256,seed225225,16 complete fit-state epochs
(1,048,576 exposures;65,536 unique correlated states). Fresh seeded
without-replacement permutation each epoch. Adam betas(.9,.95),eps1e-8,
weight decay0,gradient norm clip1. LR3e-4,128-step linear warmup times
cosine decay to3e-5. Loss=mean squared output error divided by fixed
fit teacher mean squared output. Enable deterministic CUDA algorithms,
CUBLAS4096:8,TF32 off. Stop nonfinite/budget; no schedule/width/optimizer
retry after outcome. Save only initialized and fixed final controls;
no best-checkpoint selection or intermediate validation. Read every
stored tensor back and require final decoded forward exact.

## Intermediate decision, without changing final thresholds

Evaluate initialized and final stored models once after training on all
128 consumed validation windows; reconcile each teacher output energy.
The common alone is an **intermediate** for conditional recovery:

- final output SSE divided by teacher output energy<=0.10;
- final SSE<=90% initialized nonlinear-control SSE;
- 10,000 paired-window bootstrap normalized gain P05>0,seed225226;
- exact tensor readbacks and effective-forward parity;
- actual trained-weight native numerical/cost checks below.

This does **not** relax METH-222's <=0.01 complete conditional-function
screen or any independent LLM BPB/top1/generation/task gate. A passing
common licenses freezing parent-anchored/shrunk child recovery with one
folded equal-width active coefficient at E16/E160. It is not accepted
quality, useful increased n, complete compact knowledge or a final model.
A failure stops this fixed training recipe before descendants.

## Bind trained values to native arithmetic

Copy the hash-bound METH-224 full99MB fixture to a new file. Replace
only layer12's three matrices and output bias with final stored values;
verify every changed segment and every unchanged other23-layer segment.
Use the same executable and existing METH-125 states. Require all
other23-layer C output bits equal to the old fixture and layer12's16
outputs to match FP32 decoded trained weights within median1e-4/worst5e-4
relative L2. Three full mixed-fixture passes with six threads must have
median<=10ms. GPU training is complete/synchronized before CPU timing.
The mixed component includes23 untrained source fixtures; it is neither
a trained24-layer artifact nor full accepted throughput. No timing retry.

Budget25 minutes after imports; local RTX3060/six host threads,20GiB RSS,
10.5GiB GPU,<120MB additional outputs. No T4, new external data, source
generation or tasks. Large-RAM useful n and CPU LUT/DRAM, full causal
composition and real multiple-family/approximately10B transfer remain open.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth225_nonlinear_common_distillation.py --initial results/native_expert_scaling/meth225_layer12_common_initial.safetensors --checkpoint results/native_expert_scaling/meth225_layer12_common_final.safetensors --binary results/native_expert_scaling/meth225_trained_layer12_mixed_fixture.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth225_nonlinear_common_distillation_result.json
```
