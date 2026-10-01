# METH-226: source derivative and full-input affine native qualification

Freeze before execution. The previous goal turn made progress: METH-225
fixed nonlinear distillation rejects its accuracy gate and records actual
trained C parity/cost. The [changed proposal](CONDITIONAL_DONOR_TANGENT_PROPOSAL_20261001.md)
compiles source-local functions into a conditional bank without a trained
common. This experiment qualifies its apparatus/operator, not its accuracy.

## Fixed source and apparatus

Use the hash-bound original Qwen0.5B-Instruct BF16 safetensors and existing
METH-125 BF16 input fixture, no new source inference. For each24 layers,
use only that layer's predetermined token0 input as center. Decode the
complete gate/up/down matrices FP32, TF32 off, and compute the full896x896
analytic SwiGLU Jacobian from the proposal's formula. Before exporting
each layer, compare the entire Jacobian with vectorized autograd of the
direct PyTorch source-weight function. Require relative Frobenius<=1e-5,
maximum element difference/source-Jacobian peak<=1e-4, and all finite.

Store the Jacobian BF16 and bias FP32, `f(c)-J_stored@c`. Reconcile center
relative L2<=1e-5. Require all24 Jacobian hashes distinct. The little-endian
binary has24-byte `M226BF01` header (24,896,896,32), followed per layer by
896x896 BF16 weights and896 FP32 bias:38,621,208bytes. Read every segment
back exactly. These are24 distinct source-layer tangents, not learned
centroids/experts and not a claim of nonlinear reconstruction quality.

## Actual CPU operator and gate

AVX2 BF16 decoding, four8-lane FMA accumulators and FP32 reduction/bias
reuse the audited METH-224 algorithm. One full-input affine function per
layer. No activation quantization or BF16 activation intermediate. Six
threads,256 existing varied inputs through all24 distinct matrices,
three fixed passes; GPU work synchronized before CPU timing. Record
16x24 outputs and compare with the decoded stored-weight FP32 oracle:
median relative L2<=1e-4,worst<=5e-4. Median full24-layer component<=10ms.
No repeat timing/alternative kernel after observing outcomes.

A pass licenses freezing a full-input E16/E160 local function comparison
with complete-function error<=.01 and paired/rotated controls unchanged.
A derivative/storage/native failure stops this fixed apparatus. There is
no routing here, no large-bank DRAM/LUT ratio, trained causal composition,
independent LLM quality, generation/tasks or full accepted token rate.

Budget20min after imports,20GiB RSS,10.5GiB GPU,<50MB new outputs, local
RTX3060/six threads. No T4, new external data or branch donor-port resume.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth226_donor_tangent_cpu.c -o benchmarks/native_expert_scaling/meth226_donor_tangent_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth226_donor_tangent_feasibility.py --binary results/native_expert_scaling/meth226_donor_tangent_fixture.bin --exe benchmarks/native_expert_scaling/meth226_donor_tangent_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth226_donor_tangent_native_result.json
```
