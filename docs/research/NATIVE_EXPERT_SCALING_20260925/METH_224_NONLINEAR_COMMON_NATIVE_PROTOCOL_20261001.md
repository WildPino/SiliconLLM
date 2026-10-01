# METH-224: native nonlinear common before learning

Freeze before execution. METH-223 identifies inaccurate fixed affine
common fit and E160 fit gain/validation loss. A learned nonlinear
common is the next variable; confirm its proposed native precision/
shape cost before distillation. No post-hoc hard carve is promoted.

One fixed width768 SwiGLU common, width896,24 layers. Source fixture:
exact pinned BF16 Qwen0.5B-Instruct weights, first768 gate/up rows and
matching down columns at each layer. These are arbitrary, predetermined
real-weight subsets, not a quality or capacity selection. Add explicit
zero FP32 output biases to exercise future learned-bias arithmetic.
All 72 matrix bit sequences and24 control arrays must read back exactly.
Binary99,176,472 bytes includes99,090,432 matrix bytes,86,016 biases
and24-byte header. No neuron-importance optimization or source replacement.

Native AVX2 kernel decodes BF16 bits to FP32 on each load, four independent
eight-lane FMA accumulators per row, combines accumulators, reduces lanes.
Gate/up, SiLU, down and output bias all FP32. No input/hidden quantization
or BF16 intermediate rounding. Thus a trained BF16-weight common could
use this operator, but its actual trained artifact needs new parity/rate.

Use existing METH-125 real BF16 source states,256 tokens x24 layers x896.
First16 tokens/all24 layers must pass median relative L2<=1e-4 and
maximum<=5e-4 against FP32 GPU computation of the identical fixture
weights, TF32 off. Time three complete256-token native passes, six
threads, loading/oracle outside timing. Median24-layer common<=10 ms
licenses a frozen actual donor-output distillation protocol; otherwise
stop this fixed H768 native kernel before training. No kernel/width or
timing retry after observing the outcome. Do not claim full accepted
rate from a component pass or combine independent timings as a rate.

Hypothetical common plus METH-222-style conditional active payload is
329,811,968 bytes/token, replacing the failed affine common by H768.
This includes retained proposed head/attention/control, parent+selected
child keys and one equal-width folded active function. Actual trained
common, regularized descendants, full composition, DRAM, large-n CPU
LUT, independent LLM quality and multi-family/10B remain unverified.

Budget15 minutes after imports; local six CPU threads and RTX3060,
20GiB RSS,10.5GiB GPU,<110MB new output. No T4, training or new source/
generation/task inference; read source tensors solely for the fixture.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth224_bf16_common_cpu.c -o benchmarks/native_expert_scaling/meth224_bf16_common_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth224_common_native_feasibility.py --binary results/native_expert_scaling/meth224_bf16_common_fixture.bin --exe benchmarks/native_expert_scaling/meth224_bf16_common_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth224_common_native_feasibility_result.json
```
