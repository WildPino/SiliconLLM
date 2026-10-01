# METH-220: localize W8A8 parity and price a float-input kernel

Freeze before execution. METH-219 stops on worst-case emulator relative
L2 0.006070 vs <=0.0005 despite median 4.095e-7; no full-model arm ran.
Its exported METH-211 FFN bytes AND native outputs exactly match METH-182.
That exact W8A8 kernel is already cost-rejected (12.829 ms vs <=10).
Do not retime it as a new feasible path or relax the emulator gate.

Compile the taped C extension including unchanged METH-182 source with
its documented flags: `clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11
-Wall -Wextra`, link `-lm -lpsapi`. On first 16 METH-125 states at all
24 layers, record input codes/scales, FP32 gate/up/hidden, hidden
codes/scales and output. Require all 384 taped W8A8 outputs to equal
METH-219 existing C outputs bit for bit before diagnosis; otherwise
stop as apparatus failure.

Compare Torch FP32 W8A8 arithmetic at each stage, with TF32 off.
Hidden-boundary diagnosis passes only if all input codes and scales
match exactly, every output row above the existing 0.0005 limit has
changed hidden codes within 0.001 of a half-integer rounding boundary,
and replacing the emulator hidden codes/scales by taped C values
makes all output errors <=0.0005. Record every token/layer independently.
This causal stage replacement concerns this consumed component fixture;
it does not validate full-model emulator quality or universal safety.

The fixed alternative changes only the compute kernel: the SAME Q8
FFN bytes, decoded FP16 scales, FP32 inputs/hidden and output. AVX2
sign extends eight weights, accumulates eight FP32 FMA lanes per group,
then applies group scale; no input/hidden quantization or BF16 rounding.
On the same 384 states require median relative L2 <=1e-4 and worst
<=5e-4 versus decoded-weight FP32 GPU gate/up/SiLU/down. Time three
full 256-token passes through all 24 FFNs with six threads, loading
outside timing. Reuse the frozen METH-182 feasibility budget <=10 ms
median for this new float-input kernel, leaving 10 ms for all other
organs at 50 tokens/s. No kernel/layout/reduction retry after outcome.

Both float numerical and cost pass license a new W8A32 quality protocol;
otherwise stop this specific float-input native kernel/core path. A
failure does not exclude all grouped-Q8 kernels, other representations
or trained transfer. W8A8 quality remains unmeasured and its existing
cost path rejected. Preserve large-n route/function and RAM/LUT goal.

Budget: local RTX 3060 and six CPU threads, <=15 minutes after imports,
20 GiB RSS, 10.5 GiB GPU; <50 MB additional output. No T4, fresh sources,
training, generation, tasks or B/native routing promotion.

Commands from repository root:

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth220_ffn_intermediates_cpu.c -o benchmarks/native_expert_scaling/meth220_ffn_intermediates_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth220_ffn_native_diagnostic.py --exe benchmarks/native_expert_scaling/meth220_ffn_intermediates_cpu.exe --tape results/native_expert_scaling/meth220_ffn_intermediates.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth220_ffn_native_diagnostic_result.json
```
