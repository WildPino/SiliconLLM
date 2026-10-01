# METH-221: packed-Q6 native cost before conditional recovery

Freeze before execution. METH-220 grouped-Q8 float-input C is numerically
accurate but misses the <=10 ms FFN budget at 12.540 ms. METH-186 stores
actual signed Q6 FFN weights; direct BF16 quality fails METH-187 and
global low-rank corrections fail METH-190/191/192. This experiment does
not repeat those quality recipes. It asks whether a cheaper active core
can be a feasible base for a **changed conditional pretrained-function
recovery**; native cost is currently unmeasured.

Use the exact saved METH-186 core and its bound export report, unchanged
72 FFN Q6 codes and half group64 scales. Repack each 64 unsigned+32
codes into 32 low-nibble bytes plus 16 high-two-bit bytes. The layout
stays 48 bytes/group; no weight/scale value changes. Independently
decode and compare every source code, then read back every exported
segment and hash. Native binary: 245,145,624 bytes, including header.
Native SIMD codec checks all 64 unsigned values in every lane across
64 deterministic mixed patterns before reading model weights.

One fixed AVX2/FMA C kernel: decode 32 signed codes at a time; sign
extend into FP32, eight FMA lanes per 64 input group, reduce lanes,
apply FP16-decoded group scale, sum groups. Gate/up, SiLU and down all
FP32; no activation quantization or BF16 weight rounding. Only fixture
inputs decode from BF16. Do not add kernel/precision/layout arms after
observing the result.

Reuse exact existing METH-125 256x24x896 BF16 states. They are real
pre-MLP source-model states, not a Q6 trace. At the first 16 tokens and
all 24 layers require median relative L2 <=1e-4, maximum <=5e-4 versus
unrounded decoded-Q6-weight FP32 GPU gate/up/SiLU/down, TF32 disabled.
Use six local CPU threads; time three complete 256-token passes,
excluding loading and GPU checking. Feasibility: median <=10 ms per
24-layer FFN token, as already frozen in METH-182/220. No timing retry.

Both gates pass: price whole-core plus route/function recovery and
freeze a bounded activation-output specialist pilot. Either fail: stop
this particular Q6 kernel before any recovery training or new quality.
Do not promote the uncorrected Q6 model. Other Q6 kernels are not
generally refuted. A later specialist must learn actual donor error on
fit training states and improve on separate validation/source states;
extra weak descendants of an accurate dense core are not useful capacity.

Hypothetical whole selected-weight ledger combining METH-211 exact
head/proposal/control with METH-186 FFNs: 559,794,176 - 78,446,592 =
**481,347,584 bytes/token**, leaving 78,652,416 under 560 MB. This
combination is not exported or quality-valid. A separate 4-active rank8
BF16 recovery A/B bank would add 2,752,512 bytes/token; illustrative
selected local keys add 114,048, total 484,214,144. RAM bank size, actual
DRAM lines, activation traffic and routing must still be measured. No
failed METH-218 route is promoted. Large-n useful capacity and real
approximately-10B/multiple-family transfer remain part of the goal.

Budget: <=15 minutes after imports; local six CPU threads and RTX 3060,
20 GiB RSS, 10.5 GiB GPU, <300 MB new output. No T4, fresh sources,
generation, tasks or model training.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth221_q6_planar_cpu.c -o benchmarks/native_expert_scaling/meth221_q6_planar_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth221_q6_native_feasibility.py --binary results/native_expert_scaling/meth221_q6_planar_ffn.bin --exe benchmarks/native_expert_scaling/meth221_q6_planar_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth221_q6_native_feasibility_result.json
```
