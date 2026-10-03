# METH-301 result: vector4 payload fits; unchanged native layout fails cost

**Decision: reject this fixed native vector4 implementation BEFORE palette
training or n640 allocation.** The complete descriptor cost fits the560MB
allotment, but the actual six-thread phase60 operator fixture measures three
stable medians **83.60765/82.27795/81.62630ms**, all above14ms. Numeric
controls pass. This closes this layout/fixture profile, not every vector
dictionary, source distribution or alternative implementation.

## Frozen source geometry and reproducible actual C

[Protocol](METH_301_VECTOR4_LUT_PREFLIGHT_PROTOCOL_20261003.md),
[controller](../../../benchmarks/native_expert_scaling/meth301_vector4_lut_preflight.py),
[native backend](../../../benchmarks/native_expert_scaling/meth301_vector4_lut_cpu.c)
and opt-in phase60 selector were committed at **`606b914` BEFORE observation**.
The exact protocol command exited0; all older source/helper bodies remain.
The new entry compiles actual `benchmarks/phase60/engine.c` with
`SILICON_VECTOR4_LUT_PREFLIGHT`, six OpenMP threads, AVX2 and no fast-math,
`-ffp-contract=off`. Compile diagnostics and exact command/compiler hash are
in the [raw result](meth301_vector4_lut_preflight_result.json).

Source geometry is pinned to300's414 real GigaChat GGUF descriptors and prior
source/header hashes. Every284 encoded projection and25 F32 router is in the
309-record/8668-byte catalogue. Real fixture banks are fully allocated and
touched in RAM:2,951,464,192B allocations,2,958,168,064B observed peak RSS.
They are deterministic synthetic codes/scales/palettes, NOT encoded source
weights or transferred knowledge. The 64 source expert SHAPES are real;
fixture parameter values cannot establish donor quality or useful n.

Raw23,270B SHA
`dc71c85a1d42058fa3eeac5cf83947c27a17e78f5f498b55caec069751aeef2f`.
Backend SHA `d07eedc895a86264ad610790df3123b214c93c8ceabe8c34c0aedfd15fd9c021`;
controller SHA `9fc3017ffff34674d3e5903b536445d4b43f3bfb45c8694fb69af62cf7d06f04`;
engine at freeze SHA
`7f8c8abfda58771980d88085f2d5b87439a193e4b52b643fea9b55b267740da4`;
binary SHA `f894c3fa7680c69b0abfd3da8291b849f7545e72ad12086f600ee9d559e13934`.
Local logs retain all30 times/hash observations and empty stderr; stdout SHA
`975928426a2bd93cb8b44718eb535140655091b86a0b17b0e90e131863e88232`.
Complete runtime7.484s, native child3.099s, initialization0.252s; all stops
pass. No GPU/T4/downloads/model inference/palette fit, no live job remains.
Post-run PE inspection records imports `libomp.dll`, `msvcrt.dll` and
`KERNEL32.dll`. The toolchain `bin/libomp.dll` used by the declared PATH is
1,262,080B, SHA `6fc163dd513538a92a187d987438bebc5509afd1f824b20a88dd51463b3d5698`;
no copy exists in the executable directory or workspace root. This is
reproduction provenance, not an in-process module-path trace.

## Descriptor arithmetic: necessary budget, insufficient implementation

| Quantity | Source-shaped value |
| --- | ---: |
| Complete addressed weight, n64 | 423,625,312B/token |
| U8 codes, all full projections | 406,405,120B/token |
| FP32 row scales | 5,840,384B/token |
| Shared256-entry 4D palettes | 1,163,264B |
| Unchanged Q4 embedding/F32 router/control bytes | 10,216,544B/token |
| Query table entries | 70,352,896/token |
| Query table writes, logical | 281,411,584B/token |
| FP32 table gathers, logical | 1,625,620,480B/token |
| Table construction multiply/add operations | 492,470,272/token |
| Encoded stored model payload | 2,770,853,632B |
| Analytical n640 complete addressed weight | 512,156,512B/token |
| Analytical n640 stored model payload | 24,328,978,432B |

Only the n64 native fixture was executed. The n640 rows are formulas, not
measured large-bank timing or trained extra capacity. Both descriptor gates
pass; shared tables/selected top4 alone do not remove flat-router growth.

Native code actually addresses **423,239,168 weight bytes/token**: the table
above also includes386,144B of omitted norms/biases/embedding row. Native
timing includes every309 matrix, all flat F32 router scans/stable top4,
per-query table construction, tiled AVX2 gathers, row scaling and dispatch.
Per-expert down inputs and all32 MLA K-B/V-B head inputs remain distinct.
It excludes source SwiGLU, normalization, bias/softmax/mixture, attention/KV,
RoPE, embedding and any causal model composition. Synthetic input preparation,
initialization, output hashes/logging are outside the measured operator body.
No physical DRAM counter/cache residency or accepted token rate was measured.

## Numeric controls and fixed decisions

| Frozen control/gate | Outcome |
| --- | --- |
|309 catalogue/75 routed/full geometry counts | PASS |
| Scalar versus AVX LUT rows, across every selected matrix bank | 16,968 byte-exact, PASS |
| FP64 independently decoded reference rows, including all router rows |18,568; pooled relative L2 2.3834171e-8<=1e-5, PASS |
| Deliberately wrong table value changes output | PASS; restored before timing |
| Full output/route hash sequence equal across3 repetitions | PASS |
| n64 complete weight<=560MB | PASS |
| n640 analytical complete weight<=560MB | PASS |
| Three n64 median max/min<=1.10 |1.02427343, PASS |
| ALL three n64 medians<=14ms |81.62630-83.60765ms, FAIL |

Each repetition uses the same10 fixtures:2 warmup and8 measured, all retained.
An independent Python recount reproduces each median, all30 stable hashes
and the primary decision. Actual selected expert-ID unions span26-32 of64
per layer over those10 inputs; no favorable reused-one-expert traffic model.
Banks exceed the CPU cache, but physical traffic/locality remains unmeasured.

Do not turn `1/0.082s` into accepted decode speed: these are input-ready
independent operators, not a composed language model. The scoped operator
cost alone fails its prospectively allocated budget by about5.9x, so the
frozen protocol correctly skips n640 and training. No gate/threshold changes.

## Next decision that needs evidence

Source preservation plus fewer coefficient indices does not automatically
make the target fast. Before choosing another implementation or structural
core transformation, attribute the SAME fixture's cost to router, table
construction and matvec by organ. Freeze read-only timing instrumentation,
retain301 as the original failure, and require identical outputs/routes.
This diagnostic should determine whether a layout/construction change could
address the observed bottleneck or a materially cheaper active core is needed.
It cannot regrade301, establish source quality or promote unchanged training.
No useful added n, same-artifact accepted50token/s or general transfer is proven.
