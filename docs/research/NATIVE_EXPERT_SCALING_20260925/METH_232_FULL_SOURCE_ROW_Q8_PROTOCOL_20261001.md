# METH-232: frozen full-source row-Q8 native prerequisite

## Uncertainty and changed variable

Can all original nonlinear source features fit the active-byte ledger and
the <=10ms native FFN component budget with a row-scaled int8/FP32 operator,
while conserving the source function on existing component states?
METH-229's H2048 combined affine/BF16 operator passed cost, but METH-231's
trained conditional function failed quality. That does not isolate feature
coverage as the cause. This variant retains all4864 source units, removes
the full affine term and changes grouped/BF16 weight storage to row-Q8.
It is a prerequisite, not a fitted expert-count or end-to-end quality test.

Reuse pinned Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, original BF16 source weights
SHA256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
all24 distinct source FFNs and METH-125 vectors SHA256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
No source selection, fitting, validation-window access or external download.

## Codec, arithmetic and binary binding

For every original gate/up/down row, encode symmetric signed int8 codes:
scale=maxabs/127 in FP32; exactly-zero rows use scale1. Divide FP32 original
decoded-BF16 weights by that scale, round ties to even, clamp[-127,127].
Preserve all original rows/columns. Bias is896 FP32 zeros per layer.
Check zero rows, positive/negative ties, positivity/finite scales and no-128.

Header `<8s4I>` = `M232RQ01`,24,896,4864,32. Per layer: gate codes
(4864,896),gate scales(4864);up codes/scales same;down codes(896,4864),
down scales(896);bias(896). Little-endian, contiguous row-major. Exactly
314,892,312bytes including24-byte header;168 separately hashed/read-back
segments. Require all24 code matrices distinct for each organ.

Actual matvec: FP32 sum of `int8_code_as_float * x`, then multiply once by
the FP32 row scale. GPU oracle `F.linear(x,q.float()) * scale`, not a
predecoded scaled-weight GEMM. FP32 SiLU(gate)*up, same output matvec+bias;
no input/hidden activation quantization. C uses AVX2 signed8-to32-toFP32
conversion, four8-lane FMA accumulators per32 columns, fixed FP32 reduction
and row-scale-after-reduction. No tail for the divisible-by32 source shapes.
The common METH-182 file supplies audited I/O/timer/BF16-state decoding.

## Fixed gates and decision

1. All bindings, finite outputs, codec edges,168 segment reads and distinct
   layer hashes pass.
2. First16 states per layer:384 C/GPU output comparisons, median relative
   L2<=1e-4 and maximum<=5e-4 against the declared row-scaled oracle.
3. All256 existing states per layer: pooled FP64 SSE/energy<=.01 against
   the original BF16-weight/FP32 FFN. Report all24 layer errors; no separate
   layer-error threshold. This is not original BF16 intermediate rounding
   or a whole-model prediction/held-out source screen.
4. Six CPU threads,256 tokens across24 distinct layers, three fixed timing
   passes after the16-state output check. Median24-layer component<=10ms.
   All three retained, no retiming or adjusted threshold.

All pass licenses a separately frozen full-feature conditional output
method; a failed gate stops this fixed codec/operator before readout fitting.
No complete model, E16/E160 quality, routing/LUT or large-RAM DRAM rate is
claimed. The proposed542.050MB whole ideal selected ledger and E16/E160
residency are shape arithmetic in the [proposal](FULL_SOURCE_ROW_Q8_PROPOSAL_20261001.md).
GigaChat's sparse FFN/top4/shared operator and wider shapes need their own
priced variant; the frozen donor-adaptation evidence remains reusable.

## Local budget and commands

Local RTX3060; six host threads. Maximum20min after imports,20GiB RSS,
10.5GiB GPU peak,2GiB free disk before launch, <400MB fixture plus check and
small report. Native subprocess timeout600s, one execution. No T4.
Resource/implementation exception records a failure with stage and completed
segments. Source-function failure is a measured result, not an exception.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth232_full_source_row_q8_cpu.c -o benchmarks/native_expert_scaling/meth232_full_source_row_q8_cpu.exe -lm -lpsapi
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth232_full_source_row_q8_feasibility.py --binary results/native_expert_scaling/meth232_full_source_row_q8_fixture.bin --exe benchmarks/native_expert_scaling/meth232_full_source_row_q8_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth232_full_source_row_q8_native_result.json
```

Freeze protocol and sources in Git before substantive execution. Result
records source/executable/script/binary/output hashes and actual resource
cost; binaries/weights stay local and ignored. Pin result JSON CRLF in
attributes so the report byte hash survives checkout settings.
