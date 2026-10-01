# METH-238: frozen fixed32 BF16 output escapes plus row-Q8 native prerequisite

## Question and changed representation

Can a small higher-precision output subset lower quantization range while
remaining within the native FFN budget? METH-234's full source-feature
row-Q8/LUT operator passes8.522ms/.01944% scoped source error, but METH-235/
237 actual projected int8 readouts fail the1% stored-derivative gate;
METH-236 excludes scale-only repair. Change output precision/codec,not
iteration count or threshold. No derivative-prior pass is assumed from
this original-source component fixture.

Bind original pinned Qwen0.5B weights/revision,all24 FFNs,METH-125 vectors,
METH-234 result SHA256
`972a451f89fdbcc567b6f1f52df0e22bd2185f28d52cf3179b560b8ba8fba815`.
All121 source gate/up/table/bias segments must remain byte-identical to
METH-234. No training/source selection/validation/new quality/source download.

## Fixed codec, layout and actual arithmetic

For each output row,stable descending absolute weight order,lower original
feature ID first at ties; select exactly32 columns. Preserve their BF16
values and uint16 IDs. Set those coefficients to zero in the remainder,
then symmetric row-max/127 int8 codes with FP32 scale; zero remainder scale1,
ties-to-even,[-127,127]. No activation quantization. Original-source escape
values are exact BF16; future projected/learned FP32 values would round.
Meaningful edges:zero row,all ties,one large outlier,unique valid IDs and
zero codes at selected columns. Verify every actual row against the source
selection rule and selected values,not just metadata.

Binary `<8s4I>`:`M238ES01`,24,896,4864,32;513 FP32 shared lookup samples.
Per layer:gate codes/scales,up codes/scales,down remainder codes/scales,
down IDs(896,32)uint16,down escapes(896,32)BF16,bias(896)FP32. Exactly
317,646,876bytes;217 hashed/read-back segments. Dense code body stays
full-size with selected codes zero; exception bytes are additional.
All24 code matrices distinct for each organ. C loader validates IDs,
uniqueness,zeroed positions,scales and finite BF16 escapes before timing.

Gate/up/SiLU LUT arithmetic remains METH-234. Output row:
`(row_scale * dot(q_as_float,hidden) + escape_dot) + bias`.
Dense dot uses existing four8-lane FP32 FMA accumulators/32 columns and
row-scale-after-reduction. Escape dot:four8-lane BF16-decoded products,
AVX2 gather using widened uint16 IDs,reduce `(a+b)+(c+d)` then scalar lane
sum. Add to scaled dense dot in the same output-row OpenMP pass. This
explicitly prices indexed access; no ideal locality assumption.

GPU oracle separately evaluates `F.linear(hidden,q.float())*scale` plus
indexed FP32 decoded-BF1632-product sum and bias. It must not replace this
with a predecoded combined-weight GEMM as if bit-equivalent. C/GPU reduction
differences must remain within frozen numeric tolerance.

## Fixed gates and decision

- All source/vector/control bindings,codec edges,source selection/value/
  zero-code checks,217 exact segment reads and24 distinct layers pass.
- First16 states/layer:384 C/GPU outputs,median relative L2<=1e-4,max<=5e-4.
- All256 existing states/layer:pooled FP64 source-output SSE/energy<=.01,
  target original BF16-weight/FP32 smooth-SiLU FFNs. Report all24 layer
  errors,no separate layer threshold. Component states are not independent
  whole-model quality or original BF16 intermediate arithmetic.
- Native six-thread24-layer component median<=10ms over exactly three
  passes on256 states after16-state output check. Keep all timings,no
  retiming,best-pass selection or adjusted threshold.

Any failure stops this fixed32 codec/operator before source-prior/conditional
fitting. All pass licenses separately frozen mixed-precision source prior
qualification with the **unchanged1% stored-derivative gate**. No full
model,prediction/generation/tasks,useful E16/E160,n-scaled route/LUT/DRAM,
approximately10B/sparse-Giga applicability or>=50 accepted token/s follows.

Shape bytes:2,752,512 extra across24 layers;ideal whole selected ledger
544,804,868bytes. Whole composition and actual traffic remain unbuilt.
Resident count/cost arithmetic is in the [proposal](OUTLIER_SPLIT_OUTPUT_PROPOSAL_20261001.md).

## Budget and commands

One local3060 run,six host threads;20min after imports,20GiB RSS,10.5GiB GPU,
2GiB free disk,<400MB new fixture plus~1.4MB check/report.600s subprocess
timeout,no T4 or new external resource. Commit source/protocol before run;
failure stage/segments/source screens are retained. Outputs stay ignored,
result JSON CRLF is pinned for byte-hash reproducibility.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth238_outlier_split_cpu.c -o benchmarks/native_expert_scaling/meth238_outlier_split_cpu.exe -lm -lpsapi
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth238_outlier_split_feasibility.py --binary results/native_expert_scaling/meth238_outlier_split_fixture.bin --exe benchmarks/native_expert_scaling/meth238_outlier_split_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth238_outlier_split_native_result.json
```
