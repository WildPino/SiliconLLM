# METH-234: frozen row-Q8 plus interpolated SiLU LUT prerequisite

## Uncertainty and new operation

METH-232 source-function fidelity passes but11.023ms fails10ms. METH-233's
bitwise-identical instrumented arithmetic measures3.004ms in scalar
SiLU/product (25.36% phase share). Qualify a **changed nonlinear operation**
that might resolve this component deficit while retaining all4864 source
features. This is not retiming METH-232 or changing its failure threshold.

Reuse METH-232 source/revision/vector bindings, all72 code/scale matrices
and24 zero output biases. Require all168 matrix/bias segments have exactly
the same bytes/hashes as its completed record SHA256
`5dad8dcb67d2b7eeae0c747d63b5cada6e2ce6bd6faed94bac32dd6052c94368`.
No input/hidden activation quantization, source selection or readout fitting.

## Shared nonlinear lookup

513 samples from-16 to16 inclusive with step1/16. Generate CPU FP64 SiLU
at those exactly represented inputs with torch, round samples to FP32 and
store/read back the actual shared table. Library version and table hash
are retained; the physical table defines the operator after export.

For gate<=-16 return0; for gate>=16 return gate. Otherwise FP32
`p=(gate+16)*16`; if p rounds to512 return gate. Let i=floor(p), f=p-i;
return `table[i] + f*(table[i+1]-table[i])`. Same GPU indexing/boundary
policy. No expf in the native nonlinear loop. Multiply by FP32 up and use
the unchanged row-scaled int8/FP32 down projection+bias. Accumulation
order may differ C/GPU within frozen numerical limits.

Binary `<8s4I>`: `M234LQ01`,24,896,4864,32,followed by513 FP32 samples
and unchanged layer payloads. Exactly314,894,364bytes,169 hashed segments.
Shared table adds2052bytes; hypothetical ideal whole selected ledger is
542,052,356bytes,not a complete artifact or measured DRAM traffic.

## Frozen checks and decision

- Lookup maximum absolute error<=.001 on131073 fixed evenly spaced
  FP32 inputs in[-32,32],including table nodes and outer branches. This
  discrete scan is not a universal analytic error bound.
- All source bindings/codec edges/169 exact readbacks/168 unchanged controls
  pass; all24 source code matrices remain distinct for each organ.
- First16 states across24 layers:384 actual C versus declared GPU LUT
  oracle outputs, median relative L2<=1e-4,maximum<=5e-4.
- All256 states across24 layers:pooled FP64 output SSE/energy<=.01 against
  original BF16-weight/FP32 source FFNs. Report all layer errors. This
  jointly measures row quantization and LUT approximation on component
  states; no BF16 whole-model or fresh-source claim.
- Six-thread native24-layer component median<=10ms over exactly three
  passes on256 states after the16-state output check. Keep all passes;
  no retiming, best-pass selection or adjusted threshold.

All pass licenses separately frozen full-feature conditional transfer;
any failure stops this fixed LUT/operator before fitting. No routing/LUT
bank selection, real large-RAM n ladder, full prediction/generation/task
quality or accepted complete token rate is established by this prerequisite.
Sparse GigaChat/wider source shapes need independently priced variants.

## Budget and reproducibility

Local3060,six CPU threads;max20min after imports,20GiB RSS,10.5GiB GPU,
2GiB free disk,<400MB new fixture plus~1.4MB check/report.600s native
subprocess timeout,one run, no concurrent project inference,no T4.
Protocol/source commit precedes execution; every stage/resource exception
is saved. Result JSON has pinned CRLF; all binaries/fixtures stay ignored.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth234_row_q8_silu_lut_cpu.c -o benchmarks/native_expert_scaling/meth234_row_q8_silu_lut_cpu.exe -lm -lpsapi
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth234_row_q8_silu_lut_feasibility.py --binary results/native_expert_scaling/meth234_row_q8_silu_lut_fixture.bin --exe benchmarks/native_expert_scaling/meth234_row_q8_silu_lut_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth234_row_q8_silu_lut_native_result.json
```
