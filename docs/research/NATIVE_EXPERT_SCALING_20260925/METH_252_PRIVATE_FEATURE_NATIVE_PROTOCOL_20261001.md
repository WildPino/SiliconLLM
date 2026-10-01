# METH-252: frozen private source nonlinear feature native operator

Implement the [private feature proposal](NONLINEAR_PRIVATE_FEATURE_PRECISION_PROPOSAL_20261001.md)
before conditional unit selection. METH-251's fixed matched output-space
hierarchy fails useful count gain. Change selected input-feature precision,
not rank, ridge strength or quality thresholds.

## Source fixture and layout

Bind original METH-245 repaired result
`7022e2063afef2339342ff515ab4788732b7d795b8b2c690c511b4f5a5e5d61d`,
fixture326,580,256bytes SHA
`065b7b7363ba1a334c43a4bcdebb11145c0906d5fcccf622d5c800eb5258095d`,
qualified one-team METH-246 raw result
`bb84bdb1544025866e7411dd830ffea3bc5daf4b73447bf194c45c55ba9da202`,
its original check output, pinned Qwen0.5B source BF16 file/tensors and
original METH-125 source vectors. Local source only, no download.

New little-endian header `<8s6I>`,32bytes:
`M252PF01,24,896,4864,32,32,32` (layers,D,H,output escapes,residual rank,
private units). Copy all289 prior table/base/residual segments byte equal.
After each layer append source unit IDs uint16[32], original source BF16
gate[32,896] and up[32,896]. Total361 segments,329,334,308bytes including
4 extra header bytes and2,754,048 added payload. Exact header/segment/EOF
readback. Source file/tensor identities checked before copying rows.

For each source layer choose32 rows by descending **FP64 squared gate/up
coefficient quantization discrepancy**: sum original-BF16-as-FP32 minus
decoded Q8 row/scale error across both projections and all896 input columns.
Stable lowest source row ID breaks ties. Store chosen IDs in ascending ID
order. All scores positive, IDs unique and exact original source rows.
No activation, source-output target or held-out score selects this fixture.

## Exact C and independent GPU functions

Same one-team METH-246 OpenMP stages,6 threads,dynamic off. Prebuild int16
unit-to-private-slot maps before checks/timing. In the first static feature
loop, an unselected row executes unchanged shared Q8 four8-lane FMA dot,
reduction and row scale. A selected row executes original BF16 gate/up
dot with the existing four8-lane FMA order/reduction. Both use unchanged
513-point FP32 SiLU lookup and gate/up product, one writer per feature row.
No redundant shared projection for selected rows. Remaining mixed readout,
BF16 right/left factors,bias addition order and barriers are unchanged.
No new output matrix. Fixed maps do not measure dynamic router/map setup.

GPU oracle reads actual new bytes. Compute ordinary row-Q8 features,
original private BF16-as-FP32 gate/up projections for selected IDs, same LUT,
overwrite those features, and execute original separated mixed/factor/bias
readout. TF32 off/highest FP32 matmul/deterministic algorithms. Reference
source function remains original BF16-weight/FP32 FFN on same256 input
states per layer. No source/conditional fitting or validation corpus.

## Gates, resources and decision

All source/fixture/vector bindings and361 readbacks pass;289 original
segments unchanged. All24 layers have nonzero private feature effects on
the256 source states and changed native check payloads versus METH-246.
Independent GPU/native check16x24x896,384 vectors: median relative L2<=1e-4,
maximum<=5e-4. All finite. Source256x24 pooled SSE/energy<=.01; record ratio
to prior source error, not a learned-count/generalization claim.

One C invocation writes M252OUT1 check then runs exactly three256-state
24-layer timing passes. Retain all passes, median<=10ms/token. Failure
closes this fixed operator before cell selection. All pass licenses a
separately frozen matched E16/E160 source-unit selection experiment, not
native learned-bank, dynamic route/LUT/DRAM/large-n or full accepted rate.

Conservative addressed-shape bound556,492,292bytes from original ledger
plus source row/ID payload; the prototype also reads unit maps occupying
`24*4864*2=233,472bytes`, yielding556,725,764 if additionally charged. Do
not assume skipped-row savings/cache residency. Actual timed fixture and
fixed maps are not a complete model or large conditional pool.

Local RTX3060/six host threads,20min after imports,20GiB RSS/10.5GiB GPU,
>=2GiB free disk,native timeout180s. GPU synchronized before native timing;
no concurrent GPU computation or project job. Retain failure/partial stage,
source row choices and all resource/cost/hash details. No T4. Compile and
freeze source/runner/protocol before execution:

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth252_private_feature_cpu.c -o benchmarks/native_expert_scaling/meth252_private_feature_cpu.exe -lm -lpsapi
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth252_private_feature_feasibility.py --binary results/native_expert_scaling/meth252_private_feature_fixture.bin --exe benchmarks/native_expert_scaling/meth252_private_feature_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth252_private_feature_native_result.json
```
