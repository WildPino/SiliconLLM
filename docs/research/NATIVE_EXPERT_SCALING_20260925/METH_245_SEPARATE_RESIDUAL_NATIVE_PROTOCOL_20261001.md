# METH-245: frozen mixed readout plus separate rank32 BF16 correction

## Decision changed and fixed representation

METH-244 exposes49.53% continuous E160 fit gain erased by encoding.
Before learned factorization,qualify the added residual operator within
the existing10ms24-layer FFN component allotment. Bind METH-238 result
`61b496f66455e764039877ddaf17cedaa3aab49c287f09178e085afd56dbe684`,
its317,646,876byte physical fixture,original Qwen source/vector identities
and METH-244 result
`5164e4deb8ee6866b9d3b1f62f0d3468f6e7d814c517000876ccbf8305949849`.
No fit/validation/route training/new source data. These are source-derived
operator fixtures,not learned conditional weights or a count comparison.

Keep **all217 existing base segments byte-exact**:input maps/scales,LUT,
mixed down codes/scales/32 indexed BF16 escapes and biases. Add right BF16
32x4864,left BF16896x32 and FP32 residual bias896 per layer. Source fixture
biases are zero but actually read/applied in both operators.

At each original layer form FP32 residual original BF16-decoded down minus
decoded mixed down. Full thin FP32 CUDA SVD,`gesvdj`,descending singular
values,rank32 fixed. Require total spectral/norm energy agreement<=1e-5.
Canonicalize each retained vector sign by largest-absolute left entry,
lowest output row at ties,positive there. Balanced factors U32*sqrt(S32)
and sqrt(S32)*V32,round separately BF16. All factors nonzero,finite and
distinct across24 layers. Report residual energy/factor rounding error;
weight-energy fraction is diagnostic,not task-quality acceptance.

## Arithmetic,layout and prerequisites

GPU oracle uses the unchanged explicit row-scale-after-reduction mixed
output plus `linear(linear(phi,right.float()),left.float())`,then base bias,
then residual bias. No precombined full output matrix. Gate/up/LUT phi is
unchanged. Native AVX2/FMA decodes BF16 factors on each addressed load.
Right projection:32-row OpenMP pass,each row four8-lane accumulators over
4864 features,reduce(a+b)+(c+d),then ascending eight scalar lanes. Left
32-column dot uses the same reduction,inside the existing896-row down
OpenMP pass;add correction after base+escape,biases afterward. Six threads.

Header `<8s5I>` magicM245RS01,24,896,4864,32 escapes,32 residual rank.
513 FP32 table,then each layer's original nine segments,followed by
right BF16,left BF16,residual FP32 bias. Exactly289 segments and326,580,256
bytes (four extra header bytes plus8,933,376 factor/bias bytes). Output
magicM245OUT1 and16x24x896 FP32 check values. Require every segment/header/
EOF readback and217 source controls exact. Retain C/executable/helper hashes.

Numerical median<=1e-4,max<=5e-4 relative output L2 on384 actual C/GPU
outputs. Complete pooled source-function SSE/energy<=.01 on256 states x24
layers versus original BF16-weight/FP32 smooth FFN. This is not whole
BF16 execution or independent model quality. After16-state check,exactly
three256-state24-layer C passes;median<=10ms,no retiming/best-pass choice.

Failure closes this fixed rank32 operator before learned factorization;
no rank/threshold retry. All pass licenses a separately frozen activity-
weighted factorization of raw conditional solutions,actual E16/E160/
rotated/prior gates and stored-bank/native route/DRAM tests. Full quality,
useful large n,>=50 acceptedtok/s and actual10B/second family remain required.

## Budget and commands

One local3060 run,six host threads,20min after imports,20GiB RSS,10.5GiB GPU.
2GiB free disk;<400MB new fixture,small outputs/reports. Native timeout600s.
Original local weights only,no T4/download. Preserve partial factors/failed
stage. Commit runner/operator/protocol before execution.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth245_separate_residual_cpu.c -o benchmarks/native_expert_scaling/meth245_separate_residual_cpu.exe -lm -lpsapi
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth245_separate_residual_feasibility.py --binary results/native_expert_scaling/meth245_separate_residual_fixture.bin --exe benchmarks/native_expert_scaling/meth245_separate_residual_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth245_separate_residual_native_result.json
```
