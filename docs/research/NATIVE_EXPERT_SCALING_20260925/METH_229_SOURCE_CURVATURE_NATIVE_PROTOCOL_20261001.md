# METH-229: actual affine plus H2048 source nonlinear operator

Freeze before execution. The previous goal turn made progress: METH-227
measures24.3% count gain but rejects absolute first-order accuracy;
METH-228 excludes bias-only repair. The [changed proposal](CONDITIONAL_SOURCE_CURVATURE_PROPOSAL_20261001.md)
retains nonlinear response within one conditional function. Price and
qualify that complete operator before source-unit selection or fitting.

Original hash-bound Qwen0.5B-Instruct BF16 source, all24 layers, existing
METH-125 token0 inputs as centers. Predetermined first2048 source-unit
rows, not quality-selected or a hard-carve candidate. Use full donor
J(c), subtract the selected branch Jacobian to initialize affine weights;
source selected gate/up/down retain original BF16 values. Combined
unrounded function must match source value and derivative at c.

Compare its complete autograd Jacobian with the analytic original donor
Jacobian: relative Frobenius<=1e-5,max element error/source peak<=1e-4.
Unrounded/stored center relative L2<=1e-5,all finite. BF16-round affine,
retain FP32 bias recomputed using rounded affine and selected branch.
Store affine896x896,gate/up2048x896,down896x2048 BF16 plus896 FP32 bias,
in that order.24-byte M229BF01 header(24,896,2048,32), total302,862,360
bytes. All120 segment readbacks exact;24 affine hashes distinct.

Native AVX2 BF16 decode/FMA uses the same audited four-accumulator
matvec reduction. Activations, SiLU, products/sums/bias FP32; no hidden
quantization. Output order `(affine+nonlinear)+bias`, matching the oracle.
Six CPU threads,256 existing varied states across all24 distinct layers,
three fixed complete passes. GPU synchronize before CPU timing. Require
16x24 stored-output oracle relative L2 median<=1e-4,worst<=5e-4, and
median24-layer combined component<=10ms. No retiming/kernel change after
outcome. Separate smaller-operator passes cannot substitute this test.

Pass licenses freezing source-anchored E16/E160 nonlinear readout learning;
complete function error<=.01 and all final requirements stay unchanged.
Failure stops this width/kernel before fitting. No source selection,
conditional route, large-bank DRAM/LUT ratio, full quality/generation/
task or accepted token rate is scored. The hypothetical whole selected
ledger530.020MB is arithmetic only, not measured traffic/rate.

Budget20min after imports,20GiB RSS,10.5GiB GPU,<350MB output; local3060
and six host threads,no T4/external data/new source inference/port resume.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth229_source_curvature_cpu.c -o benchmarks/native_expert_scaling/meth229_source_curvature_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth229_source_curvature_feasibility.py --binary results/native_expert_scaling/meth229_source_curvature_fixture.bin --exe benchmarks/native_expert_scaling/meth229_source_curvature_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth229_source_curvature_native_result.json
```
