# METH-246: frozen single-team execution of unchanged rank32 functions

## Changed execution mechanism

METH-245's source-derived combined operator passes every numeric/source/
serialization gate but four-team execution costs10.390391ms>10. Preserve
rank32,all326,580,256 fixture bytes and exact FP32 reduction arithmetic;
change only scheduling/fusion. Bind METH-245 repaired result SHA256
`7022e2063afef2339342ff515ab4788732b7d795b8b2c690c511b4f5a5e5d61d`,
physical fixture `065b7b7363ba1a334c43a4bcdebb11145c0906d5fcccf622d5c800eb5258095d`,
actual original C output and METH-125 vectors.

Single OpenMP parallel region per layer. First static workshare computes
both gate/up rows,with unchanged independent four8-lane q8/FMA reductions,
row scales and scalar LUT/product into shared phi. Barrier. Second static
workshare computes base mixed output rows with unchanged escape reduction,
nowait. Third static workshare computes32 right BF16 projections with
unchanged four8-lane reductions,then barrier waits for both output/right
tasks. Fourth static workshare adds left BF16 dot,base bias,residual bias
in the original order. Six threads,dynamic teams disabled. No new weights,
factor compilation,source/validation/route fitting or rank change.

## Fixed gates and decision

Readback all289 existing segments/header/EOF exact. C check output M246OUT1,
16x24x896 FP32. Require **all344,064 elements bitwise equal** to METH-245
output payload (header differs),and exact original three-pass aggregate
checksum over256 states. Existing GPU-oracle numeric/source-function
gates carry only with bitwise equality and same physical functions.
Retain all timing passes;one new executable invocation after16-state
check,three256-state24-layer passes,median<=10ms. No retiming/rank/threshold
retry. Any failure closes this fixed single-team kernel before learning.
Pass licenses separately frozen actual raw-solution factorization,not
full-model/large-n/10B quality or >=50 acceptedtok/s.

## Budget and commands

One localCPU run,six threads,6min after imports,20GiB harness RSS,native
timeout180s;existing fixture,small check/JSON,no GPU computation/T4/new
source/download. Record harness/native memory/time and exact payload
hashes. Commit source/runner/protocol before execution.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth246_single_team_residual_cpu.c -o benchmarks/native_expert_scaling/meth246_single_team_residual_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth246_single_team_residual_feasibility.py --exe benchmarks/native_expert_scaling/meth246_single_team_residual_cpu.exe --check results/native_expert_scaling/meth246_single_team_residual.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth246_single_team_residual_native_result.json
```
