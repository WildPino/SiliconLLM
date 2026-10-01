# METH-253: frozen split shared/private feature execution

METH-252 source-only BF1632 feature replacements reduce local source
error81.03% and pass numeric/serialization but median10.299ms fails10ms.
Change only execution stages, not weights/function/thresholds or selector.

Bind raw252 result
`7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce`,
329,334,308byte fixture
`b90193f090812b9fd991616c0150f28f16a6553f1e2af2c1af8d8f5a489ba395`,
original C check and256-state METH-125 vectors. Read all361 segments and
header/EOF exactly. No new fixture, selector, fitting, source/target scoring.

Same single OpenMP team/layer,6 threads,original shared workshare computes
gate/up Q8 dots and LUT/product for all4864 rows without a private branch.
Barrier. New static32-slot workshare computes identical BF16 gate/up dots,
LUT/product and overwrites each distinct private feature. Barrier before
original mixed down nowait/BF16 right, barrier, original left/bias stages.
No unit-to-private maps; IDs directly select private output slots. Exactly
one writer per private slot in that stage, no concurrent readers/writers.
The32 shared projections are redundant, a measured compute tradeoff.

One new executable invocation:16x24x896 check, then exactly three256-state
24-layer timing passes. All344,064 check elements bitwise equal252 and
all256 aggregate checksum exact; prior numeric/source function gates carry
only on bitwise equality with same physical functions. Median<=10ms, no
retiming/rank/threshold retry. Pass licenses separately frozen matched
E16/E160 actual private source-unit selection; otherwise close this kernel.

One local CPU run,6min after imports,20GiB RSS,native timeout180s, GPU not
used/no T4/download. Actual fixture bytes/readout unchanged; no learned
capacity,dynamic router/map setup,large-n/DRAM,new/full quality or full
accepted rate. Freeze source/runner/protocol before execution:

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth253_split_private_feature_cpu.c -o benchmarks/native_expert_scaling/meth253_split_private_feature_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth253_split_private_feature_feasibility.py --exe benchmarks/native_expert_scaling/meth253_split_private_feature_cpu.exe --check results/native_expert_scaling/meth253_split_private_feature.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth253_split_private_feature_native_result.json
```
