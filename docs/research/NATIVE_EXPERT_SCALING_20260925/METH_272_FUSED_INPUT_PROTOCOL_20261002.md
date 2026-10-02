# METH-272: same private128 function, fused shared input projections

271's output-aware rows pass source/reserve/native numerical gates but
11.398ms exceeds10ms. Changed variable: shared gate/up Q8 dots execute
interleaved in one loop, loading each input vector once for both matrices.
Keep each dot's four FMA chains,reduction order,scale application,LUT/product,
private BF16 overrides,down/residual/bias stages,all weights/IDs and thread
workshares identical. No arithmetic precision/selector/count changes.
This tests operator scheduling,not another source fit or new capacity.

Bind271 raw SHA
`638594a650189ab45efa62c08be04a3e42b21f7eff00c52d372d5991077120c3`,
fixture337,596,452bytes SHA
`ece28ea3344d9ecef674d30b4c43b067ee4290379cf9130c97c2eafee1e344d8`,
prior C/executable/check/include hashes,all361 segments and125vectors.
Run one unchanged271 invocation then one changed272 invocation,each with
384 output vectors and exactly three256-token24-layer sweeps,six threads.
This provides a contemporaneous control; it cannot reclassify271's failure.
No Torch/CUDA import/model job; same compiler flags,CPU and physical fixture.

Require all344,064 changed/check values bitwise equal both original271 and
contemporaneous control; aggregate256-state checksum exact. Only under
these finite controls inherit271's source fit/reserve and GPU numerical
gates. Changed kernel median must meet the original absolute<=10ms gate
AND be at least5% faster than contemporaneous control. No retiming,gate
widening,thread/affinity changes or alternate optimizations after failure.
Compiler/check/timing localCPU10min/20GiB RSS,each native timeout180s,
1GiB output,no GPU/T4/download. Preserve all failures/partial evidence.

Code `meth272_fused_source_input_cpu.c` and runner
`meth272_fused_source_input.py` are frozen with this protocol before any
execution. A component pass only licenses separately frozen full archive
and consumed regression; it does not reopen259 or establish full quality,
useful n,RAM routing/LUT/DRAM,accepted>=50 or family/10B/100B transfer.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth272_fused_source_input.py --exe benchmarks/native_expert_scaling/meth272_fused_source_input_cpu.exe --check results/native_expert_scaling/meth272_fused_source_input.check.bin --control-check results/native_expert_scaling/meth272_unchanged271.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth272_fused_source_input_result.json
```
