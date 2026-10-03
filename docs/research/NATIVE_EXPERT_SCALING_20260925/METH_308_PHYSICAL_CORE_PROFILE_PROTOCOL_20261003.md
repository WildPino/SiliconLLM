# METH-308: exact306 binary on six verified physical cores, ACTIVE200ms

Freeze NEW controller/protocol before observations. 307's unbound ACTIVE
waiting profile has11.3-14.8ms medians but fails14ms/repeatability and fresh
PASSIVE A/C drift. New variable: physically distinct CPU bindings, not new
warmups, precision, weights, geometry, compiler, kernel or timing subset.
No evidence yet that affinity caused307 variation or will solve it.

## Actual resource and execution profile

Re-probe Windows GetLogicalProcessorInformation/RelationProcessorCore and
caller allowed affinity. This declared local topology requires six disjoint
core masks covering12 logical processors, choose lowest allowed ID per core;
currently0,2,4,6,8,10. Win64 structure32 bytes/required-size122 flow, fail
unsupported topology or inaccessible core rather than use sibling duplicates.
Native metadata read is retained with API/bytes/masks/current allowed IDs.

Use exact306 executable SHA
f7a22f25a7531bce4d9f8d67a05839039e764f9889907af69bc2727ccfa16de1.
Fresh source components/spec bytes equal306; original source/helper/controller
pins checked. No recompilation/new C/engine mutation. All640 synthetic
functions per layer/all9.437B routed coefficients, packedI4/row-tile4/full
original Q6 head/source router/norms remain306. No learned quality claim.

Three fixed fresh native processes, ALL with six threads/OMP_DYNAMIC=FALSE,
OMP_WAIT_POLICY=ACTIVE,KMP_BLOCKTIME=200,KMP_SETTINGS=TRUE. KMP_AFFINITY:
verbose,granularity=thread,proclist=[selected six IDs],explicit. Clear competing
KMP_LIBRARY/OMP_PLACES/OMP_PROC_BIND in CHILD environment only; caller
environment unchanged. Record config and effective runtime policy/blocktime.
Verbose runtime must show thread0..5 each bound to its declared single OS
logical ID, one per distinct physical core. Allow repeated identical binding
lines, forbid missing/non-singleton/changed/extra thread bindings. Environment
requests alone are insufficient proof. Primary implementation/config context:
[LLVM runtime](https://openmp.llvm.org/design/Runtimes.html),
[LLVM affinity message source](https://lists.llvm.org/pipermail/openmp-commits/2017-January/002037.html).

## Unchanged controls and stronger cross-process cost gate

- Each process original10 inputs/3 repetitions/two warmup/eight measured;
  preserve all90 timings/output/route hashes. All90 hashes and entire
  selftest/ready counts must equal306. Old row/nibble/tile/extreme/packed-head/
  sum-bias/bad-bank/source-router controls unchanged, no new ignored samples.
- Each process max/min of three medians<=1.10. Additionally max/min of the
  three PROCESS medians<=1.10. Any variation failure stops inconclusive.
- EACH of ALLnine medians<=14ms required; stable cost failure closes this
  unchanged profile before teacher capture/training. Preserve307 failure,
  no favorable process/repetition selection or14ms gate revision.
- Pass licenses only separately frozen real whole teacher-function fit/
  I4 precision-quality work. Input-ready operator costs omit causal attention,
  cache/RoPE/embedding/residual model composition; keep6ms balance. Actual
  quality and accepted>=50 batch1decode must later use the SAME trained artifact.

Active workers consume CPU while waiting; declared six physical cores/
finite200ms idle window. No overlap with model/performance jobs/GPU/T4/
downloads/delegation. Same600s/24GiB per process,>=16GiB available before each,
expected15-30s all three sequential processes,~172MB fresh source hashes,
~5.1GB bank allocation each. Save all logs/settings/spec/result/failure.
An apparatus repair retains failed execution and is frozen before rerun.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth308_physical_core_profile.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth308_physical_core_profile_result.json
```

Full pretrained transfer/quality/useful RAM-scale n/real DRAM/routing and
SAMEartifact accepted50 across verified multiple families/scales remain open.
