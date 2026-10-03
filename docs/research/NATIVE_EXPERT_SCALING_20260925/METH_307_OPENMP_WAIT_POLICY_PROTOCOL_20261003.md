# METH-307: exact306 binary, declared worker-wait profile A/B/A

Freeze controller/protocol BEFORE observations. New variable is OpenMP
worker waiting between hundreds of projections. The explicitly PASSIVE
306 profile fails14ms despite310.572MB addressed-weight scenario. No
weights, geometry, compiler, kernel or binary changes in this experiment.

LLVM documents active spinning versus passive yielding/sleeping, with
KMP_BLOCKTIME controlling post-region time before sleep; this motivates
measurement and does not promise a speedup. Primary sources:
[LLVM runtime configuration](https://openmp.llvm.org/design/Runtimes.html),
[LLVM policy test](https://openmp.llvm.org/doxygen/omp__wait__policy_8c_source.html).
Runtime effective settings must be recorded/verified, not inferred from docs.

Use EXACT306 executable SHAf7a22f25a7531bce4d9f8d67a05839039e764f9889907af69bc2727ccfa16de1,
its source/controller/result pins and freshly hash-checked source components/
same spec bytes. No recompilation/new C/engine mutation. Three sequential
fresh processes, order fixed: A PASSIVE, B ACTIVE, C PASSIVE. For each retain
all10 fixtures×three repetitions/two warmup/eight measured, numeric/format/
source/packed-head/route/fault controls and fully populated640-function bank.

All arms six threads/OMP_DYNAMIC=FALSE, identical inherited affinity settings.
Remove KMP_LIBRARY to avoid a competing policy override. PASSIVE removes
KMP_BLOCKTIME, expecting effective0. ACTIVE sets KMP_BLOCKTIME=200ms, allowing
workers to remain ready between nearby projections with a finite idle window.
Set KMP_SETTINGS=TRUE, retain stderr and verify actual effective policy/block
time. New profile uses CPU while workers wait; this resource behavior is
declared, not assumed free. No other model/performance job or GPU/T4.

ALL90 output/route hashes and entire selftest/ready counts must equal306.
Each arm max/min three medians<=1.10; fresh passive A/C whole median drift
<=1.10 or comparison inconclusive/no training. Active EACH median<=14ms
required. Report all arm medians and active/passive ratios without subsets.
Stable active failure closes this profile before teacher collection/training.
An active pass only licenses separately frozen real teacher-function fit/
I4 precision quality; original306 PASSIVE failure remains unchanged.

No accepted-token or whole quality inference: causal attention/cache/RoPE/
embedding/residual model composition still absent; reserve6ms complete target
balance. Synthetic children still establish no useful transferred capacity.
Original coefficient geometry, full original Q6 head and source macro router
must remain; do not shrink n, omit the head or relax the final goal.

Resources:600s/24GiB per child, three fixed arms,>=16GiB available before
each, CPU6 threads only, expected approximately15–30s including full bank
initialization. Poll/kill through existing306 launcher, preserve failures.
Fresh~172MB source hashes, three~5.1GB fully populated allocations sequentially;
save isolated logs/effective settings/spec/result, source artifacts unchanged.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth307_openmp_wait_policy.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth307_openmp_wait_policy_result.json
```

Method still needs real pretrained capacity transfer, heldout/generation/tasks,
useful large-n CPU routing/DRAM, SAMEartifact accepted>=50batch1token/s and
verified multiple-family/scale applicability. Runtime scheduling is enabling.
