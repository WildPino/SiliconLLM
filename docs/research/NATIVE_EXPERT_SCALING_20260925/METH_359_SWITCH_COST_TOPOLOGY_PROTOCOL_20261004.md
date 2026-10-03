# METH-359: consumed cost variability/actual Windows topology

Freeze before analysis,358 full repeat failure preservedc5738ff. Read only
exact committed358 raw and local actual Windows topology; no inference,
new native timings or changed candidate/data. API GetLogicalProcessorInformation,
RelationProcessorCore, single local group<=64 logical processors. ctypes x64
structure32B union/reserved16B checked; API insufficient-buffer122 handshake,
actual disjoint core masks/SMT flags. All6 physical/12 logical processors must
match psutil counts. Intersect actual process-allowed mask, select minimum
allowed logical processor per ACTUAL core; no assumption that odd/even IDs
mean physical cores. No affinity changes in this analysis.

Fixed consumed358 ALL4 CPU1/6 source9/64 rows: encoder/crossKV/decode/full
primary max/min ratio/range; each32-position median/max/argmax and descriptive
excess above same-row median. No independent per-position significance,
noise distribution or causal SMT/migration/DRAM attribution. Full rows already
retained358, no new timing observations. MAIN<=60s/RSS<=256MiB; preserve any
failure before separate repair. Require raw identity and closed physical
masks/allowed one-per-core mapping. If closed, separately freeze new process-
affinity profile, SAME356 executable/payload/inputs/gates, with every actual
child mask readback and exact numerical bridges before rates.351/358 failures
remain; NEW quality and actual accepted rate still required, final goal open.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth359_switch_cost_topology.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth359_switch_cost_topology_result.json
```
