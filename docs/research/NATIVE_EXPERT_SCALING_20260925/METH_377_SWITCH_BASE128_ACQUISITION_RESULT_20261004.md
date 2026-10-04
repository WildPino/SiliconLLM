# METH-377: acquisition apparatus failure before network work

Frozen controller/protocol: `e20771a`. Invocation:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth377_switch_base128_acquire.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth377_switch_base128_acquisition_result.json
```

Exit 1 at the committed-file binding stage. The controller references
`METH_377_SWITCH_BASE128_ACQUISITION_PROTOCOL_20261003.md`; the frozen actual
protocol is dated `20261004`. Git cat-file returns 128 before any network
request or output-directory creation. Raw failure:
`meth377_switch_base128_acquisition_result.failure.json`, elapsed 0.078s,
downloaded bytes 0, maximum checked RSS 0 (resource sampling not reached).

This is an apparatus failure, with no checkpoint observation, acquisition
qualification, quality or performance result. Preserve the frozen controller,
protocol and raw first failure. METH-378 will correct the protocol path under
a new experiment name and retain the same fixed source and resource guards.
The subsequent all-original tensor/tie/distinct-function binding becomes
METH-379. The broader final goal remains active.
