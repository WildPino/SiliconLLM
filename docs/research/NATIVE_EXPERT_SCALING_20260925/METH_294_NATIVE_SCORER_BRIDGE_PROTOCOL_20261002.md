# METH-294: native full-head scoring bridge

## Prospective question and decision

Does the new phase60 scoring entry preserve original285 arithmetic and
correctly serialize top1 and full-head NLL? Reuse consumed285 source0,
both bank-off/on arms, last8 prefix positions. No new292 model observation.
Passing licenses a separately frozen native-primary prediction study under
[292's declared policy](METH_292_NATIVE_PRIMARY_EVALUATION_POLICY_20261002.md).
It does not repair the prior5% numerical failures or qualify model quality.

The new window forward is mechanically checked to equal original285 after
only function/signature renaming and adding a global RoPE position offset.
Local cache indices stay relative to each independently rebuilt window.
This run uses offset0; nonzero-offset dataset alignment needs its own
prospective assertions before new-source scoring.

## Fixed gates and identity

Original276 archive SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`;
original285 result SHA
`c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda`.
Bind actual old native-output bytes and278 manifest through that result.
Actual entry is `benchmarks/phase60/engine.c` with
`SILICON_COMPLETE_I16_NATIVE_SCORE`, original285 model/source operators,
original direct725-field loader, RNE/ISA/edge guards and six threads.

Before observation freeze these gates:

- All725 fields bound from the same archive; no fallback.
- Exact original forward body after reversing the two declared edits.
- First full-head row in EACH arm byte exact to actual original285 CPU.
- All16 top1 IDs exactly match original285.
- All16 NLL values within absolute1e-9 of NumPy float64 logsumexp of the
  SAME saved actual CPU logits. Last target=-1 sentinel must give0.
- Binary headers, positions, targets and EOF verified exactly.

Store executable/source/entry/bundle/output hashes, compile command,
actual loader report, per-row errors and elapsed time. Any failed gate
stops new-source scoring. Preserve apparatus failure before narrow repair.

## Cost and command

Expected10-30s CPU only; compile hard60s, execution hard120s. No GPU/model
job or performance benchmark overlaps. The elapsed value is assay cost,
not accepted decoding throughput. No downloads/T4/new learned capacity.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth294_native_scorer_bridge.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth294_native_scorer_bridge_result.json
```
