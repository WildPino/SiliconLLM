# METH-162: enlarged structural table fixes the observed hot slot, not support

**Decision:** retain only as an in-sample candidate. The [frozen diagnostic](METH_162_POSTFAILURE_STRUCTURAL_TABLE_DIAGNOSTIC_PROTOCOL_20260928.md) added every previously nonstructural causal `(token, previous, local position)` tuple appearing at least 32 times among the 887,330 METH-158 input positions. Exactly 63 tuples and 3,487 positions were added to the original 75-entry table, for [138 entries](meth162_training_derived_shared_table.json), SHA-256 `dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1`. No tuple was chosen from its route outcome.

The [runner](../../../benchmarks/native_expert_scaling/meth162_postfailure_table_diagnostic.py) required the METH-161 trace result SHA-256 `b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c` and exactly replayed all old METH-159 histograms before applying the new table. The [machine result](meth162_postfailure_table_result.json) SHA-256 is `d10ede395f1f4008b754028c840a120d41523e47e08f4aa6b6848978a98df01b`. Reproduce with:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth162_postfailure_table_diagnostic.py --trace-sha b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c --table-out docs\research\NATIVE_EXPERT_SCALING_20260925\meth162_training_derived_shared_table.json --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth162_postfailure_table_result.json
```

The original worst layer-2/source-child-314/local-1 slot received 186 content selections. One causal tuple `(1380, 6109, 164)` supplied 146 of them and occurred 148 times globally. The enlarged table redirects those 146 selected occurrences to shared local zero. Structural traffic rises from 659,704 to 673,652 selections/layer, exactly four times the 3,487 newly shared input positions. On the same training inputs, minimum content coverage is 10,829/11,520, worst global content load ratio 1.032, maximum under-32 fraction 48.26%, and worst hot-parent grandchild share 24.26%; those four gates pass. **Minimum active content median stays 39 versus the frozen >=50 gate and fails.**

The CPU replay took 31.953 seconds and 1,668,165,632 bytes final RSS, within the 15-minute/16-GiB stops. It performed no training, quality evaluation or native C cost measurement. The table must be frozen and checked on independent sources/context lengths before any route promotion. More distinct training contexts are required for the median-support failure; this in-sample correction cannot retroactively pass METH-159.
