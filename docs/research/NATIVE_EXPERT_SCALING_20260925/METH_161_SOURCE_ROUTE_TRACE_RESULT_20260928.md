# METH-161: exact source-route trace recovered

**Decision:** valid trace for offline postfailure diagnostics. The frozen [protocol](METH_161_POSTFAILURE_SOURCE_ROUTE_TRACE_PROTOCOL_20260928.md) captured the actual E1,280 source-child selections for all METH-158 training inputs and replayed the unchanged METH-150 grandchild rule. All 24 layers' all-slot and content-slot histograms match METH-159 exactly. This is a routing trace, not a new quality or training result.

The [runner](../../../benchmarks/native_expert_scaling/meth161_capture_source_route_trace.py) used the METH-158 manifest SHA-256 `09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69`, merged teacher SHA-256 `cdcb22a4273148dede97a17eac81345c4fc5f8e40f09cb18ecac4e35115e437c`, METH-159 result SHA-256 `48074f7b404b409ba177acbbee97c196c1e2aa33f170930267fd25489edba509`, and pinned source bank/child checkpoint/table hashes recorded in the [machine result](meth161_source_route_trace.result.json). The result SHA-256 is `b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c`. The exact launch was:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth161_capture_source_route_trace.py --out-prefix docs\research\NATIVE_EXPERT_SCALING_20260925\meth161_source_route_trace
```

The trace contains 5,120 sequences, 887,330 input tokens and four source-child IDs per token/layer (3,549,320 selections/layer). The original table sends 659,704 selections/layer to shared local zero. The saved arrays and offsets total 179,526,964 bytes; the result records each file's SHA-256 and shape. Their [ZIP archive](meth161_source_route_trace.zip) is 98,414,202 bytes, SHA-256 `0fb44bb84a51e7fc567dd9365e7352d79def5eba59cc29c82400a61c39f132e9`. Extract it into this directory to replay METH-162/163; the scripts verify every extracted entry against the result and resolve its basename if the original absolute path differs. Minimum layer source-child coverage is 1,270/1,280. BF16 teacher/control parity and all 24 exact histogram comparisons passed. RTX 3060 elapsed time was 979.578 seconds, peak allocated GPU memory 2,945,090,560 bytes and final process RSS 4,475,260,928 bytes, within the preregistered stops.

The source-child trace enables [METH-162](METH_162_POSTFAILURE_STRUCTURAL_TABLE_DIAGNOSTIC_PROTOCOL_20260928.md) and [METH-163](METH_163_PREFIX_CONTEXT_HASH_DIAGNOSTIC_PROTOCOL_20260928.md) to compare grandchild assignment without another donor forward. Those diagnostics must still be treated as in-sample because they inspect the METH-158 training inputs on which METH-159 failed.
