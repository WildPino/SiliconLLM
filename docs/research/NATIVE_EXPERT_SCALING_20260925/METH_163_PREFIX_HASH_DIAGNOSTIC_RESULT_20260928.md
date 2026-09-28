# METH-163: prefix hash redistributes one collision but fails elsewhere

**Decision:** reject the full-prefix hash as the current tenfold route. The [frozen diagnostic](METH_163_PREFIX_CONTEXT_HASH_DIAGNOSTIC_PROTOCOL_20260928.md) retained the 75-entry shared table and mixed one causal FNV-1a 64-bit prefix state into the nine-way content hash. Golden prefix states and the predicted content/shared grandchildren passed. The old route exactly reproduced all 24 METH-159 histograms from the METH-161 trace.

The [runner](../../../benchmarks/native_expert_scaling/meth163_prefix_context_hash_diagnostic.py) required METH-161 result SHA-256 `b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c`. Its [machine result](meth163_prefix_hash_result.json) SHA-256 is `7ca7e987b380c8c46b78359473815eafb149670311b6dba59f9d8c6b7a35e49b`; saved prefix states are [here](meth163_prefix_states.npy) and SHA-bound in that result. Reproduce with:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth163_prefix_context_hash_diagnostic.py --trace-sha b717b8db5f5565d377c3d2d043286c017ba98a661bedc9c6426ae660ce07b93c --prefix-out docs\research\NATIVE_EXPERT_SCALING_20260925\meth163_prefix_states.npy --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth163_prefix_hash_result.json
```

The original layer-2/source-child-314/local-1 group of 186 selections spreads across the nine content locals as `[19, 34, 24, 24, 23, 15, 17, 16, 14]`. But other repeated contexts yield a new worst hot-parent grandchild share **35.81%** in layer 22, above the <=25% gate. Minimum active content median remains **39** versus >=50. Coverage, under-32 fraction and global load ratio pass (minimum 10,829; maximum 48.28% and 1.039). The result is therefore a local collision repair, not a general routing pass.

The CPU replay took 23.390 seconds and 1,568,321,536 bytes final RSS. No native C parity/cost, training or quality test was run. An alternative may combine shared-context handling with prefix state, but it must be specified prospectively and checked on new data; this rejected variant cannot license E12,800 training.
