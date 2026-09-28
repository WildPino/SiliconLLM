# METH-166: 138-entry native CPU lookup passes its component gate

**Decision:** the enlarged structural-table lookup passes the frozen [native component protocol](METH_166_NATIVE_TABLE_COST_PROTOCOL_20260928.md). This only prices parent/child/grandchild routing on 256 actual METH-125 hidden states, with old versus new table paired inside one process. METH-165 independently rejects the route's pooled hot-parent behavior, so this cost pass does not license training.

The [fixture exporter](../../../benchmarks/native_expert_scaling/meth166_export_table_fixtures.py) bound METH-162's 138-entry JSON SHA-256 `dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1`, the original 75-entry table and three 256-context cells: old shared hits, 63 new shared tuples cycled, and misses. The [fixture report](meth166_native_fixtures.json), SHA-256 `9d9d0311b8fc33f9e458516c71ec6dfe3736612db32011cdca1c401e9fe10145`, binds the binary files. Old/new binary tables contain 916/1,672 bytes. Python and [C](../../../benchmarks/native_expert_scaling/meth166_native_table_cost.c) agree on the checked child-89/layer-3 grandchild for each cell, including `893→890` when the new table shares a formerly content-routed tuple.

The [runner](../../../benchmarks/native_expert_scaling/meth166_run_native_table_cost.py) compiled with Clang `-O3 -mavx2 -mfma -std=c11` and ran five paired repetitions per cell after the METH-165 GPU work exited. The [machine result](meth166_native_table_cost_result.json), SHA-256 `736adcdfa2a754aff9d0862bee1d58eafcb82e607e482a853de68b10ecf99bcc`, retains compiler/input hashes, per-repetition times, page faults and checksums.

| Context cell | 75-entry median ms/token | 138-entry median ms/token | Ratio |
|---|---:|---:|---:|
| Old shared hit | 2.501 | 2.492 | 0.997× |
| Newly shared hit | 2.491 | 2.487 | 0.998× |
| Content miss | 2.498 | 2.484 | 0.995× |

All new-table medians satisfy <=3.5 ms/token and <=1.25× paired old-table gates. All timed page-fault deltas are zero after warmup. Maximum reported RSS is 534,659,072 bytes; the three-cell process took 23.213 seconds, within the five-minute/4-GiB stops. Ratios slightly below one are timing noise, not evidence that a larger table is intrinsically faster. This is one-thread warm actual-state **routing only**: METH-160 separately priced actual BF16 B-factor access for a quality-rejected bank. Neither result measures compact LUT factor arithmetic, cold DRAM traffic, full `engine.c` generation, a quality-valid E12,800 bank or the >=50 accepted-token/s goal.
