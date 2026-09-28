# METH-143: fixed token hash passes new-source route-load audit

**Decision: fresh route-load pass; full-model hash routing and native CPU cost pending.** The unchanged METH-142 token-context hash passes every frozen load, coverage and within-parent share gate on 24 new source/fragment-disjoint documents. This is an offline grandchild assignment on original E1280 routes. It does not establish useful learned E12800 experts, full-model hash-routed parity, native CPU speed or quality.

The [protocol](METH_143_FRESH_HASH_ROUTE_AUDIT_PROTOCOL_20260928.md) was committed at `0a15cc2`. The [selector](../../../benchmarks/native_expert_scaling/meth143_select_fresh_hash_manifest.py) and [manifest](meth143_fresh_hash_route_manifest.json) were committed at `68030f2` and `46d867e` before route inference; the manifest SHA-256 is `e6c4fb6a92929eb29332c55a0887adb6627c9528f80b21c09f7759139070ccff`. The [evaluator](../../../benchmarks/native_expert_scaling/meth143_fresh_hash_route_audit.py) was committed at `a2519dc` and binds the [METH-142 result](meth142_token_hash_route_screen_result.json) by SHA-256 `e7a302671fc9e24290e39b8e0c1864c7b7b053f26057fa9c357d5f201e0decd3`. The [result](meth143_fresh_hash_route_audit_result.json) has SHA-256 `d13aeb90b8211fc5be6a1289f0209d1f846691b9be7f4621a06e0845538d749f` and retains every layer count, top slot and gate. The unchanged golden tuple `(123,45,67,89,3)` selects grandchild `899`.

| Window width | Tokens | Worst candidate/control load ratio | Worst hot-parent grandchild share | Minimum per-layer coverage |
|---|---:|---:|---:|---:|
| 128 | 29,751 | 1.1732× | 18.60% | 7,211 |
| 512 | 29,751 | 1.1915× | 17.14% | 7,182 |

The frozen limits are at most 1.25× load in every layer and width, at most 25% of any parent's selections for a grandchild when the parent receives at least 250, and at least 4,000 selected grandchildren per layer. All pass. The original E1280 teacher/control BF16 logits match exactly on eight bound prompts. The hash was applied **after** the original full-model route, so these logits are not a parity test of a hash-routed model. The 24 documents use eight code, eight technical and eight book sources; PG19 shard 6 remains reserved for later quality evaluation.

The local RTX 3060 run took 71.359 seconds, with 4.648 GB final process RSS and 3.019 GB peak allocated GPU memory. No T4 was used. The next frozen gate is exact cloned full-model BF16 logit parity with the hash inside the route, plus native C route cost and a C/Python hash golden-vector check. Only a pass there licenses matched B-only training and untouched semantic quality. Balanced traffic alone is insufficient for the user's useful-expert ladder.
