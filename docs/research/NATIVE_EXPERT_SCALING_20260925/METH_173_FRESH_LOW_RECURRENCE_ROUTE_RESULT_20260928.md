# METH-173: threshold-8 route passes a new source and two lengths

**Decision:** the fixed METH-172 route passes all frozen prospective source/length gates on previously unused PG19 shard 12, including the train+new-source pool that exposed earlier failures. The route is now eligible for the separate native C parity/cost gate. This result does not yet show useful learned E12,800 specialists.

The [input manifest](meth173_fresh_recurrence_route_manifest.json), SHA-256 `6ebfc36fd452f2a20cfcc9e8210505e36b725ef95992949655c19f3c94eaee20`, freezes 512 chat, 512 raw and 24 distinct document rows. The greedy BF16 teacher merge SHA-256 is `c1ce0318ee62484a650d4f7825c0fc417aa59c2606c73226c54eef0a47cf6e53`: 512 responses, 31,287 continuation tokens, 502 EOS terminations and two triple-repeat 8-grams. Generation took 1,352.812 s with 1.014 GB peak allocated GPU memory. Eight-prompt teacher/control BF16 parity passes in the route run.

| Cell | Input tokens | Minimum content coverage/layer | Maximum nine-way load ratio | Worst hot-parent share | Structural traffic maximum |
|---|---:|---:|---:|---:|---:|
| Raw | 65,024 | 7,356 | 1.097 | 19.38% | 2.26% |
| Chat | 112,046 | 7,474 | 1.087 | 19.25% | 31.98% |
| Documents, width 128 | 24,576 | 5,679 | 1.207 | 17.60% | 1.93% |
| Documents, width 512 | 24,576 | 5,328 | 1.186 | 20.31% | 0.61% |
| New raw+chat pool | 177,070 | 8,540 | 1.066 | 18.71% | 21.06% |
| Training+new raw/chat pool | 1,507,166 | 11,029 | 1.025 | **20.32% PASS** | 21.05% |

The composed pool's minimum active content median is 61 selections and maximum fraction of content slots below 32 selections is 39.05%. Every layer has a >=250-selection hot parent, and the worst grandchild share remains below 25%. The candidate adds 1,389/65,024 raw, 2,401/112,046 chat, 444/24,576 width-128 document and 142/24,576 width-512 document input positions to the structural route versus the fixed 138-tuple-plus-delimiters baseline; all are below the frozen 5% cap. The chat-specific 35% structural cap and 25% caps elsewhere pass. Source children and four choices/token are identical between route variants.

All 24-layer 12,800-slot all/content histograms for four cells, baseline comparisons and two pools are preserved in `meth173_fresh_recurrence_route_result.zip`, 6,042,616 bytes, SHA-256 `9083f66384bca49bd9506a4e4f5e1da71fa14224c50857749ce13357094ca496`. Its verified single JSON member is 99,419,618 bytes, SHA-256 `8cf875ade5a2996e7f2e9db352f5dde1afb65efe537e0038a0e61033c97770e3`. The local RTX 3060 source-ordered route run took 277.250 s, with 2.945 GB peak allocated GPU memory and 4.710 GB RSS. PG19 shard 9 remains untouched for a later quality audit.
