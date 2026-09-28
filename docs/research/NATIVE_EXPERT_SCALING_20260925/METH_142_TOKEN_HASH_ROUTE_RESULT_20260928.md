# METH-142: token-context hash passes six viewed load screens

**Decision: development load pass; fresh validation required.** A
calibration-free 64-bit hash of token, preceding token, window
position, source child and layer assigns one of ten grandchildren.
It passes every frozen load, coverage and within-parent share gate
on H0 raw and both previously viewed document manifests at 128 and
512-token windows. The worst candidate/control load ratio is
1.240×, the worst hot-parent grandchild share is 20.47%, and every
cell selects at least 6,992 grandchildren per layer. This is an
offline route-load result. No hash-routed full-model logits, native
CPU time, trained B specialization or new-source quality were measured.

The [protocol](METH_142_TOKEN_HASH_THIRD_ROUTE_PROTOCOL_20260928.md)
was committed at `3d2561d` before execution, and the
[implementation](../../../benchmarks/native_expert_scaling/meth142_token_hash_route_screen.py)
at `b1c90a9`. It binds the METH-126 exact centered BF16 E1280 bank
and checkpoint, checks exact BF16 teacher/control logits on eight
METH-121 prompts, captures the original E1280 child IDs, and computes
the grandchild ID by the frozen unsigned 64-bit formula. Every
grandchild divided by ten equals its source child. The golden vector
`(token=123, previous=45, position=67, child=89, layer=3)` produces
grandchild `899`, for later C/Python parity. The
[result](meth142_token_hash_route_screen_result.json), SHA-256
`e7a302671fc9e24290e39b8e0c1864c7b7b053f26057fa9c357d5f201e0decd3`,
retains all 144 layer/cell summaries and the exact constants. The
local RTX 3060 run took 252.219 seconds, 3.848 GB final process RSS
and 3.019 GB peak allocated GPU memory, below the frozen stops. No
T4 was used.

| Fixed source/window | Worst candidate/control load ratio | Worst hot-parent share | Minimum per-layer slot coverage |
|---|---:|---:|---:|
| H0 raw, 128 | 1.218× | 17.67% | 7,910 |
| H0 raw, 512 | 1.076× | 20.47% | 10,383 |
| METH-121 documents, 128 | 1.240× | 17.97% | 6,992 |
| METH-121 documents, 512 | 1.185× | 17.90% | 7,020 |
| METH-133 documents, 128 | 1.195× | 19.44% | 7,115 |
| METH-133 documents, 512 | 1.187× | 17.94% | 7,188 |

The frozen limits are <=1.25× in every layer/cell, <=25% hot-parent
share, and >=4,000 selected grandchildren. All pass. These sources
and windows were viewed in METH-141; the hash constants were fixed
before this run, but a fresh source-held-out route test is still
required before promoting the mechanism. Equal cloned B factors would
preserve the expert residual regardless of which grandchild is
selected, but this program calculated the hash outside the model and
did not verify full-model hash routing. Whether random partitions
can learn useful distinct B factors without harming quality is the
next scientific question. CPU hash cost and factor LUT arithmetic
also remain unmeasured.
