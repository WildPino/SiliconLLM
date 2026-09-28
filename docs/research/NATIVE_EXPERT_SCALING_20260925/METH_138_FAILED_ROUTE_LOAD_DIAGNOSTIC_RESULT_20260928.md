# METH-138: third-tier choices amplify parent hotspots

**Diagnostic conclusion:** the failed METH-136 candidate has many
BF16-distinct and selected grandchildren, but seeded argmax keys
concentrate traffic sharply within already popular E1280 children.
This does not repair METH-136's frozen route-load failure.

The [protocol](METH_138_FAILED_ROUTE_LOAD_DIAGNOSTIC_PROTOCOL_20260928.md)
was committed at `78a3717` before replay, and the
[diagnostic](../../../benchmarks/native_expert_scaling/meth138_failed_route_load_diagnostic.py)
at `f2ec205`. It verified the exported BF16 bank hashes, headers and
sizes, then ran every METH-136 raw/chat draw through each **final**
artifact with fixed donor/router weights. It made no quality inference
or optimizer update. The [result](meth138_failed_route_load_diagnostic_result.json),
SHA-256 `57ee6d02f30b1621b7d57c3633633af45093ec8cc02396c0e7c4473f2b3e978e`,
includes all 24 layer summaries and the input report hashes. The
replay took 209.969 seconds, 22.728 GB process RSS and 1.347 GB peak
allocated GPU memory on the local RTX 3060, below its frozen stops.
No T4 was used.

| Measure on final-state replay | Continued E1280 | Trained E12800 |
|---|---:|---:|
| Selections per layer | 346,428 | 346,428 |
| Selected-slot coverage per layer | 1,183–1,278 | 7,996–11,482 |
| BF16-distinct rows per layer | 1,183–1,278 | 11,820–12,780 |
| Maximum/mean slot load, largest layer | 56.417× | 287.349× |
| Layers over the METH-136 limit of 50× | 2/24 | 24/24 |
| Maximum/mean after aggregating candidate to source child | — | 56.516× |
| Selected B transfer bytes over replay | 238,386,806,784 | 238,386,806,784 |

In layer 17, the candidate's hottest grandchild (slot 9834) receives
7,777 of the 15,296 selections for source child 983: 50.8% of that
parent's traffic lands on one of ten grandchildren. A uniform split
would be 10%. The same source child has 15,269 control selections.
The maximum load is 287.349× the mean across 12,800 slots, while
aggregating candidate slots back to 1,280 parents gives 56.516×,
close to the 56.417× control. Layer 0 is similarly concentrated at
286.28×; every candidate layer exceeds 50×. The issue is therefore
mostly the third-tier partition of parent traffic, although the
first two tiers also contain two baseline hotspots above 50×.

Post-BF16 candidate sibling means differ from original E1280 source
B values by at most `4.883e-5` absolute across layers. This is the
rounding residual after METH-136's FP32 centering, not a held-out
function-parity or quality claim. The large number of different rows
does not establish useful specialization: most slots receive much
less training traffic than the few hotspots.

This replay uses final B values on all 256 draws. METH-136's failed
gate accumulated counts as B changed during training; its precise
candidate train-time skew was not saved. The control replay maximum
matches its recorded train-time maximum, but that agreement does not
make the candidate replay an exact train-time measure. The frozen
METH-136 decision stays **rejected**. A next experiment should allocate
and choose grandchildren with a measured traffic-balancing mechanism,
then train from the same quality-valid base and reapply untouched
quality and CPU cost gates. Lowering the failed 50× threshold after
seeing this result would not validate the current bank.
