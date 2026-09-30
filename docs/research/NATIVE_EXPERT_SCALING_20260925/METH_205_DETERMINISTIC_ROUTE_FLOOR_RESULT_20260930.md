# METH-205: soft-count calibration does not balance deployed argmax

The [protocol](METH_205_DETERMINISTIC_ROUTE_FLOOR_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth205_deterministic_route_floor.py)
were frozen at `b83aefc`. The exact METH-204 router and first 1,024
METH-175 paired raw/chat draws were replayed on the pinned E1280
control. Eight-prompt BF16 donor/control parity was exact; both cells'
totals, route gates and metrics reproduced METH-204 within the frozen
1e-6 numerical reconciliation tolerance.

No hot parent (>=250 content selections) has an exact normalized
FP32 q-vector group above 25%. Worst duplicate shares are 2.344% raw,
1.931% chat and 1.701% pooled. Thus this exact-input recurrence bound
does not explain the current load failure. It does not establish
that an arbitrary deterministic partition will generalize.

| Cell | Hot parent/layer pairs | Pairs above 25% hard share | Pairs above 25% soft share | Worst hard share | Worst soft share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 10,356 | 290 | 94 | 87.244% | 40.791% |
| Chat | 11,653 | 123 | 11 | 43.411% | 41.078% |
| Pooled | 16,029 | 67 | 0 | 73.769% | 11.277% |

The per-layer maximum pooled soft share ranges only 11.111–11.277%
at the fitted temperature 0.05, close to the 1/9 target. Yet deployed
argmax yields a 17.406–73.769% maximum across those same layers.
The fitting surrogate is therefore well balanced on its pooled
objective while its hard inference decisions are not. The separate
raw/chat soft shares also show a domain mixture effect; a lower
temperature alone must still pass each cell, not only pooled load.

The [raw result](meth205_deterministic_route_floor_result.json), SHA-256
`f297107e589c5152e27a6f4facb86f5e5ed3129acd71ede3ff38f73434055745`,
contains every hot-parent record for raw/chat/pooled, exact vector
counts, and soft/hard shares. Runtime was 534.703 s on the local
RTX 3060, with 2.945 GB peak GPU allocation and 8.277 GB ending RSS.
No new source, T4 or B training was used. A synthetic 300-identical-
vector check verified a 100% exact-state floor and hard share with
balanced soft counts; its first startup was interrupted, then the
instrumented six-thread check completed.

**Decision:** calibrate against an annealed soft surrogate approaching
hard selection before changing router inputs or training another
expert bank. Preserve the shared keys and start from METH-204 biases;
freeze the schedule and require the original separate raw/chat fit,
reserved and source-transfer gates. This diagnoses routing mechanics,
not useful E12800 quality, native cost or large-donor transfer.
