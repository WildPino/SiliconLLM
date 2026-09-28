# METH-135: a ten-way third routing tier passes the actual-state CPU cost gate

**Decision.** The untrained E1280→E12800 third routing tier is
affordable as a CPU component on this 24-layer Qwen geometry: five
paired single-thread repetitions give a **2.002 versus 1.483
ms/token-equivalent** median for 12,800 versus 1,280 candidate slots,
a **1.350×** ratio. Both frozen limits (<=3.0 ms and <=2.0×) pass.
This licenses a separately registered learned-grandchild training and
quality experiment. It is not evidence that the 12,800 slots have
distinct useful capacity, that a quality-valid LUT factor format
exists, or that the full model reaches 50 accepted tok/s.

The [protocol](METH_135_THIRD_TIER_CPU_ROUTE_PROTOCOL_20260928.md)
was committed at `b55b989` before export/timing. The
[sidecar exporter](../../../benchmarks/native_expert_scaling/meth135_export_third_router.py)
and [native checker](../../../benchmarks/native_expert_scaling/meth135_third_tier_cpu_route.c)
were committed at `9bb05f8`; the addressed-byte label was clarified
at `7ab3fc8` before the retained timing run. The input METH-126 exact
BF16 bank SHA-256 is
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`;
the 256 actual E1280 vectors are
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
The seeded sidecar is 42,074,144 bytes, SHA-256
`692db3dd1a647debb051ef66612263b5e16ef27c73978e9abd3beb6b3229abc9`.
Its [export ledger](meth135_third_router_export.json) verifies every
projection/key record after writing. The checker is built with
`clang -O3 -std=c11 -Wall -Wextra -lpsapi` on an AMD Ryzen 5 3600X.
No T4 or GPU was used for this timing.

On 6,144 real fixed states across 24 layers, the first two route
tiers and their four BF16 gates remain identical; every grandchild ID
divided by ten yields its selected E1280 child. Each grandchild
inherits that exact child's B factor for this check, and the maximum
factor-residual difference is zero. There are 8,145 distinct selected
E1280 children summed across layers and 13,525 distinct selected
grandchild IDs, with per-layer ranges 252–402 and 467–641. These are
routes from seeded **untrained** keys and reused states; they are not
learned specialization or new-source quality evidence.

| Paired repetition | E1280 route ms/token | E12800 route ms/token |
|---:|---:|---:|
| 0 | 1.504187 | 2.041312 |
| 1 | 1.485835 | 2.050866 |
| 2 | 1.476652 | 2.002184 |
| 3 | 1.483383 | 1.995982 |
| 4 | 1.478941 | 1.993872 |
| **Median** | **1.483383** | **2.002184** |

The [native log](meth135_native_raw.log), SHA-256
`2d61c5ee9160cfa508a88833ba64b53ac6528a6e25bec7776139fd968b35573f`,
retains all paired times and per-layer coverage. The new tier
nominally addresses 2,875,392 bytes/token across 24 projections and
40 selected keys per layer; it stores 42.1 MB of projections/keys.
The process used 575,959,040 bytes RSS and 6.048 seconds elapsed,
below both resource stops. Deliberately corrupting the sidecar magic
returned exit 2 with `invalid third-tier sidecar header`; the exact
sidecar SHA was restored. The [negative log](meth135_bad_magic.log)
records the result.

The repeated 256-state sequence can reuse router data in cache, and
this checker does not materialize a 4.40 GB E12800 BF16 factor bank.
It therefore does not measure cold random DRAM traffic, a compact
LUT factor kernel at E12800, or full `engine.c` latency. METH-134's
CPU-master path and this routing result jointly make a *testable*
E12800 training rung; donor-relative held-out quality, load balance,
semantic grounding, memory/optimizer cost and native full-model rate
remain prerequisites. The eventual 10B/100B donor-scale claim also
requires new source-model transfer evidence.
