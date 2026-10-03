# METH-341: exact batched prefill and complete CPU cost-margin PASS

Freeze429307c; raw commitdca8d18, authoritative exec52292 exit0.
Raw SHA `a73b41e01867915f0b0c5b2e1b1add2000bd97acdda329c1edecf98f9db8607c`.
MAIN51.329s; max checked combined RSS927,547,392B. ALL five prospective gates
PASS. All original340 failures retained, no old threshold/gate changed.

## Exact implementation and measured cost

Only encoder Q/K/V and cross-K/V prefill execution batches actual source
vectors by row. Same activation codes/I32 dots/scales/F32 returns, unchanged
decoder/router/feed/cache/head/weights/precision. Eight336 bridges and four
complete3409/64-source32-position bridges are SHA exact for threads1/6;
ALL measured/warmup/profile complete outputs exact340. No fit or extra banks.

| Source tokens | Threads | Median decode ms/position | Median full incl prefill ms/position | Decode repetition max/min |
| --- | --- | --- | --- | --- |
| 9 | 1 comparison | 9.0846 | 10.8371 | 1.0591 |
| 64 | 6 primary | 8.4420 | 17.5105 | 1.0478 |
| 9 | 6 primary | 6.6523 | 7.6428 | 1.0385 |
| 64 | 1 comparison | 10.3538 | 24.9708 | 1.1935 |

Both thread6 medians<=20ms INCLUDING encoder/crossKV amortized over32 forced
positions, all thread6 repetition ratios<=1.10. Thread1 remains comparison,
not silently promoted; source64t1 exceeds20ms/1.10. New fixed experiment pass
does not prove batching caused the earlier repeat drift or a paired speedup.
Primary complete64t6 median improves versus prior independent experiment,
but no statistically paired causal speedup claim.

Separate64t6 profile prefill CORE112.798ms/EXPERT64.704ms/ROUTER12.989ms;
profile encoder276.419ms/crossKV16.514ms. Each batch uses one matrix clock;
profile includes overhead and is not primary performance. Logical decoder
matrix addressed bytes unchanged129,017,344/position. Actual14.818GB mapped
bank/hash unchanged; decoder unions identical340. Physical DRAM/cache behavior
is not measured by these logical counters.

## Decision and exact next step

CPU cost-margin PASS licenses NEW prospectively selected/excluded source
cohort and original UNMODIFIED donor paired full quality evaluation. Native
341 SAMEartifact is the quality target. Need actual original reconstruction/
prediction, autoregressive span generation and pertinent reconstruction task,
then accepted batch1 generation timing with TTFT/context separately stated.
Do not call forced positions accepted generated tokens or infer preserved
quality from336 arithmetic identity. Its consumed original-donor differences
remain. Useful higher-n/LUT/physical DRAM/cross-family/~100B still unqualified.

Original defaults byte-exact0ff9705. Reproduction precision/commands/gates in
[341 protocol](METH_341_SWITCH_BATCHED_PREFILL_PROTOCOL_20261003.md).
