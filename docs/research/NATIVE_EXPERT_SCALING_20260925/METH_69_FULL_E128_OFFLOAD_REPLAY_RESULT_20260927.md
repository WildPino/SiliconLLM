# METH-69: full donor replay diverges after three matching updates

**Decision:** fail the precommitted exact replay gate. The CPU-offloaded
E128 run reproduces METH-55's four microbatch losses exactly for its
first three updates, then first differs at the third microbatch of
update four. Small prior numerical differences can be amplified by
top-four routing, but the precise first route/weight cause has not
been established. The final offloaded checkpoint does not reproduce
the frozen METH-55 weights or raw BPB.

The [protocol](METH_69_FULL_E128_OFFLOAD_REPLAY_PROTOCOL_20260927.md)
and [runner](../../../benchmarks/donor_adaptation/s1/meth69_full_e128_offload_replay.py)
bound the original donor, data, seed and 16-update sample schedule.
The [raw result](meth69_full_e128_offload_replay_result.json), SHA-256
`5a34e7f1346fbfd108288634998319b11e0a40d01102d51473606faae88ea7e6`,
contains every update's loss and gradient norm versus the original.

| Terminal measure | Original METH-55 | CPU-offloaded replay |
|---|---:|---:|
| Prompt top-1 on viewed METH-55 dev | 3,658/3,811 = 95.985% | 3,660/3,811 = 96.038% |
| Held-out raw BPB | 0.9697036 | 0.9705151 |
| Max absolute A/B/router weight error vs original | — | 0.004041 / 0.004034 / 0.000433 |
| Peak GPU allocation | 2.582 GB original | 3.659 GB replay |
| End process RSS | 3.207 GB original | 3.744 GB replay |

The replay's E128 top-1 clears the old 95% screen, but that is on a
viewed set and cannot override the exact replay failure. The original
per-update gradient norm and replay norm agree closely through update
three; their first objective difference occurs in update four's third
microbatch. The locally installed PyTorch AdamW implementation updates
moments with `lerp_` and applies a bias-corrected `addcdiv_` step,
whereas METH-68's CPU implementation uses algebraically equivalent
operations with different rounding. This is a plausible source of the
tiny early weight differences that could alter close route choices;
it is an inference, not yet a measured attribution.

No E1280 model is promoted by this result. A quality-directed next
experiment can use GPU-paged dense AdamW to reproduce the reference
update rule while keeping the full bank and state in RAM, then compare
E128/E1280 on new development and external sources. A strict bitwise
replay is an apparatus target; final success still requires distinct
learned large-E quality and the same-artifact native rate.
