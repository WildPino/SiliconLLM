# METH-188: the grouped-Q6 FFNs dominate the compact-core ranking loss

The [frozen protocol](METH_188_CORE_ORGAN_ATTRIBUTION_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth188_core_organ_attribution.py)
were committed at `e3efe4e` before execution. The run used only the 24
already viewed METH-121 sources, the exact METH-186 stored core, and the
same centered E1280 bank. Its BF16+E1280 and full R8-head/Q6-FFN arms
reproduced every METH-187 document negative log likelihood and prompt
match count exactly.

| E1280 arm | Pooled BPB | Donor prompt top-1 | Loss vs BF16+E1280 |
|---|---:|---:|---:|
| BF16 head + FFNs | 1.184887 | 94.504% | 0.000 points |
| R8 tied head only | 1.185061 | 94.215% | 0.289 points |
| Grouped-Q6 FFNs only | 1.187518 | 90.142% | 4.361 points |
| R8 tied head + grouped-Q6 FFNs | 1.187487 | 89.186% | 5.318 points |

The FFN-only arm accounts for 82.0% of the full ranking loss and the
head-only arm 5.4%; their interaction adds 0.668 points. The frozen
pooled rule therefore selects **FFN-first correction**. Category detail
is less uniform: code and prose trigger the protocol's joint label,
while technical follows FFN-first. This calls for checking every
category after FFN correction and retaining a tied-head correction as
a possible later step. It does not justify dropping the head from the
final design by assumption.

The [raw result](meth188_core_organ_attribution_result.json), SHA-256
`5b655cbdfa895287169c8f60125c57f0ab9b2fa87abb45c65bef0c89ed97ab71`,
contains all per-source values and costs. Runtime was 51.813 s, peak
GPU allocation 4.220 GB and ending RSS 2.009 GB on the local RTX 3060.
No T4 or new source was used. This is attribution, not a quality pass:
the uncorrected artifact still fails METH-187 ranking gates and has
no native same-artifact rate test.

The next fixed intervention is a trainable, stored rank-64 correction
for each of the 72 grouped-Q6 FFN projections, with the R8 tied head
and E1280 bank frozen. Its BF16 factors add 53,084,160 ideal addressed
bytes/token, bringing the existing 481,534,976-byte core-plus-router
ledger to 534,619,136 bytes/token, under the 560 MB design allotment
by 25,380,864 bytes. This arithmetic is not a bandwidth measurement.
