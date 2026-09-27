# METH-84: one affordable BF16 FFN layer cannot rescue grouped Q4

**Decision.** On the already viewed METH-83 prompts, restoring any
single full FFN layer from grouped Q4 to donor BF16 remains far below
the 95% donor top-1 screen. The best is layer 2 at 3,099/4,199 =
73.803%, versus the all-Q4 FFN baseline 2,875/4,199 = 68.469%.
The single-layer correction fits the arithmetic 560 MB allowance,
but is insufficient under the tested rule. No arm is promoted.

The [protocol](METH_84_SINGLE_FFN_LAYER_RESCUE_PROTOCOL_20260927.md)
was committed at `0ebe156` before scoring. A one-character transcription
error in the bound METH-83 result hash stopped the first apparatus
attempt before model inference; it was corrected at `983040a`.
The [runner](../../../benchmarks/donor_adaptation/s1/meth84_single_ffn_layer_rescue.py)
then reproduced the BF16+E128 control at 4,045/4,199 and all-Q4
FFN baseline at 2,875/4,199 before testing the 24 layers in order.
It restored each layer's original gate/up/down matrices, scored the
same prompts, restored that layer's saved Q4 reconstruction and
verified the final all-Q4 baseline again. The
[machine result](meth84_single_ffn_layer_rescue_result.json) contains
all 24 layer scores and per-prompt counts; no test or external set
was queried.

| Viewed-set one-layer arm | Donor top-1 matches / 4,199 | Change vs all-Q4 | Ideal addressed bytes/token |
|---|---:|---:|---:|
| All grouped-Q4 FFN | 2,875 = 68.469% | — | 535,739,904 |
| BF16 FFN layer 2, best | 3,099 = 73.803% | +224 | 554,942,976 |
| BF16 FFN layer 23 | 3,068 = 73.065% | +193 | 554,942,976 |
| BF16 FFN layer 0 | 3,015 = 71.803% | +140 | 554,942,976 |
| BF16+E128 full core control | 4,045 = 96.332% | +1,170 | above this single-layer map |

Restoring layer 22 yields only 2,829 matches, 46 fewer than the
all-Q4 baseline. This shows that layer effects can be non-additive;
the individual gains cannot simply be summed to predict a multi-layer
map. One layer adds exactly 19,203,072 ideal bytes, leaving a
554,942,976-byte total and 5,057,024 bytes under the allotment.
The best observed arm remains 891 matches short of the ≥95% threshold
(at least 3,990/4,199).

The completed diagnostic took 47.125 seconds, peaked at 1.283 GB
allocated GPU and 3.523 GB RSS; no T4 was used. Because the prompts
were already viewed in METH-83, these scores locate error under the
tested representation and are not independent quality evidence.
The METH-82 packed export remains technically valid, while METH-83's
quality failure still blocks native integration. A new FFN rule must
reduce distributed quantization damage at similar or lower active
bytes, then pass fresh source-disjoint quality before CPU LUT and
full-artifact rate can be interpreted.
