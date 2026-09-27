# METH-65: one pretrained Instruct update reaches E1280 factors

**Decision:** pass the one-update apparatus gate. The CPU-resident
E1280 factors receive nonzero gradients and an actual sparse row-wise
Adam update through the complete frozen Qwen2.5-0.5B-Instruct donor.
This is not a quality-valid E1280 checkpoint; no independent
generation, task or semantic endpoint was opened.

The [protocol](METH_65_E1280_REAL_DONOR_UPDATE_PROTOCOL_20260927.md)
and [runner](../../../benchmarks/donor_adaptation/s1/meth65_e1280_real_donor_update.py)
were committed before the successful run. The first attempt stopped
before backward because autograd was disabled; the apparatus-only
repair explicitly enabled and asserted it. No training decision or
external metric was seen in that attempt. The successful runner
SHA-256 is `9cd776aaeda3c506191931dfd1607ee101989752193fa09bd3d653ea7c7d6566`;
its [raw result](meth65_e1280_real_donor_update_result.json) SHA-256
is `f91b456b5e4c661fc470981f888740886dc70174679f5c0dcce2b227f98ca885`.

The bound Instruct donor and student begin with exactly identical
logits on both frozen 16-token raw and chat prefixes. One 128-token
raw and one existing donor-continuation chat training item produce
finite objectives 3.034261 and 0.731219 respectively; initial
donor KL is zero by construction. All 24 layers have nonzero B
gradient norms (minimum 0.001258), with 244–471 nonzero B-gradient
rows/layer. Initial A/router gradients are zero as expected while B
is zero. After the sparse update, 244–471 B slots/layer have changed,
and subsequent student logits remain finite. The row-wise optimizer
updates 9,076 factor rows and holds 520,454,144 bytes of CPU moment
state. Peak allocated GPU is 3,281,052,160 bytes, end process RSS
5,292,175,360 bytes, elapsed 18.203 seconds on the local RTX 3060.

The result shows physical feasibility of the first update with 10×
as many independently stored experts as E128. It does not show that
all 1,280 learn useful distinct roles, or that routing retains donor
quality as choices grow. A controlled E128/E1280 training ladder with
held-out quality, route-load and semantic checks is required. The
native C LUT and full-model accepted-token rate remain separate.
