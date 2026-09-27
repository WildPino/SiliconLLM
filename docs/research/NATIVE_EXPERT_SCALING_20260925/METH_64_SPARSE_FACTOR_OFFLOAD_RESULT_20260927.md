# METH-64: selected expert factors can be trained through CPU offload

**Decision:** the selected-factor offload apparatus passes its frozen
numerical and memory checks. It makes an E1280 donor-training attempt
plausible on the occupied local GPU. It does not yet show a trained
E1280 model, learned expert utility, donor-relative quality, native LUT
traffic or end-to-end speed.

The [protocol](METH_64_SPARSE_FACTOR_OFFLOAD_PROTOCOL_20260927.md)
preceded the [runner](../../../benchmarks/donor_adaptation/s1/meth64_sparse_factor_offload.py),
SHA-256 `e8b15d8723ff3aa27ee845d3a883e646d5a05e636a53beef6c44512996eb72d4`.
The [raw result](meth64_sparse_factor_offload_result.json) SHA-256 is
`b9c1c5c85976eff397ab2f9514effcccf81ae95ed8e43ce268f2216016b05579`.

On the fixed 5×8, width-16, rank-3 case, CPU-offloaded and dense GPU
factors chose identical routes, produced exactly equal outputs and
exactly equal A/B gradients on all 19 nonzero rows. The CPU path
aggregates duplicate selected IDs; it does not create a full-bank
gradient. At 24 layers of E1280, width 896 and rank 8, all 768
product-key route cases matched exhaustive pair scoring. Each layer
had 113–125 unique selected gradient rows among the 32 fixed inputs.
All router and selected factor gradients were finite, and no factor
bank tensor resided on the GPU.

| E1280 synthetic backward measure | Value |
|---|---:|
| Selected A+B rows moved per direction, 24 layers × 32 inputs | 176,160,768 bytes |
| Peak GPU allocated | 385,184,256 bytes |
| End process RSS | 3,106,267,136 bytes |
| Cumulative time including Python/PyTorch start | 27.641 s |

This measures one synthetic backward, not a whole Qwen training
update or PCIe traffic counters. The factor copies travel CPU→GPU and
their gradients GPU→CPU; the measured byte figure is the nominal
selected tensor payload before duplicates. Sparse optimizer moments
and donor activations are absent. The next test must use the bound
pretrained Instruct donor for actual gradients and quantify the total
update memory/time before starting a longer E1280 training ladder.
