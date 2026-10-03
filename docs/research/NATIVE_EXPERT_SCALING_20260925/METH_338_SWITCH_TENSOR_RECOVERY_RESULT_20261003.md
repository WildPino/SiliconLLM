# METH-338: complete immutable compact artifact verified

Freeze `a653f43`; raw committed `08be22e`. Authoritative exec78511 exit0.
Raw SHA256 `19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c`.
All five prospectively fixed gates PASS. MAIN1032.625s after imports;
maximum checked RSS25,245,736,960B, end15,248,433,152B, below32GiB.

## Actual result and reusable output

ALL6392 original tensor names/shapes/finite F32 bytes freshly match327.
Every original coefficient is compared with the unchanged335 target codes,
original F32 controls, scales and zero padding. All12 banks each retain256
byte-distinct WI/WO code-and-scale tuples. These hashes prove distinct stored
parameters, not useful extra functions or quality.

Immutable payload `results/native_expert_scaling/meth335_switch_w8a8_export/weights.bin`:
14,818,015,744B, SHA256
`e0e5a940b0150b78d0080815a1fddd2a6b50f011351012ed5b4a48a88ba49056`.
New manifest `results/native_expert_scaling/meth338_switch_tensor_recovery/manifest.bin`:
SHA256 `9c5be95504291cf9a71b087f714fc0b474ac6948389647be93919f611117fd81`.
6313 I8 tensor names and79 original F32 tensor names. The three F32 lookup
aliases share ONE physical payload copy; the separately encoded I8 head
comes from the byte-identical tied original source. Original architectural
unique parameters14,664,154,368. No new payload bytes written in338.

Original335 quantization, scalar rounding controls and exact337 optimized
quantizer equivalence PASS. Actual safe-subnormal values bypassed in338:0.
No evidence that subnormal handling caused prior conversion latency.
Read each source shard in physical tensor order, immutable whole payload
once into RAM; hash those actual RAM bytes and compare all reconstructed
target segments there. Shard checks29.907/84.562/166.422/178.594/174.609/155.781s.

## Explicit identity scope and retained costs

Fresh complete canonical coefficient identity plus pinned config/tokenizer
metadata; no fresh full ZIP-envelope checksum in338. Original326/335 archive
provenance remains. This is an explicit new verification policy, frozen before
observations. Original335 and33720min failures remain preserved; their MAIN
costs1200.015s and1200.093s are additional conversion/verification costs.
None of these costs is an inference rate.

## Decision

Complete artifact licenses the separately frozen336 native I32/scaling
contract on these exact bytes. Full original-donor NEW untouched quality,
actual complete CPU cost, useful RAM-scale n, LUT/routing/real DRAM behavior,
multiple families and50 accepted tokens/s on the SAME artifact remain open.
No training or capacity duplication performed.

Reproduce with the command in
[METH-338 protocol](METH_338_SWITCH_TENSOR_RECOVERY_PROTOCOL_20261003.md).
