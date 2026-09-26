# METH-13: Qwen 0.5B progressive joint upcycle, CPU gate before GPU

**Prospective protocol.** This is the first transfer cell that will train
router and expert corrections together after METH-11/12's post-hoc failures.
The present rung only binds its pretrained donor, constructs a real E128
partition/router, measures step-zero damage and decides whether a bounded
GPU run is justified. A small-donor pass will not prove 10B/100B transfer.

## Distinct hypothesis and decision

Keep the Qwen2.5-0.5B donor's original attention, tokenizer and 24 FFNs.
Partition each FFN's 4,864 distinct neurons into **128 groups of 38**;
route **top-32**, so FFN work is 25% of dense at inference. Unlike
METH-11/12, the proposed next training rung will preserve the exact donor
at the beginning through a dense-to-sparse blend and jointly update
router, low-rank FFN corrections and a compact shared path with donor
distillation. The CPU rung asks whether the partition and input-only
router are viable enough to spend GPU time; it cannot demonstrate
training quality itself.

Source: `Qwen/Qwen2.5-0.5B`, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, `model.safetensors`
988,097,824 bytes, SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
Config: L24, D896, F4864, 14 attention heads/two KV heads, tied
151,936-row head. The local tokenizer fingerprint is
`4efeeb9382a77a06`, matching Qwen2.5-1.5B's, but all data IDs are
bound separately for this cell.

## Fixed CPU apparatus

1. Capture post-SwiGLU `z` from the intact fp32/eager donor on `calib`
   4×256, seed 42424, IDs SHA-256
   `40f40c2f29a7c0507bf41d1ae35d4342151870ebafd12f2d59d7e101997a7509`.
   Define a binary activity mask from each token's largest 10% of `|z|`
   (487 neurons, stable lower-neuron tie). Reuse
   `d0_layout.balanced_labels` with E128, dimension 64, seed 4242.
   Require exactly 38 neurons in every group. This is an exact donor
   channel partition, not 128 newly created experts.
2. Fit E23's existing sqrt-group-mass ridge router on `calib` 16×256,
   seed 42424, IDs SHA-256
   `b3c06b88955d9ab6593a52df2ff6005ccca2532ff69f844038b399a4fbe56ea8`.
   E23's fixed lambda is 0.01×mean diagonal of X'X. Score only the MLP
   input `x`; no post-SwiGLU oracle is available at native inference.
3. On `heldout` 4×256, seed 1234, IDs SHA-256
   `938abee0680d9c15df08d4f669e7a8dd9a158621779763464cc0d05a65055259`,
   report exact top-32 oracle group-mass recall, captured mass fraction
   and a seeded random-router null per layer. Fitted router must improve
   mean mass fraction by at least **0.15** over random and capture at
   least **0.60** of the oracle's mass on average, or stop before GPU.
4. Score the intact donor and hard top-32 routed donor weights, all 24
   layers, in CPU fp32 on `heldout` 2×512, seed 1234, IDs SHA-256
   `3808624ff6fc5cb8fa5911c9fbfb37a8bcbf4f08db83bb5b146887c267a9e62c`;
   4,396 scored bytes. The all-groups-active wrapper must agree with
   donor logits on the first 16 positions within 1e-4 absolute. The
   sparse arm must change logits. If paired step-zero loss exceeds
   **+0.40 BPB**, stop this geometry before GPU; the later training
   would otherwise need to heal more than twice METH-11's +0.20 screen.

These heldout slices share a corpus with earlier Qwen experiments and
are **internal selection instruments**, not fresh final quality data.
No rank, k, labels, router objective or gates may be retuned against
their outcome. Record full per-layer results and all source/file hashes.

## Cost and next-rung limits

At BF16 and hard top-32, shape arithmetic gives 78,446,592 active FFN
weights, 136,134,656 tied-head weights, 44,040,192 attention weights
and 2,752,512 exhaustive-router weights/token: 522,747,904 bytes/token
before norms, biases, KV, scales and code. At 40 GB/s this payload alone
takes 13.07 ms, leaving 6.93 ms of the 20 ms target. This is *not*
an actual export, LUT rate or quality pass. The current Qwen C donor
engine still stores its tied head fp32 and has no jointly trained
shared/expert format; both require explicit native work.

The CPU preflight has a 30-minute and 20-GiB process RAM stop on six
threads, no GPU. If it passes, freeze a second training protocol with
the exact dense-to-sparse schedule, loss, optimizer RAM, exposure per
expert, GPU budget, full-candidate BPB/generation stops and a native
export path before dispatch. A future 10× E experiment must add
distinct trained experts and measure routing quality and CPU LUT costs,
not duplicate these 128 donor groups.
