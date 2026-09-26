# METH-16: longer exact-donor residual-expert continuation

**Prospective protocol.** METH-15's 16-update RTX 3060 apparatus run
starts exactly at Qwen2.5-0.5B and ends at −0.008125 BPB on its small
internal slice, with all 24 routers receiving gradient and at least
125/128 expert output factors changed in each layer. One preliminary
invocation selected the GTX 1660 because CUDA enumeration differed
from `nvidia-smi`; it is recorded separately and does not satisfy the
predeclared RTX 3060 hardware gate. METH-16 asks whether the correctly
bound student keeps donor-relative quality after more exposure, whether
its router matters, and whether greedy continuation regresses.

## Fixed inputs and schedule

The source remains Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, safetensors SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`. Resume the METH-15 RTX 3060
optimizer/RNG checkpoint at 16 applied updates:
`results/native_expert_scaling/meth15_zero_residual_expert_smoke_rtx3060.pt`,
SHA-256
`cff4d2cbe4851d736f76d7f0838c42752a9d7bceec83054c34239d413d9a727e`.
The wrong-GPU checkpoint is excluded. Retain all 24 frozen donor FFNs,
E128 rank-8 residual experts/top-4, router, BF16/SDPA, 128-token
training windows, batch 1, four microbatches/update, AdamW expert
LR 1e-3/router LR 1e-4, zero weight decay, CE+0.1
KL(donor || student), and gradient norm clip 1.0. Continue the saved
RNG and optimizer **without restart** to total update 1024.

Train data are the same bound calibration ID array, SHA-256
`9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da`.
The diagnostic heldout was selected before this continuation:
`heldout` 16×512, seed 314159, ID SHA-256
`48bb4258c69a93dcc3e421444213dfa0eca38856dc9bdd24edf96c408310d1aa`,
33,374 scored bytes. It is disjoint from the METH-15 **windows** but
comes from the same document corpus used in earlier method selection;
this is a development gate, not a final independent test.

Measure donor and student BPB on identical IDs at update 16, then
student BPB at 256, 512, 768 and 1024. Save resumable adapter and
optimizer checkpoints at those boundaries. If any interim student
loss exceeds donor by **+0.05 BPB**, stop for gross quality failure.
No learning-rate or architecture changes are allowed after seeing the
curve. Record all applied updates, actual tokens, time, GPU/RSS peaks,
losses, and checkpoint hashes.

## Terminal joint quality checks

At update 1024, freeze the adapter. On the same 16-window slice:

1. Paired donor-relative BPB must be **≤+0.02**. Verify disabled
   experts reproduce the update-16 donor BPB within 1e-5.
2. Permute the trained router's **expert rows** with one seeded
   permutation per layer (seed 1616), leaving expert factors fixed.
   The trained route must beat that misaligned-route null by at least
   **0.002 BPB**. This is a routing-utility control, not a comparison
   to every possible router.
3. Generate greedily for 128 tokens from the first 128 IDs of each of
   the 16 heldout windows, once with experts disabled and once enabled.
   Count prompts whose generated continuation repeats a token 8-gram
   at least three times; the student count must be **≤donor+1**.
   Record generated IDs, early stops and counts.
4. At least 64 expert output factors in each layer must differ from
   zero. This guards against a nominal E128 bank with mostly dead slots.

The joint gate passes only if all four do. Even a pass leaves a dense
donor core active and uses a full-scan router; it does not prove
≥50 accepted tok/s in `engine.c`, large-E selection cost, 10B/100B
transfer, or fresh document/task quality. Those require native export,
core precision and independently trained larger-E cells.

Local RTX 3060 selected by name only; no T4. Stop at 35 minutes wall
time, 10.5 GiB peak allocated GPU memory, 20 GiB process RSS,
nonfinite update or OOM. The run is expected to take about 25 minutes
at METH-15's observed ~1.4–1.6 seconds/update; that is a projection,
not a measured continuation time.
