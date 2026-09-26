# METH-15: exact-donor residual experts, local GPU apparatus gate

**Prospective protocol.** METH-13/14 reject routing donor FFN channels
away: the fitted E128/top-32 model loses +1.360474 BPB, and even a
current-activation local oracle loses +0.775726. This cell changes the
target geometry. It retains Qwen2.5-0.5B's pretrained FFN as a compact
shared core and **adds** 128 independently trainable rank-8 residual
experts at each of 24 layers. Four experts contribute per token. Zero
output factors make the complete student exactly the donor at step zero.
The apparatus gate asks whether all layers can be trained end to end on
the local RTX 3060 with actual sparse selection and a trainable router.
It does not claim a quality or native-rate pass.

## Bound source and data

- Donor `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`,
  safetensors SHA-256
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
  tokenizer fingerprint `4efeeb9382a77a06`. L24/D896/F4864,
  tied 151,936-row embedding/head. Base weights frozen.
- Train IDs: existing calibration `s1/results/h0/h0_train.npz`,
  array `(31250,512)` int32, NPZ SHA-256
  `0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9`,
  raw ID SHA-256
  `9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da`.
  The tokenizer fingerprint matches the prior Qwen1.5B source.
- Diagnostic heldout: corpus `heldout`, four 256-token windows,
  seed 271828, ID SHA-256
  `c5345782fc0b08ef3212403878a33960ae0da236c999272e79e418f29c277d3c`,
  4,588 scored bytes. This shares a document corpus with prior
  selection and is **internal only**; final quality needs fresh documents.

## Exact architecture and cost hypothesis

For each FFN input `x`, calculate the frozen donor FFN `f(x)`. A
trainable router `R_l ∈ R^{128×896}` chooses top four. For selected
expert `e`, add `B_{l,e} SiLU(A_{l,e} x)`, with
`A ∈ R^{8×896}` and `B ∈ R^{896×8}`; selected softmax gates sum to
one. All `B` factors begin at zero and `A` factors are seeded random.
At step zero, `f(x)+0` must match the intact donor logits within 1e-3
absolute on the first 16 positions in BF16. Expert factors and router
must both receive nonzero gradients by the second applied update.

This stores 44,040,192 expert factors and 2,752,512 router weights,
plus the donor. The four selected rank-8 experts address 1,376,256
weights or 2,752,512 BF16 bytes/token across all layers; a full fp32 router addresses
11,010,048 bytes/token. The shared donor remains an active cost and
requires a quality-preserving low-bit representation before the 50
token/s claim. E scaling also needs bounded candidate routing at much
larger pools. These are shape counts, not a measured C result. No
existing `engine.c` artifact represents this module yet.

## Frozen 16-update apparatus run

Use the local RTX 3060 selected by its exact device name, BF16, SDPA,
gradient checkpointing,
seed 5151, 16 AdamW updates, sequence length 128, batch 1,
four microbatches/update. Sample a training row and a 128-token offset
uniformly with the seeded generator at each microbatch. Only factors
and router are trainable. Loss is student next-token cross entropy
plus 0.1 times full-vocabulary KL(frozen donor || student) at
temperature one; compute donor logits with residuals disabled on the
same IDs. Expert LR 1e-3, router LR 1e-4, zero weight decay,
gradient norm clip 1.0. Evaluate donor and student BPB on the fixed
diagnostic slice at step zero and after update 16.

Limits: 20 minutes wall time, 10.5 GiB peak allocated GPU memory,
20 GiB process RSS, no nonfinite loss/gradient. Stop and record failure
on OOM, identity failure, missing parameter gradient, or either limit.
The apparatus passes only if 16 updates complete, both expert-output
and router gradients are nonzero after the first update, at least 16
expert slots per layer have a nonzero output factor, and heldout BPB
is no more than **+0.10** worse than the disabled-expert donor. The
quality threshold is a gross smoke stop, not promotion. Record actual
timing, peak memory, gradient and slot exposure, output checkpoint
hash and all controls. No T4 work is requested.

If the apparatus passes, freeze a separate longer-run quality and
generation protocol before training further. A larger-E comparison
must train distinct experts and match tokens/exposure controls; it
cannot be inferred from this one smoke or from duplicated weights.
