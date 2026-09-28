# METH-136: matched E1280/E12800 sparse training from a pretrained donor

## Uncertainty and decision

METH-121/123 gives donor-relative quality for the centered, learned
E1280 BF16 adapter. METH-134 verifies selected-row CPU-master
gradients for a cloned E12800 third tier, and METH-135 measures a
1.350× CPU route-cost ratio on actual states. Neither establishes that
the extra slots learn useful distinct behavior. This experiment trains
the new third-tier B factors in the full 24-layer BF16
Qwen2.5-0.5B-Instruct model and a matched continued E1280 control.
It decides whether a distinct E12800 artifact can be produced at
bounded local resource cost and is eligible for a new-source quality
comparison. It cannot by itself prove held-out quality, full C parity,
10B/100B transfer or >=50 accepted tok/s.

Bind the donor SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-56 parent SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
METH-107 learned child checkpoint SHA-256
`15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`,
METH-126 centered exact BF16 bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`,
and METH-135 third-tier sidecar SHA-256
`692db3dd1a647debb051ef66612263b5e16ef27c73978e9abd3beb6b3229abc9`.
Bind the same pinned raw/train-chat and long teacher-response files
used by METH-107 by their in-code hashes. Reusing those training
sources is visible; a fresh external audit must exclude their
sources/fragments and all prior audit manifests.

## Frozen paired recipe and gates

- Start both arms at the exact METH-126 BF16 E1280 function. The
  E12800 arm copies each centered E1280 B tensor to ten grandchild
  rows. Keep the donor core, E128 product-key router, E1280 child
  projection/keys, shared A and four gates fixed. Use METH-135's
  seeded third projection/keys, also fixed. Train B only, with the
  large arm's FP32 master and selected-row moments on CPU. Before
  training require exact BF16 logits on eight viewed bound prompts
  against the E1280 teacher, with the third-tier IDs mapping back
  to the same E1280 children. The eight viewed binding prompts are
  items 0–7 of METH-121 manifest SHA-256
  `7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366`.
  Stop on parity failure.
- Train an E1280 continued control and E12800 candidate for 256
  updates each, serially, on the same deterministic raw/chat draws
  with seed 136136. Each update has one 128-token raw window and one
  long teacher chat continuation. Use METH-107's objective against
  the **centered E1280 teacher**: 0.25 CE plus KL weights 4/8 for
  raw/chat and its confident-top1 hinge. Use selected-row AdamW with
  lr 1e-5, beta1 .9, beta2 .999, eps 1e-8, weight decay zero,
  per-row bias correction, accumulation over the two microbatches
  and global gradient norm clip 1.0. No gradient to core, A or any
  routing keys. Log draws, losses, selected-row counts, transfer
  bytes, process RSS, GPU peak and elapsed time. After update 16,
  stop if a loss/gradient is nonfinite or projected total runtime
  exceeds the cap; otherwise run through update 256.
- At export, leave the continued E1280 control as trained. For each
  E12800 source child, subtract its ten-grandchild mean shift and
  add the original centered E1280 B, then round to BF16. This
  mean-preserving operation is an explicit part of the candidate
  transformation, matching METH-121's lesson. Count BF16-distinct
  grandchildren against their source child after centering. Require
  at least 3,200 distinct and 6,400 selected grandchildren per
  layer and maximum/mean route load <=50 on training selections.
  Save versioned BF16 control/candidate artifacts with hashes and
  verify readback. These are training/coverage gates, not quality.

Use only the local RTX 3060. Stop at 45 minutes total, 10.5 GiB peak
allocated GPU memory, 50 GiB process RSS or 10 GB new disk. No T4.
If the paired training gates pass, freeze a fresh, source-disjoint
E1280-control versus E12800 candidate versus unchanged donor audit
of document BPB, prompt top1, greedy behavior, task regression and
blind grounding **before** reading its outcomes. A quality gain must
be attributed against the continued E1280 control as well as the
original checkpoint; distinct factor bytes or training loss alone
cannot demonstrate useful extra capacity.
