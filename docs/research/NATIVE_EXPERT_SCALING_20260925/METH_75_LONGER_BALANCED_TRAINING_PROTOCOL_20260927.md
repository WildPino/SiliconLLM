# METH-75: fixed 64→256 update continuation of balanced E128/E1280

**Question.** METH-71's 64-update E1280 bank passes development
retention and route-balance gates, but METH-72 shows weak learned-pair
utility and METH-74 finds one excess severe unsupported claim. Test
whether more updates of the same precommitted rule strengthen the
expert factors while preserving donor behavior. The METH-72/74
failures remain failures for the update-64 artifact.

Resume the exact METH-71 E128 and E1280 checkpoints, including dense
AdamW state and NumPy/Torch RNG states, from update 64. Perform 192
further updates to fixed terminal update 256. Keep the same BF16
Instruct donor, FP32 independent rank-8 factors, rank-64 product keys,
top-four routing, two raw/two chat microbatches, raw/chat KL weights
2/4, factor/router learning rates 1e-4/1e-5, clip 1, no weight decay
and product-axis balance weight 0.02. Do not change top-four inference.
Require both arms to consume identical continuation training rows.

Before inference or continuation, freeze 24 new chat prompts using
seed 757575. Exclude all METH-43/44/47/55/56/66/70/71 prompt rows,
the METH-70 sampled raw training rows, and all raw rows that the
METH-71 RNG would sample in the full 256-update sequence, including
both past and future training. Reconstruct the METH-71 raw pool and
advance the exact 717172 RNG stream over two raw and two chat draws
per update. Continue with the original METH-71 raw pool and saved RNG
state, so selecting the new prompts does not change training draws.
Record row lists and source/token hashes.

At update 256, require exact checkpoint/source identity and finite
nonzero B/router gradients; donor/student initial logit identity
remains the source model's zero-B control from METH-71. On the new
prompts require donor-position top-1 ≥95% and no more than one
percentage point below the same-arm frozen update-64 checkpoint
scored on those prompts. Require held-out raw ΔBPB≤+0.05; minimum
changed B slots per layer ≥64 E128/≥640 E1280; minimum selected
slots ≥96 E128/≥640 E1280; and worst maximum/mean route load
≤32× E128/≤64× E1280. Check product-key top-four against exhaustive
pair scores on the evaluated hidden states. A joint development
pass only authorizes a new source-disjoint external quality and
semantic audit; it cannot override METH-72/74.

Local RTX 3060 only, ≤10.5 GiB peak allocated GPU, ≤20 GiB RSS and
≤15 minutes per arm including checkpoint save/hash. Stop with partial
evidence on a budget or nonfinite failure. No T4. Native CPU LUT
parity, memory traffic and ≥50 accepted tok/s remain separate.
