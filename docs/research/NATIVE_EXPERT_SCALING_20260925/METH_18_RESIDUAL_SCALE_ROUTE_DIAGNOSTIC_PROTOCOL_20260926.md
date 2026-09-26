# METH-18: diagnose fresh-document residual magnitude and route

**Prospective diagnostic, not a promotion test.** METH-17 rejected the
METH-16 E128 checkpoint at +0.013215 pooled BPB, including a positive
delta on every selected code document. This test asks whether the
generalization loss follows residual magnitude, route assignment, or
both. The 60 METH-17 documents are now development evidence. No
outcome here can establish fresh quality or authorize native export.

## Fixed inputs and arms

Reuse the exact METH-17 manifest SHA-256
`7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8`
and the METH-16 update-1024 checkpoint SHA-256
`a9e74c9ceeb8a91fd04f9b9e35a981965db9ab824d2d25ea728dd940ea5365a4`.
Reconstruct the manifest and verify the model, tokenizer, corpus and
checkpoint hashes before scoring. Use the RTX 3060 BF16/SDPA model and
METH-17's EOS prefix, 512-token stride, 512-token left context and
document absolute positions. Score all document text tokens.

In this order, score each document with the following shared model:

1. Donor: residual experts disabled.
2. Trained route, all expert output factors multiplied by 0.25.
3. Trained route, all expert output factors multiplied by 0.50.
4. Unchanged checkpoint: factor 1.00.
5. Factor 1.00, but independently permute the 128 router rows in each
   layer with NumPy seed 1818. Leave expert factors in original order.

Scaling only output factors multiplies the selected residual by that
factor while preserving selected expert IDs. Restore exact factor
1.00 and router tensors between arms. The route permutation is a
destructive null and is not an implementable candidate. Verify the
factor-1.00 scores reproduce METH-17's pooled donor/student values
within 1e-5 BPB and the same 24/24 code loss signs before interpreting
the new arms.

Report pooled and category BPB deltas to donor, each code-document
delta, and permuted-minus-trained BPB by category. Also score the
fixed 16×512 METH-16 internal development windows at factor 0.25,
0.50 and 1.00, with their same byte denominator; do not use them for
promotion. No scalar is selected from these outcomes for deployment.

## Decision and bounds

If lower factors reduce the code loss while the trained route beats
the permutation on code, prioritize a domain-sensitive residual
magnitude or training constraint. If permutation beats trained route
on code, prioritize router transfer. Mixed signs require separating
layers/domains in a subsequent registered diagnostic. Every proposed
repair must use data distinct from these 60 documents and pass a
separate prospective document/task/generation test before export.

One local RTX 3060 run; stop at 15 minutes wall time, 10.5 GiB peak
allocated GPU memory or 20 GiB process RSS. No T4. This experiment
changes no checkpoint or engine artifact. The reason to spend this
cost is to choose the next transfer repair, not to re-score a known
failure as success.
