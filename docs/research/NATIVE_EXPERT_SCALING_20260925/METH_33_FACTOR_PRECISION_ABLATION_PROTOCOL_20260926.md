# METH-33: A/B mixed precision diagnosis for the stored rank-8 factors

METH-32 showed that jointly replacing the trained rank-8 A and B
factors with their fixed ternary LUT encoding passes reused-document
BPB but matches only 96.012% of the intact model's next-token top-1
IDs, below its 99% fidelity gate. This cell changes **one factor
organ at a time** to identify whether a mixed format can preserve
ranking while reducing selected-factor bytes. It is a diagnostic
on reused METH-25 texts, not independent promotion.

Bind the packed R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`,
the intact E128 factor-0.50 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
and the already exported METH-32 two-trit LUT factor artifact SHA-256
`a99542407aea7f1ae9eb7ac0401f9406e7c0cb1516129955c427538950cf7eee`.
Use the unchanged fp32 fine router. Compare exactly two arms:

1. **A ternary / B fp32**: decode A from the stored METH-32 codes
   and scales; load B from the intact adapter. Selected A/B factor
   bytes are 4,131,840/token at L24/top-4, 75% of all-fp32 factors.
2. **A fp32 / B ternary**: load A from the intact adapter; decode B
   from the stored METH-32 codes and scales. Selected factor bytes
   are 3,440,640/token, 62.5% of all-fp32 factors.

The METH-32 all-ternary and intact fp32 results are fixed references,
not tuning arms. No re-quantization, retraining, amplitude adjustment
or route change is allowed. The byte counts exclude the unchanged
exhaustive fp32 router, R8 core and dispatch overhead. The selected
factor cost gate is ≤4,200,000 bytes/token, which both arms meet by
construction; measured C speed remains open.

Reconstruct the R8 core from stored bytes and reproduce the intact
adapter's 48 METH-25 document nats within 0.01 and all 24 saved
top-1 streams exactly before scoring either arm. Use the same
METH-25 manifest SHA-256
`20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77`
and METH-32 audit SHA-256
`c378cf06e26934144c603fd9cd2ffe4b12e305fd6af272c5bb9da4bae33ad960`.
Score each of 48 full documents with the same EOS-prefix,
512-target-stride, 512-left-context BPB method. Compare 6,144
top-1 positions on the fixed 24 prefixes. Report full top-4
route-set agreement on the resulting model trajectories as a
diagnostic, not an approximate-route fidelity gate.

Each arm passes its **diagnostic** only if its selected-factor bytes
are ≤4,200,000/token, pooled BPB penalty versus intact ≤+0.001,
code ≤+0.002, prose/technical each ≤+0.003, and a 20,000-draw
category-stratified bootstrap one-sided 95% penalty upper bound
with seed 333333 is ≤+0.003; top-1 agreement must be ≥99%.
It must also retain the METH-25 donor screen: pooled ΔBPB≤+0.01,
code≤+0.01 and prose/technical≤+0.03 versus original BF16 donor.
If both pass, choose the smaller selected-factor byte count; if
one passes, choose it; if neither passes, reject both. A passing
diagnostic permits a stored mixed export and **new independent**
document/generation/task audits before C promotion. It does not
establish the full model's quality or ≥50 accepted tok/s.

Local RTX 3060 only; no T4. Stop at 15 minutes audit wall time,
10.5 GiB allocated GPU memory or 20 GiB RSS. If baseline controls
fail to reproduce, stop without interpreting candidate scores.
