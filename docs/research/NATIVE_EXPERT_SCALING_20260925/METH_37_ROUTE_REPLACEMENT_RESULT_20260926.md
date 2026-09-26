# METH-37: route replacement misses the fixed top-1 gate

The [prospective protocol](METH_37_ROUTE_REPLACEMENT_PROTOCOL_20260926.md)
replaced the trained E128 model's exhaustive route with the
stored METH-36 rank-64 int8 sketch and exact rescoring of 64
candidates. The R8 core, expert factors, fine-router weights,
top-4 and factor-0.50 amplitude remained fixed. The 24 selected
METH-19 documents (eight per category, 98,276 bytes) are disjoint
by source ID from METH-27 and METH-36. Their [manifest](meth37_route_quality_manifest.json)
was frozen before scoring, SHA-256
`0c9004f36803a600deda1f884591e61a1a5ffb2b1ac0558ba75727565d30a913`.
The exact-route control reproduced all 24 saved METH-25 top-1
streams before this comparison.

| Category | BF16 donor BPB | R8 exact BPB | R8 int8 route BPB | Int8 minus exact |
|---|---:|---:|---:|---:|
| Code | 0.995217 | 1.001565 | 1.001482 | −0.000083 |
| Prose | 1.153487 | 1.151403 | 1.151289 | −0.000114 |
| Technical/general | 0.715441 | 0.704640 | 0.704415 | −0.000225 |
| Pooled | 0.954725 | 0.952546 | 0.952406 | **−0.000140** |

The category-stratified 20,000-draw bootstrap one-sided 95%
upper bound for int8-minus-exact BPB is +0.000026. Both the
paired document gate and the separate donor-relative document
screen pass on these already parent-screened source documents.
However, next-token top-1 agreement is **6,043/6,144 = 98.356%**,
below the predeclared ≥99% gate. These are full model trajectories,
not simply the route-ID inclusion counts on exact-route inputs.

On 24 fixed 256-token prompts with 128 greedy tokens, repeated
8-gram≥3× counts are **17/24 exact versus 16/24 int8 route**:
code 7→7, prose 7→7, technical 3→2. Neither arm had early
non-EOS termination. The relative generation gate passes, but
both arms retain severe absolute repetition; this does not
repair the parent's METH-27 usefulness failure. The int8 arm's
lower loop count and tiny negative BPB delta are sample outcomes,
not a general quality improvement.

The [machine result](meth37_route_replacement_result.json),
SHA-256
`d3d69a24c767987b38b57bf6dc240194259174ad5476ab59f757661d12b4b5d5`,
retains every paired document score, top-1 vector, generated ID
sequence and decoded continuation. Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth37_route_replacement.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth37_route_quality_manifest.json --manifest-sha256 0c9004f36803a600deda1f884591e61a1a5ffb2b1ac0558ba75727565d30a913 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth37_route_replacement_result.json`.
The local RTX 3060 scoring phase took 414.77 s, peaked at
2.620 GB allocated GPU memory and ended at 3.235 GB RSS.
No T4 was used.

**Decision:** reject rank-64/64-candidate route replacement under
the prospective joint gate. METH-36's 99.939% top-4 ID
inclusion on an exact-route trajectory was too weak a proxy for
top-1 preservation after applying changed routes. Diagnose
whether the difference comes from omitted candidates or from
numeric rescoring before changing candidate budget or index.
No C timing or large-E quality claim follows from this failure.
