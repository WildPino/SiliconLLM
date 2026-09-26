# METH-38: diagnose the failed route-replacement ranking gate

METH-37's stored rank-64 int8 shortlist preserves document BPB
and relative generation but matches only 98.356% of exact
next-token top-1 IDs at 64 candidates. On the *same already
observed 24 prompts*, distinguish missed candidates from
numeric differences in exact rescoring. This is diagnostic
reuse; no result here independently promotes a new budget.

Bind source/core/adapter/sketch identities as METH-37, its
selected prompt manifest SHA-256
`0c9004f36803a600deda1f884591e61a1a5ffb2b1ac0558ba75727565d30a913`,
and its complete result SHA-256
`d3d69a24c767987b38b57bf6dc240194259174ad5476ab59f757661d12b4b5d5`.
Before interpreting new arms, reproduce METH-37's exact
and 64-candidate top-1 streams bit for bit.

Keep the same stored rank-64 basis/codes, score all 128 sketch
rows, and test two additional fixed exact-rescore budgets:
**96** and **128** candidates. The 128-candidate arm includes
every expert and isolates the alternative scoring path's
numerical effect; the 96-candidate arm tests whether 32 more
exact rows resolve the observed shortlist misses. Run all
arms on their own 256-token model trajectories. At each
input-layer case also calculate the exhaustive fp32 top-4
*for diagnostic counting only*; never use it to choose or gate
the shortlist arm. Record exact-ID inclusion, full-set match,
and per-layer route differences on each arm's own trajectory,
plus 6,144 top-1 comparisons to the original exact arm.

The apparatus gate for 128 candidates is **100% exact top-4
set match and ≥99.99% top-1 agreement**. A failure points to
rescoring/implementation numerics or a control mismatch.
The 96-candidate diagnostic gate is ≥99% top-1 agreement
and ≥99.9% exact-ID inclusion on its own trajectory. A pass
permits a new independent quality test with 96 candidates;
it does not override METH-37's failed 64-candidate gate.
No CPU speed or large-E quality is inferred.

One local RTX 3060 run, ≤15 minutes wall time, 10.5 GiB
allocated GPU memory and 20 GiB RSS. No T4 job is planned.
