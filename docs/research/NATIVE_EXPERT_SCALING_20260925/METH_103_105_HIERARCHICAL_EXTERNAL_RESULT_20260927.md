# METH-103/105: independent E1280 quality audit and blind semantic stop

**Decision.** The METH-99 learned hierarchical E1280 checkpoint passes all frozen automatic gates on 24 new source-disjoint METH-102 documents and all 1,838 PIQA items. The separately committed arm-blind semantic verdict fails one of three gates: E1280 has one answer missing the requested grounded detail versus zero for E128. The checkpoint is **not promoted** as donor-relative quality-preserving. The E1280 route and factor bank also have no full native/LUT result.

The [protocol](METH_103_HIERARCHICAL_EXTERNAL_PROTOCOL_20260927.md), [manifest](meth102_hierarchical_external_manifest.json), [audit runner](../../../benchmarks/donor_adaptation/s1/meth103_hierarchical_external_audit.py) and [full automatic result](meth103_hierarchical_external_audit_result.json) bind the source and METH-56/METH-99 checkpoints. The automatic audit used intact BF16 donor, BF16+E128 and BF16+E1280 on identical IDs. E1280−E128 pooled BPB is −0.000805 (1.181619 versus 1.182424); every category improves. Donor prompt top-1 is 92.782% for E1280 versus 93.679% for E128, a 0.897-point loss within the frozen one-point bound; all category losses are within two points.

Greedy 128-token generation terminates with EOS in 21/24 E1280 answers versus 23/24 E128 answers, exactly the allowed two-answer difference. Both arms have one repeated 8-gram three times and no early non-EOS exit under 16 tokens. On full PIQA, intact BF16 donor scores 1,291/1,838 and E128 and E1280 each score 1,292/1,838; 11 examples change from E128-correct to E1280-wrong and 11 in the reverse direction. The paired bootstrap fifth percentile is −0.00435, above the frozen −0.05 floor. These results pass all automatic gates but do not measure grounded answer fidelity.

The [blind pair file](meth105_hierarchical_blind_pairs.json) showed only the 384-character excerpt and A/B continuations. The [verdict](meth105_hierarchical_blind_verdict.json) and its [builder](../../../benchmarks/donor_adaptation/s1/meth105_blind_verdict_build.py) were committed at `ebef488` before running the [unblinding scorer](../../../benchmarks/donor_adaptation/s1/meth105_hierarchical_blind_semantic.py). The [unblinded score](meth105_hierarchical_unblinded_score.json) is:

| Excerpt-only finding | E128 | E1280 | Frozen gate |
|---|---:|---:|---|
| Unsupported claims | 35 | 30 | Pass |
| Severe unsupported claims | 11 | 8 | Pass |
| Answers missing a supported specific detail | 0 | 1 | **Fail** |

The only missing-detail case is the technical-report table excerpt from `STRAT_02_W4_BF16_QUALITY_RESULT.md`: the E1280 continuation lists misinterpreted and fabricated statistics without a supported specific detail, while the E128 continuation at least states the visible total/category limit violation. This judgment is recorded in the blind verdict, before arm identity was revealed. The review is by one agent on short excerpts and is not a population estimate; the predeclared zero-regression rule still rejects this checkpoint.

Next quality work should protect grounded detail in long or numerical excerpts, then use a new source-disjoint development and external set. Reusing METH-102 to tune the checkpoint would invalidate an independent claim. A full C/LUT implementation and accepted-token throughput remain separate gates.
