# METH-115–117: useful E1280 child choice passes automatic audit, fails blind semantics

**Decision.** The alpha=0.75 E1280 candidate passes every frozen automatic external gate, including a matched no-child ablation showing a small benefit from selecting among distinct children. The separately committed arm-blind semantic verdict has more unsupported and severe claims than E128. The joint quality gate **fails**; do not promote this candidate or reuse these viewed examples to select another alpha. The earlier alpha=1.0 METH-112 rejection remains in force.

The [protocol](METH_115_117_ALPHA075_EXTERNAL_PROTOCOL_20260927.md), [source manifest](meth115_alpha075_external_manifest.json), [audit runner](../../../benchmarks/donor_adaptation/s1/meth116_alpha075_external_audit.py) and [automatic result](meth116_alpha075_external_audit_result.json) bind alpha 0.75, the METH-107 checkpoint, METH-56 E128 parent and intact BF16 donor. METH-115 excludes previous source IDs and overlapping text through METH-113 and freezes eight code, eight technical and eight separate prose books from PG19 train shard 00001. The donor calibration corpus used a different PG19 train shard; prior external prose came from test/validation or Europarl. The Git code/technical pool still shares a broad domain with prior audits, though file IDs and fragments differ.

| New external metric | E128 | E1280 alpha=0.75 | Gate |
|---|---:|---:|---|
| Pooled document BPB | 1.177652 | 1.177196 | E1280−E128 −0.000456: pass |
| Donor prompt top-1 agreement | 95.235% | 94.600% | Loss 0.635 point: pass |
| Greedy EOS / 24 | 24 | 23 | Pass |
| Repeated 8-gram three times / 24 | 0 | 0 | Pass |
| PIQA correct / 1,838 | 1,292 | 1,288 | −0.218 point: pass |

All category gates pass. The 1,838-item PIQA paired bootstrap lower fifth percentile is −0.00653, above the fixed −0.05 floor; 12 E128-correct items become wrong and eight move the other way. PIQA is a repeated regression task, not a new independent task sample. The E1280 mean distinct-bigram ratio is lower (0.908 versus 0.941); this was not a stop gate but remains a visible risk. There are 1,069–1,152 sibling-distinct B slots per layer after scaling. The matched no-child ablation replaces each parent's ten B factors with their mean and yields BPB 1.177280: actual child selection improves it by 0.000084, above the frozen 0.00005 floor. This is narrow evidence that the 10× child choice matters on this set, not proof of robust utility at large donor scale. The RTX 3060 run took 1,034.58 s, peaked at 5.183 GB allocated GPU memory and used 3.524 GB RSS.

The [anonymous pairs](meth117_alpha075_blind_pairs.json) showed only each 384-character excerpt and A/B continuations. The [builder](../../../benchmarks/donor_adaptation/s1/meth117_blind_verdict_build.py) and [verdict](meth117_alpha075_blind_verdict.json) were committed at `b291808` before [unblinding](../../../benchmarks/donor_adaptation/s1/meth117_alpha075_blind_semantic.py). The [score](meth117_alpha075_unblinded_score.json) is:

| Excerpt-only finding | E128 | E1280 alpha=0.75 | Gate |
|---|---:|---:|---|
| Unsupported claims | 33 | 40 | **Fail** |
| Severe unsupported claims | 14 | 15 | **Fail** |
| Answers lacking a supported specific detail | 0 | 0 | Pass |

The excess is concentrated in several code and technical explanations: one E1280 answer confuses a server host with its port and invents a literal `mode` value, another says an experiment used T4 and speed measurements although the excerpt says neither was used, and another reverses the purpose of measuring free-running rollout trajectories. These findings were recorded before arm reveal. A single agent judged 24 short excerpts, so the counts do not estimate population risk; the predeclared zero-regression gate still rejects the candidate.

METH-104's exact CPU router fixture and route-only time apply to the unchanged parent/child routing weights, but its bank omits these effective A/B factors. No learned-factor CPU/LUT benchmark, full `engine.c` execution or accepted-token rate exists for this candidate. The current result shows a technically useful 10× choice geometry can pass automatic gates yet fail grounded response fidelity. A new quality method needs new development and external evidence; lowering alpha now on the viewed METH-113/METH-115 sequence would violate the frozen selection rule.
