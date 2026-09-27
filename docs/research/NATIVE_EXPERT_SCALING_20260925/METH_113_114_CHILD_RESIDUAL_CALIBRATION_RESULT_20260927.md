# METH-113/114: alpha=0.75 passes development quality and child-choice ablation

**Decision.** The frozen 0.75 child-residual scale passes new-source automatic development gates, a matched no-child utility ablation, and a precommitted arm-blind semantic development review. It is the largest eligible scale in the descending grid and proceeds to METH-115–117 external audit. This is a development selection, not quality promotion; alpha=1.0 remains rejected by METH-112.

The [protocol](METH_113_114_CHILD_RESIDUAL_CALIBRATION_PROTOCOL_20260927.md) fixed alpha candidates and selection rules before scoring. Each effective child factor is `B_parent + alpha*(B_child-B_parent)`, preserving the METH-107 parent and child routes, copied A factors and four active slots. Alpha zero is parent parity, alpha one is the already rejected full-strength control. At alpha 0.75, 1,069–1,152 child slots per layer still differ from sibling zero by more than 0.001 in a B coordinate. This counts distinct factors; the utility ablation below tests whether their selection matters.

The [METH-113 manifest](meth113_child_scale_dev_manifest.json) froze eight code, eight prose and eight technical documents with source and token hashes before evaluation, excluding prior IDs and overlapping fragments through METH-110. Prose came from disjoint 1-MiB windows 128–255 of the pinned Europarl English concatenation. These are separate byte windows of one corpus, not separate books. The [METH-114 runner](../../../benchmarks/donor_adaptation/s1/meth114_child_scale_development.py) scores intact BF16 donor, E128 and alpha=0.75 E1280 on identical IDs. Its [automatic result](meth114_alpha075_development_result.json) records:

| New development metric | E128 | E1280 alpha=0.75 |
|---|---:|---:|
| Pooled BPB | 1.041458 | 1.040999 |
| Donor prompt top-1 agreement | 96.146% | 96.443% |
| Greedy EOS / 24 | 23 | 23 |

Every category and pooled automatic gate passes. The matched no-child ablation keeps the same routes and replaces all ten child B factors of each parent by their mean; its BPB is 1.041215. Selecting distinct children improves BPB by 0.000216, above the predeclared 0.00005 minimum. This is a small but directly measured child-choice benefit, not proof of broad semantic utility. The run took 282.39 s, peaked at 5.134 GB allocated GPU memory and used 3.320 GB RSS.

The [anonymous continuation pairs](meth114_alpha075_blind_pairs.json) and [verdict](meth114_alpha075_blind_verdict.json) were committed at `2d44d67` before the [unblinding scorer](../../../benchmarks/donor_adaptation/s1/meth114_child_scale_blind_semantic.py) ran. The [score](meth114_alpha075_unblinded_score.json) finds E1280 versus E128: 26 versus 30 unsupported claims, 6 versus 8 severe claims and 0 versus 0 answers lacking a supported requested detail. All three development semantic gates pass. One agent judged 24 short excerpts; this is a limited screening signal, not a population estimate. The separately [frozen external protocol](METH_115_117_ALPHA075_EXTERNAL_PROTOCOL_20260927.md) must pass before any quality promotion, and native/LUT performance remains unmeasured.
