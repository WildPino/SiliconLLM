# METH-178: most trained B-bank difference is shared across children

The long-trained E12,800 BF16 bank has changed almost every row, but its nine content children within each parent remain numerically close. Across all 24 layers, only **1.286%** of the unweighted candidate-versus-matched-control squared B difference is within-parent child variation; the remaining 98.714% is a common content-parent mean shift. Weighting each child by its recorded METH-175 training selections raises the within-parent fraction to **2.318%**, still leaving 97.682% in the route-weighted parent mean. Every layer's weighted fraction is 1.683–3.431%. This is a weight-geometry diagnosis, not an inference or quality result.

The [protocol](METH_178_TRAINED_CHILD_DIVERSITY_PROTOCOL_20260930.md) was committed at `7d2f8ff` before measurement. The [streaming audit](../../../benchmarks/native_expert_scaling/meth178_trained_child_diversity.py) was committed at `38917cb` before execution. The [machine result](meth178_trained_child_diversity_result.json), SHA-256 `d011418b336a046c1a58880ba8e8d608d6a6a7b66a44cda1443883022dedb65b`, binds both exact METH-175 result and BF16 bank hashes. It checks every bank header, all 24×1,280×9 content rows, 4,200,432 content selections per layer, and the unweighted and route-weighted sum-of-squares identities.

| Pooled per-layer measurement | Unweighted RMS | Training-route-weighted RMS |
| --- | ---: | ---: |
| Continued E1,280 B reference | 0.00341415 | 0.00351875 |
| E12,800 content child minus matched control | 0.00027885 | 0.00040993 |
| Content-parent mean shift | 0.00027705 | 0.00040515 |
| Content child variation within parent | **0.00003163** | **0.00006241** |

The shared structural slot's candidate-minus-control RMS is 0.00027890. The complete audit took 60.844 seconds with 4.910 GB final process RSS, below the 15-minute and 8-GiB caps. No model inference, GPU or T4 was used. The result never reads the METH-176 quality cohort, whose failed bootstrap remains the authoritative held-out decision.

This explains why BF16 row-change counts are too weak to establish useful new capacity: most of the measured B difference relative to the matched control is common to siblings. It does **not** prove that the small child-specific part has no effect on logits, because direction and activation can matter more than weight RMS. The next mechanism test should ablate or permute child routing on previously consumed diagnostic inputs, then use a genuinely untouched cohort for any new candidate's quality claim. Simply extending this fixed-route training without evidence of functional specialization is poorly motivated.
