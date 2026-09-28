# METH-118–120: mean-preserving children pass development, external audit pending

**Decision.** The frozen mean-preserving E1280 transformation passes the new-source automatic and arm-blind semantic development gates. This is a development result, not external quality promotion. The external protocol was fixed separately before reading its outputs.

The [protocol](METH_118_119_ZERO_MEAN_CHILD_PROTOCOL_20260928.md), [source selector](../../../benchmarks/donor_adaptation/s1/meth118_zero_mean_child_dev_manifest.py), [manifest](meth118_zero_mean_child_dev_manifest.json), [evaluator](../../../benchmarks/donor_adaptation/s1/meth119_zero_mean_child_development.py), and [result](meth119_zero_mean_child_development_result.json) bind the METH-56 E128 parent and METH-107 E1280 child checkpoint. For each parent and layer, the effective child B is `parent_B + child_B - mean_10(child_B)`. A factors, parent and child routers, donor, and four active parent slots are fixed. The weight-only [diagnostic](meth118_weight_mode_diagnostic_result.json) finds common-mode RMS only 0.332–0.348 times child-specific RMS across 24 layers. Centering leaves 1,084–1,152 sibling-distinct child slots per layer and a maximum parent-mean error of 3.73e-9 in the actual model. Replacing every child B with its parent B reproduces E128 logits exactly on a source prompt and pooled BPB exactly on all 24 development documents.

METH-118 used eight code, eight technical and eight separate PG19 books from train shard 00002. Source IDs and overlapping text fragments were excluded through METH-115. Code and technical sources still share broad domains with earlier work.

| Development metric | E128 | Centered E1280 | Fixed gate |
|---|---:|---:|---|
| Pooled document BPB | 1.247689 | 1.247558 | Improvement 0.000132 ≥ 0.00005: pass |
| Donor prompt top-1 agreement | 94.769% | 93.810% | Loss 0.959 point ≤ 1: pass |
| Greedy EOS / 24 | 22 | 20 | At most two fewer: pass |

All category BPB, prompt top-1, repeated 8-gram and early non-EOS gates pass. The local RTX 3060 run took 291.4 s, peaked at 5.152 GB allocated GPU memory and used 3.485 GB RSS.

The [anonymous pairs](meth120_zero_mean_child_blind_pairs.json) exposed only each 384-character excerpt and A/B continuations. The [verdict builder](../../../benchmarks/donor_adaptation/s1/meth120_zero_mean_child_blind_verdict_build.py) and [verdict](meth120_zero_mean_child_blind_verdict.json) were committed at `b0a989a` before [unblinding](../../../benchmarks/donor_adaptation/s1/meth120_zero_mean_child_blind_semantic.py). The [score](meth120_zero_mean_child_unblinded_score.json) gives 28 unsupported claims for each arm, 11 severe claims for E128 versus 10 for E1280, and zero answers lacking a supported detail in either arm. The fixed no-regression semantic gate passes exactly on the unsupported count, leaving little margin. This is one reviewer on 24 short excerpts, not a population estimate.

This transformation has no demonstrated CPU factor/LUT cost, full `engine.c` execution, 10B/100B donor transfer or same-artifact accepted-token rate. The route-only METH-104 measurement applies to unchanged routes, not to the effective centered factors. Continue only through the separately frozen [external audit](METH_121_123_ZERO_MEAN_EXTERNAL_PROTOCOL_20260928.md).
