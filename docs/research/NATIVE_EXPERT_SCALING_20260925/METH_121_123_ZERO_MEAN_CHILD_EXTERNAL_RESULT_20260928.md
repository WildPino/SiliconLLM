# METH-121–123: mean-preserving E1280 passes new-source external quality gates

**Decision.** The centered METH-107 E1280 child bank passes every frozen automatic external gate and the separately committed arm-blind semantic gate against METH-56 E128. This is evidence for useful 10× distinct child choice on the tested BF16 Qwen2.5-0.5B-Instruct donor. It advances the candidate to a native-factor CPU and full-engine feasibility test. It is not an end-to-end speed or 10B/100B result.

The [protocol](METH_121_123_ZERO_MEAN_EXTERNAL_PROTOCOL_20260928.md) was committed as `e2dc849` before source selection. The [selector](../../../benchmarks/donor_adaptation/s1/meth121_zero_mean_child_external_manifest.py) and [manifest](meth121_zero_mean_child_external_manifest.json) use eight code, eight technical and eight separate PG19 books from train shard 00003, excluding prior source IDs and overlapping text fragments through METH-118. The Git code/technical sources remain in a shared broad domain. The [evaluator](../../../benchmarks/donor_adaptation/s1/meth122_zero_mean_child_external_audit.py) holds donor, checkpoint, routing and centered-B transformation fixed; its [automatic result](meth122_zero_mean_child_external_audit_result.json) was committed before blind review.

| External measure | E128 | Centered E1280 | Frozen gate |
|---|---:|---:|---|
| Pooled document BPB | 1.185238 | 1.184887 | Improvement 0.000350 ≥ 0.00005: pass |
| Donor prompt top-1 agreement | 95.194% | 94.504% | Loss 0.690 point ≤ 1: pass |
| Greedy EOS / 24 | 19 | 20 | At most two fewer: pass |
| Repeated 8-gram three times / 24 | 0 | 0 | At most one extra: pass |
| PIQA correct / 1,838 | 1,292 | 1,291 | Accuracy difference −0.0544 point: pass |

Every category BPB improves: code −0.000278, prose −0.000278 and technical −0.000495. Every category prompt gate passes; prose loses 1.190 points versus its two-point limit. The paired PIQA bootstrap lower fifth percentile is −0.004897, above the fixed −0.05 floor, with 12 E128-correct items lost and 11 gained. PIQA is a repeated regression set, not a new independent task sample. No-child B factors copied directly from the E128 parent reproduce exact first-prompt logits and identical pooled document BPB; the measured E1280 improvement therefore depends on selecting among children. There are 1,084–1,152 sibling-distinct B slots per layer and maximum parent-mean discrepancy 3.73e-9. The local RTX 3060 run took 1,090.5 s, peaked at 5.214 GB allocated GPU memory and used 2.668 GB RSS.

The [anonymous pairs](meth123_zero_mean_child_blind_pairs.json) contained each 384-character source excerpt and A/B continuations. The [builder](../../../benchmarks/donor_adaptation/s1/meth123_zero_mean_child_blind_verdict_build.py) and [verdict](meth123_zero_mean_child_blind_verdict.json) were committed at `ecc94d6` before [unblinding](../../../benchmarks/donor_adaptation/s1/meth123_zero_mean_child_blind_semantic.py). The [unblinded score](meth123_zero_mean_child_unblinded_score.json) is:

| Excerpt-only finding | E128 | Centered E1280 | Gate |
|---|---:|---:|---|
| Unsupported claims | 31 | 25 | Pass |
| Severe unsupported claims | 12 | 7 | Pass |
| Answers without a supported specific detail | 0 | 0 | Pass |

One agent reviewed 24 short excerpts, so these counts are a screen rather than a population estimate. The A/B review deliberately contains paired shared mistakes; it does not establish that either model is generally error-free or superior to the dense donor. METH-104's route-only native result uses the same fixed parent/child routers and is relevant for routing cost, but its bank lacks these centered A/B factors. A native export with actual selected-factor access, numeric parity and varied-token timing is next. LUT-coded factor arithmetic, integrated `engine.c` execution, same-artifact accepted-token throughput and transfer to larger donor families/scales remain unverified.
