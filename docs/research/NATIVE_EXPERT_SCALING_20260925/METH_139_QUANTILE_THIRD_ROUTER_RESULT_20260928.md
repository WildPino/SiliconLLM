# METH-139: conditional deciles greatly reduce skew but fail two frozen gates

**Decision: routing screen fail.** The quantile third tier preserves
the exact initial BF16 E1280 full-model logits on all eight bound
prompts and selects 7,299–10,897 grandchildren per layer on the
held-out half of the routing draws. It reduces the extreme
concentration seen with METH-135's seeded argmax keys. It misses the
frozen 1.25× relative load gate in one layer (1.311×) and the 25%
within-parent share gate in another (26.94%). No B training or
new-source quality test follows this screen.

The [protocol](METH_139_QUANTILE_THIRD_ROUTER_PROTOCOL_20260928.md)
was committed at `b0673c7` before execution, and the
[implementation](../../../benchmarks/native_expert_scaling/meth139_quantile_third_router.py)
at `5b45990`. It bound the exact METH-126 BF16 E1280 bank and METH-135
third projection by their pinned hashes, then captured unchanged
E1280 child IDs and a projected scalar for all METH-136 raw/chat draws.
It computed child-specific linear deciles from updates 1–128 only;
children with fewer than ten samples used a layer-wide decile for
their `child mod 32` projection dimension. The 129–256 samples were
excluded from threshold fitting. The resulting sidecar is 3,858,464
bytes, SHA-256
`967d69ae867e15da804d1e7c244bc514b9c44efb95b71926c55222ca7adf8206`,
with exact binary readback. It replaces the third-tier argmax keys
with nine thresholds per child; its native CPU cost has not been
measured.

The [result](meth139_quantile_third_router_result.json), SHA-256
`bcf3a0f8747495e7d843f3d293b92ecbb532a8382489c57a8196fce3e1c003e5`,
contains all 24 layer rows, calibration counts, gate outcomes and
resource telemetry. The local RTX 3060 run took 104.562 seconds,
8.462 GB process RSS and 3.116 GB peak allocated GPU memory, all
below the frozen stops. No T4 was used.

| Frozen validation gate, updates 129–256 | Observed | Decision |
|---|---:|---|
| Full-model initial BF16 logit maximum error, eight prompts | 0 | pass |
| Minimum selected grandchildren per layer | 7,299 | pass, >=5,000 |
| Worst candidate/control maximum-to-mean ratio | 1.311×, layer 23 | **fail**, <=1.25× |
| Worst grandchild share within parent selected >=500 times | 26.94%, layer 0 | **fail**, <=25% |
| Candidate maximum-to-mean load, largest layer | 61.203× | diagnostic |
| Control maximum-to-mean load, largest layer | 57.010× | diagnostic |

The two failing conditions occur in different layers; only one layer
fails each. Across layers, 145–533 children needed fallback deciles
because they had fewer than ten calibration selections. The remaining
conditional skew is therefore modest compared with the METH-138
seeded-key replay, but the runs use different evaluation windows and
must not be turned into a direct percentage improvement claim.
METH-139 does not override METH-136's failed gate, and its thresholds
have not been tested for quality or native CPU cost. A follow-up may
use more training-only calibration samples and separate source-held-out
routing inputs, with a new frozen protocol and no reuse of the viewed
METH-139 validation measurements as a promotion test.
