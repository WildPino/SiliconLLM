# METH-140: training-fit deciles do not generalize to separate documents

**Decision: routing screen fail.** Fitting the same conditional-decile
third tier on all 256 training draws preserves exact initial BF16
full-model logits and selects at least 6,047 grandchildren per layer
on each of two source-held-out manifests. Its load balance collapses
on both document sets. Maximum candidate/control skew reaches 3.214×
and one grandchild receives 88.09% of a source child's selections.
No native CPU cost test, B training or quality audit follows.

The [protocol](METH_140_FULL_CALIBRATION_EXTERNAL_ROUTE_PROTOCOL_20260928.md)
was committed at `7a75082` before the run, and the
[implementation](../../../benchmarks/native_expert_scaling/meth140_external_quantile_route.py)
at `7276aec`. It used the exact METH-126 centered E1280 bank and
METH-135 projection, calibrated only on METH-136's training draws,
then ran nonoverlapping <=512-token windows from the previously
frozen METH-121 and METH-133 manifests. Neither set changed the
thresholds. Those manifests were viewed in older quality experiments;
this is a route-load generalization test, not new quality evidence.
The new 3,858,464-byte sidecar SHA-256 is
`a581272ed84c154b660a4fd9a7c108f0dcced394e0d763ddfdd366aa92eadc5f`;
header and every projection/threshold byte passed readback.

The [result](meth140_external_quantile_route_result.json), SHA-256
`98ce4d54e57fe9d246231538c7ade968d6228546cccf462b32322546ea8568a4`,
contains all 48 layer/manifest summaries. The local RTX 3060 run
took 157.735 seconds, 4.010 GB final process RSS and 3.116 GB peak
allocated GPU memory, below the frozen stops. No T4 was used.

| Frozen measure | METH-121 documents | METH-133 documents |
|---|---:|---:|
| Document tokens | 29,186 | 29,926 |
| Minimum grandchild coverage per layer | 6,047 | 6,155 |
| Worst candidate/control maximum-to-mean load ratio | 3.198× | 3.214× |
| Layers failing <=1.25× load ratio | 24/24 | 23/24 |
| Worst hot-parent grandchild share | 87.93% | 88.09% |
| Layers failing <=25% hot-parent share | 24/24 | 24/24 |

All eight pinned prompts have exactly equal BF16 logits between the
original E1280 model and a quantile-routed model gathering identical
source B factors. Thus this failure concerns slot allocation, not
initial model function. Increasing calibration from 128 to 256 of the
same short training draws did not solve the distribution shift to
document windows. The data and input construction differ in both
content and context length; this result does not isolate which factor
causes the shift. A follow-up should calibrate with long training-only
contexts and validate on separate sources, or adopt a routing rule
whose balance does not depend on fitted scalar deciles. The current
sidecar remains rejected and has no quality-valid learned bank.
