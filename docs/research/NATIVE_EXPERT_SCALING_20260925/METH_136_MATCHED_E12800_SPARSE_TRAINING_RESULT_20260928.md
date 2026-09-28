# METH-136: learned E12800 artifact fails the frozen route-load gate

**Decision: reject this E12800 router/training combination.** Both
matched 256-update arms completed, their initial BF16 logits were
exactly equal on all eight bound prompts, and a 4.40 GB trained
candidate BF16 bank was exported. The run then raised at the frozen
candidate maximum/mean route-load limit of 50. The candidate therefore
did **not** earn the METH-137 fresh quality audit. Its distinct B rows
and training losses are not a claim of useful extra capacity.

The [protocol](METH_136_MATCHED_E12800_SPARSE_TRAINING_PROTOCOL_20260928.md)
was committed at `e5b8177` before the run, and the
[trainer](../../../benchmarks/donor_adaptation/s1/meth136_matched_sparse_train.py)
at `58c2d40`. It binds the BF16 Qwen2.5-0.5B-Instruct donor, METH-56
parent, METH-107 child, METH-126 exact shared-A bank and METH-135
third-tier sidecar by their hashes in that protocol. Both arms started
from the same exact centered E1280 BF16 function. The control retained
1,280 children; the candidate selected among 12,800 grandchildren
with frozen seeded third-tier keys. They used the same 256 deterministic
raw/chat draws and B-only selected-row Adam updates. No T4 was used.

The run ended after 1,627.016 seconds on the local RTX 3060, with
33,908,428,800 bytes final process RSS and 5,040,299,520 bytes peak
allocated GPU memory. These are below the 45-minute, 50 GiB and
10.5 GiB stops. The [partial report](meth136_matched_sparse_train_result.partial.json),
SHA-256 `068c19a2fa34b95cca2a357674a1f7e1185ea04095593d54eec1e892ecb29be6`,
retains the full control arm and runtime. The
[candidate progress](meth136_matched_sparse_train_result.candidate.progress.json),
SHA-256 `1082800fbc5d363e7b824aa0b9e8062e498509a26d90ddf952b3c9737cf9afac`,
retains all 256 updates, draw-linked losses, coverage and peak memory.

| Frozen training measure | Continued E1280 | Trained E12800 |
|---|---:|---:|
| Initial eight-prompt BF16 logit maximum error against teacher | 0 | 0 |
| Completed updates | 256 | 256 |
| Per-layer selected-row coverage | 1,183–1,278 | 7,996–11,488 |
| BF16-distinct rows after export versus source child | 1,183–1,278 | 11,820–12,780 |
| Exported bank bytes | 440,401,944 | 4,404,019,224 |
| Last-64 raw objective mean, training draws | 0.768630 | 0.768568 |
| Last-64 chat objective mean, training draws | 0.069905 | 0.069129 |

The control bank SHA-256 is
`d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299`;
the candidate bank SHA-256 is
`61570f9efb291fcd43e5bb83c4f0ac18b7fbeb8e44e4ce80791efbd80b56808a`.
The control export recorded exact binary readback. The candidate
binary is full-sized and its hash, header and BF16 rows were checked
separately in [METH-138](METH_138_FAILED_ROUTE_LOAD_DIAGNOSTIC_RESULT_20260928.md).
These large banks are local evidence artifacts, not committed source.

The candidate passed its 6,400 selected and 3,200 BF16-distinct
minimums per layer. Its exact train-time maximum/mean skew was not
written before the assertion, so the saved run proves only that the
frozen `<=50` skew gate failed. The control's exact train-time maximum
was 56.417, with layers 17 and 1 above 50. METH-138 replays both final
artifacts on the same draws to localize the failure, clearly labelled
as a final-state diagnostic rather than the train-time gate value.

The [METH-137 protocol](METH_137_MATCHED_E12800_EXTERNAL_QUALITY_PROTOCOL_20260928.md)
and source selector were frozen in advance but were **not run** on this
rejected bank. No new-source BPB, donor agreement, generation, PIQA or
blind grounding result exists for these artifacts. The earlier
METH-135 1.350× CPU routing cost remains a component result for the
seeded third tier; it does not validate the skewed learned artifact,
compact LUT arithmetic, full-model accepted-token speed, or 10B/100B
transfer. A new router design must be registered and tested separately.
