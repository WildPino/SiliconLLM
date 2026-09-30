# METH-204: shared learned content keys retain signal but fail load

The [frozen protocol](METH_204_SHARED_CONTENT_KEYS_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth204_shared_content_keys.py)
were committed at `9d66939` before execution. The run bound the exact
METH-126 E1,280 bank, METH-135 rank-32 projection, METH-175 draws and
structural table, and METH-150 manifest by their recorded hashes.
Eight-prompt donor/control BF16 logit parity was exact. Nine global
content centroids per layer were fitted on the first 1,024 paired
raw/chat draws; 1,280×9 per-parent biases were then calibrated using
only those draws. The exact readback router artifact is local at
`results/native_expert_scaling/meth204_shared_content_router.npz`,
SHA-256 `10a8d695f51bff84bf1fcc0e90971a9fb332bfd3226eb059af0256770ae9d6e1`.

The preregistered **fit gate fails**. The runner correctly stopped
before the 256 reserved draws and 24 source-separated documents.
Consequently there is no held-out route, specialist-quality or native
speed evidence from this artifact.

| Fit cell, 24 layers | Content selections/layer | Layers failing load ratio ≤1.25 | Layers failing hot-parent share ≤25% | Coverage range | Score-advantage range | Raw-argmax agreement range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 509,520 | 19 | 24 | 9,358–11,253 | 0.997–1.504 | 45.94–69.10% |
| Chat | 613,836 | 19 | 24 | 9,459–11,328 | 0.999–1.512 | 44.09–70.86% |

Coverage exceeds the frozen 4,000-slot threshold in every layer, as
do the ≥0.05 standardized score-advantage and ≥15% un-biased argmax
agreement thresholds. Thus the route contains measurable state
dependence. Load remains concentrated: the worst candidate/control
ratio is 1.999 raw and 2.074 chat; the worst hot-parent child share is
87.24% raw and 43.41% chat. The observed per-layer maximum absolute
calibration bias is 0.329–0.669. Structural selections per layer were
10,672 raw and 288,128 chat and were excluded from all content load
gates. The chat structural path reduces repeated-token traffic, yet
the remaining content traffic still fails balance.

The [raw result](meth204_shared_content_keys_result.json), SHA-256
`af36a9d1ff119b1407c395c769afcf40b6023ef64ce430ca9a062c89ae59e160`,
contains all 48 fit-layer rows, gates, bindings and runtime. The
stored projection, shared keys and per-parent biases occupy
2,752,512 + 27,648 + 1,105,920 = 3,886,080 payload bytes; the
artifact is 3,886,864 bytes. Ideal addressed router bytes are
2,783,616 per token at 24 layers. Tenfold parent count would make
the stored bias 11,059,200 bytes while selected-bias reads remain
constant; this is accounting, not measured CPU LUT latency.

The local RTX 3060 run took 392.422 s, with 2.945 GB peak GPU
allocation and 4.341 GB ending RSS. The largest progress observation
was 7.845 GB RSS; this is not an instrumented process peak. No T4
or new source was used.

**Decision:** reject this exact globally shared nine-key plus
soft-count-bias route before B training. A next candidate must tailor
selection more strongly within each parent or address repeated
content states, while retaining explicit load and source-transfer
gates. The passing content-score metrics do not imply useful
specialists. An E12,800 quality-valid bank, CPU route/LUT cost,
10B/100B transfer and same-artifact ≥50 accepted tok/s remain open.
