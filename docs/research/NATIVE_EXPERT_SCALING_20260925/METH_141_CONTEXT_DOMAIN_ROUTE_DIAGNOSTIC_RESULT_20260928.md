# METH-141: both source and context length shift the decile router

The fixed, rejected METH-140 sidecar becomes more imbalanced on longer
windows from the same source, and more imbalanced on external
documents even when window length is held at 128. Both effects are
visible; the source shift is larger in this six-cell comparison.
This is a diagnostic on viewed routing inputs, not a new promotion
gate or a quality result.

The [protocol](METH_141_CONTEXT_DOMAIN_ROUTE_DIAGNOSTIC_PROTOCOL_20260928.md)
was committed at `9ef2fb7` before replay, and the
[implementation](../../../benchmarks/native_expert_scaling/meth141_context_domain_route_diagnostic.py)
at `d8456b6`. It bound the unchanged METH-140 sidecar SHA-256
`a581272ed84c154b660a4fd9a7c108f0dcced394e0d763ddfdd366aa92eadc5f`
and used identical routing code in all six cells. Clone-factor
full-model BF16 logits were exactly equal on eight prompts, and each
grandchild mapped back to its original child. The
[result](meth141_context_domain_route_diagnostic_result.json), SHA-256
`06fa399b1f1c717a20284bb8ec327aaba3582c089780f747876e01f6c728322f`,
retains all 24 layer rows per cell. It took 266.406 seconds on the
local RTX 3060, 3.949 GB final process RSS and 3.023 GB peak
allocated GPU memory; no T4 was used.

| Source, window | Tokens | Worst candidate/control load ratio | Mean ratio over 24 layers | Worst hot-parent share |
|---|---:|---:|---:|---:|
| H0 training raw, 128 | 32,768 | 1.670× | 1.33× | 43.2% |
| H0 training raw, 512 | 131,072 | 2.023× | 1.43× | 81.7% |
| METH-121 documents, 128 | 29,186 | 2.798× | 1.77× | 87.6% |
| METH-121 documents, 512 | 29,186 | 3.198× | 1.93× | 87.9% |
| METH-133 documents, 128 | 29,926 | 2.823× | 1.76× | 86.3% |
| METH-133 documents, 512 | 29,926 | 3.214× | 1.91× | 88.1% |

Within H0 raw, the 512-window ratio exceeds the 128-window ratio in
15/24 layers; within METH-121 in 19/24 and METH-133 in 15/24.
At the same 128-window length, METH-121 exceeds H0 raw in 21/24
layers and METH-133 in 22/24. At 512, those counts are 20/24 and
19/24. The H0 128 cell reuses the raw training source and is only a
positive control; its residual 1.670× worst ratio also shows that
METH-140's joint raw/chat deciles do not perfectly balance raw alone.

Changing context length and changing source each predict additional
concentration. The comparison does not prove a unique causal split:
the H0 cells use different token totals, the external content differs,
and the displayed worst layers may differ. It does rule out a simple
"more samples of the same short training mixture" repair for this
fixed scalar-decile router. A next mechanism should explicitly handle
source/context variation and be tested for learned useful capacity,
CPU cost and fresh quality before scaling claims.
