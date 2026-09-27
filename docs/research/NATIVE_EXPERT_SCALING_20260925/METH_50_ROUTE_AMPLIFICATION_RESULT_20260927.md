# METH-50: factor arithmetic still fails with original expert IDs

The [frozen diagnostic](METH_50_ROUTE_AMPLIFICATION_PROTOCOL_20260927.md)
uses the exact METH-47 update-512 checkpoint and METH-49 row-int8 factor
artifact on the same 12 previously viewed external prompts. The
[runner](../../../benchmarks/donor_adaptation/s1/meth50_route_amplification.py)
captured original and packed top-4 routes at all 24 layers and 2,091
positions. Its intervention replayed the original expert IDs with
reconstructed int8 factors while recomputing gate probabilities from the
intervened hidden states. Replaying the original IDs with original factors
gave **exact logit parity** on every prompt, validating the intervention.

| Arm versus original | Matching prompt top-1 | Interpretation |
|---|---:|---|
| Ordinary row-int8 factors and free routing | 2,002/2,091 = **95.744%** | Reproduces METH-49 exactly |
| Row-int8 factors with original top-4 IDs forced | 2,013/2,091 = **96.270%** | Recovers 11 net positions; still below the fixed 99% gate |

Original versus ordinary packed routes have **98.350% pooled top-4 ID
overlap** and **93.424% full-set agreement** across 50,184
input-layer cases. Layer 0 has exact parity because its input has not
yet seen a changed factor. The first changed route appears at layer 1
on 8 prompts and layer 2 on 4. Full-set agreement falls to 90.20%
at layer 18 and is 91.49% at layer 23. Thus factor rounding causes
downstream route changes, but suppressing those ID changes is
insufficient to retain the original token ranking. The forced arm is
a causal diagnostic, not an index that can be deployed on CPU.

The original fourth-versus-fifth fine-router score margin has median
0.08294 and first percentile 0.001131 on these positions. At least
115/128 experts are selected in each layer, yet the most loaded layer's
single expert receives 18.30× its uniform mean share. Use and load are
separate: most experts receive some traffic, but a few dominate it.
These E128 measurements cannot establish the margin, retrieval quality
or load at E27,355/E273,547 with **distinct learned** experts.

The [machine result](meth50_route_amplification_result.json), SHA-256
`73ce7f6d1dadb62e4758083527a3f9a4f5061aa860c7e40e9d327db8bd432acd`,
contains per-prompt and per-layer route/ranking counts, margin and load
statistics, and route-stream hashes. Reproduction command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth50_route_amplification.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth50_route_amplification_result.json
```

The local RTX 3060 run took 11.391 s, peaked at 1.435 GB GPU allocation
and 4.025 GB RSS; no T4 was used.

**Decision:** METH-49's factor arithmetic fails ranking even with exact
original expert IDs, while small factor changes also induce later route
churn. A better router alone cannot make this packed bank quality-valid.
Preserve an exact effective-factor anchor, and require direct factor
quality plus route stability in any new quantization-aware adaptation.
For scaling *n*, a jointly trained bounded router still needs a fresh
E128→larger-E route/quality audit and CPU timing; forcing old IDs offers
no evidence that a large choice set can be navigated cheaply.
