# METH-25: stored R8 core plus E128 adapter passes new-text quality screen

Command: `.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth25_fresh_r8_artifact_audit.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth25_fresh_r8_artifact_result.json`.
The [predeclared protocol](METH_25_FRESH_R8_ARTIFACT_PROTOCOL_20260926.md)
fixes the 496,122,224-byte core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`,
the 187,177,472-byte separate adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
and new 48-document manifest SHA-256
`20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77`.
The scoring model was reconstructed from the **stored int8 codes and
fp32 row scales**, not re-quantized from the donor. All 169 matrix and
121 control tensors, metadata and tied head passed load checks.

| Category | Docs | Original BF16 donor BPB | Stored R8 donor Δ | R8+adapter Δ vs original donor |
|---|---:|---:|---:|---:|
| Code | 24 | 0.902336 | +0.001195 | +0.005305 |
| Prose | 16 | 1.097833 | +0.000362 | −0.002406 |
| Technical | 8 | 0.615798 | +0.000417 | −0.024793 |
| Pooled | 48 | 0.919747 | +0.000788 | **−0.002282** |

The category-stratified 20,000-draw document bootstrap gives a
one-sided 95% upper bound of **−0.000909 BPB** for the pooled adapter
delta. Permuting router identities worsens pooled BPB **+0.007488**
versus intact routing. On 6,144 positions from new documents, stored
R8 versus BF16 exact top-1 agreement is **95.915%** for donor and
**95.996%** for adapter. Both predeclared document and ranking gates
pass. The code subset still worsens on average; the quality evidence
is bounded to this source and these evaluation slices.

The RTX 3060 scoring phase took 74.8 s after source selection, with
2.470 GB peak allocated GPU memory and 3.140 GB final RSS. PyTorch
dequantized matrices into BF16, so runtime is **not** compressed-model
throughput. Source-document IDs, per-document nats and top-1 vectors
are retained in the [result JSON](meth25_fresh_r8_artifact_result.json).

**Decision:** the stored artifact clears the independent document and
ranking screen. The method has a quality-valid compact core at 0.5B
scale, but downstream task/generation, native `engine.c` parity/rate,
packed-only E128 export and larger distinct learned expert counts
remain required. No ≥50 accepted tok/s or 10B/100B claim follows.
