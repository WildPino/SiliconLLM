# METH-34: 15-level LUT factors improve fidelity but miss the fixed gate

The [prospective protocol](METH_34_I4_LUT_FACTOR_PROTOCOL_20260926.md)
replaces both trained rank-8 A/B factors of the Qwen R8+E128
adapter with per-row max-scaled 15-level codes (`q ∈ [-7,7]`,
stored index `q+7`) in tile-major LUT order. The fine router and
factor-0.50 amplitude remain unchanged. The local exported
`results/native_expert_scaling/meth34_qwen05b_i4_lut_adapter.safetensors`
is 132,229,896 bytes, SHA-256
`f4b77f2cd47d8c220872e2609e9ddb01068f4766cd25b7f922fd1ec2a32dc6d8`.
All 120 tensors passed exact reload and code decode checks before
quality scoring. The [export ledger](meth34_i4_lut_factor_export.json),
SHA-256 `c7c377685a277e382cca157b1ea649c3512c7de12885219756f2ab2d85684abe`,
records 110,100,480 code bytes, 11,108,352 scale bytes and
11,010,048 unchanged router bytes. Selected factor code+scale
addresses are **3,787,776 bytes/token** at L24/top-4, below the
fixed 4.2 MB ceiling. Relative weight squared error is 1.963%
for A and 0.492% for B; this is a weight-only statistic.

The intact R8+fp32 baseline reproduced all 48 METH-25 document
nats within 0.01 and all 24 top-1 streams exactly. The audit
then decoded the saved 15-level artifact and scored the same
already observed texts. Results are diagnostic, not independent
evidence for promotion.

| Reused category | Intact BPB | 15-level BPB | 15-level minus intact | 15-level minus original BF16 donor |
|---|---:|---:|---:|---:|
| Code, 24 docs | 0.907641 | 0.907589 | −0.000052 | +0.005254 |
| Prose, 16 docs | 1.095427 | 1.095542 | +0.000115 | −0.002291 |
| Technical/general, 8 docs | 0.591005 | 0.591343 | +0.000338 | −0.024456 |
| Pooled, 48 docs | 0.917465 | 0.917534 | **+0.000069** | −0.002213 |

The 20,000-draw category-stratified one-sided 95% upper bound for
the BPB penalty is +0.000241. Byte, paired BPB and donor-relative
screens pass. However next-token top-1 agreement is only
**6,041/6,144 = 98.324%**, short of the fixed ≥99% gate by 42
matching positions. The exact fine-route top-4 set stays the
same in **138,631/147,456 = 94.015%** of input-layer cases;
these are trajectory changes from factor quantization, not an
approximate-router recall test.

The [full audit](meth34_i4_lut_factor_audit.json), SHA-256
`01aa60ac7dbc42d861b38969367c1f9d4ffe0545df7795eeb07197dc9aea5708`,
retains every paired document score and the route/top-1 counts.
Export:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth34_export_i4_factors.py --out results/native_expert_scaling/meth34_qwen05b_i4_lut_adapter.safetensors --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth34_i4_lut_factor_export.json`.
Audit:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth34_i4_factor_audit.py --artifact-sha256 f4b77f2cd47d8c220872e2609e9ddb01068f4766cd25b7f922fd1ec2a32dc6d8 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth34_i4_lut_factor_audit.json`.
Export took 2.953 s CPU and ended at 1.536 GB RSS. The local
RTX 3060 audit took 46.796 s, peaked at 2.439 GB allocated GPU
memory and ended at 3.481 GB RSS. No T4 was used.

**Decision:** reject this exact 15-level weight transform under its
prospective top-1 gate despite a much smaller error and near-flat
reused-document BPB. This does not prove useful generation or
tasks worsened; they were not run here. The PyTorch audit decoded
weights but did not quantize activations, execute a C LUT builder,
or measure actual-factor CPU rate. Further factor-code tuning on
these reused texts would add little to the current decision while
the parent R8+E128 artifact still fails generation and no
large-E router passes fidelity. Return to those two blockers before
promoting any compressed factor format.
