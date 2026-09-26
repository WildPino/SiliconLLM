# METH-22: Qwen0.5B donor-core traffic preflight

**Prospective exact-shape arithmetic.** METH-19/21 establish scoped
donor-relative document and PIQA retention for the factor-0.50 E128
adapter, but the donor core is intact. This preflight asks which
organ must change before a native artifact can plausibly reach
≥50 accepted token/s. It is not a latency benchmark or a
quantization-quality result.

Read the pinned Qwen2.5-0.5B source safetensors SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, and verify
its tied-head configuration and exact tensor shapes. Bind the
factor-0.50 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Classify the tied embedding/output-head matrix, FFN matrices,
attention matrices and control vectors. Because the head is tied,
charge **all vocabulary rows** every decoded token for its output
projection, although the input embedding uses just one row.

With E128/top-4/rank-8 and 24 layers, charge the full fp32 router
matrix per token and the selected fp32 expert A/B factors per token.
Compute ideal addressed-payload scenarios:

1. BF16 donor core plus fp32 adapter.
2. FP32 donor core plus fp32 adapter, matching the native donor
   export's dense format before compression.
3. Ideal one-byte FFN/attention matrices with fp32 tied head.
4. Ideal half-byte FFN/attention matrices with fp32 tied head.
5. Ideal one-byte FFN/attention/head matrices.
6. Ideal half-byte head and one-byte FFN/attention matrices.

Keep norms and biases fp32 in every scenario. Ignore scales,
format headers, metadata, memory-transaction granularity, compute,
cache misses, attention and dispatch overhead: each result is an
**optimistic lower bound** on addressed weight payload, not measured
DRAM traffic. Price bytes at a favorable **40 GB/s** and compare with
the complete **20 ms/token** budget and the existing **560 MB/token**
streaming design allotment (14 ms at that bandwidth). Do not infer a
50 tok/s pass from a payload below either bound. If an ideal
one-byte body with fp32 tied head already exceeds 20 ms under this
40 GB/s streaming assumption, the planned one-byte-body path must
compress the head or demonstrate a materially different memory rate;
if fp32 head alone
uses nearly all the 560 MB allotment, the same priority holds for
more aggressive body packing.

Run the local [ledger tool](../../../benchmarks/donor_adaptation/s1/meth22_qwen05b_active_ledger.py)
on source metadata only; no GPU, T4, quantizer or new model artifact.
Stop if source/adapter identity or shape assertions fail. The next
decision is which head/body precision transformation to test with
donor-relative quality and then native parity, while separately
designing sublinear routing for much larger E.
