# METH-34: 15-level per-weight LUT factor export and diagnostic

**Uncertainty.** METH-32/33 reject one- or two-organ ternary
factor replacement under the fixed 99% top-1 fidelity gate,
despite acceptable reused-document BPB. METH-31 shows an affordable
synthetic selected LUT path at the Qwen rank-8 shape. This cell
tests whether a richer, still LUT-addressable weight codebook
preserves the actual trained factor behavior within the same
selected-byte envelope. It does not test large-E routing or
end-to-end C throughput.

**Bound inputs.** Use the METH-24 stored R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`
and METH-19 factor-0.50 E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Retain the exact fp32 router. The diagnostic reuses the 48 METH-25
documents, manifest SHA-256
`20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77`,
and the METH-25 result SHA-256
`025c333a9edb4dcf73f6e421f91cdf54ead8cd11351e3fa17d8a6956d7e56a08`.
They were already observed; a pass requires a separate independent
document/generation/task audit.

**Fixed weight transform.** For each A `[E,8,896]` and B
`[E,896,8]` row, set scale=max(abs(weight))/7. Quantize each
weight by round-to-nearest-even of weight/scale, clamp to
`[-7,+7]`, and store uint8 code `q+7` in tile-major
`[E,input,Mpad]` order. All-zero rows use scale 0/code 7.
Pad A's 8 outputs to 32 with code 7; B's 896 outputs need no
padding. Code 15 is reserved and invalid. Store fp32 row scales
and unchanged fp32 router in one safetensors adapter. Export,
reload, verify every tensor and decode, then pin the artifact
SHA-256 before quality scoring. No activation calibration,
training, amplitude change, route change or scale search.
The resulting local
`results/native_expert_scaling/meth34_qwen05b_i4_lut_adapter.safetensors`
is 132,229,896 bytes, SHA-256
`f4b77f2cd47d8c220872e2609e9ddb01068f4766cd25b7f922fd1ec2a32dc6d8`.
Its [pre-score export ledger](meth34_i4_lut_factor_export.json)
records all 120 reloaded tensors and the exact byte counts below.

The existing `engine.c` pshufb kernel accepts 4-bit lookup indices.
A future activation LUT for this format would use
`lut[t,code]=xq[t]*(code-7)` for code 0–14 and zero for code 15,
with `|xq|≤18` to avoid signed-byte overflow. This cell scores
**weight reconstruction only** in PyTorch; it does not claim
activation quantization fidelity, C parity or a native runtime
for the saved artifact. Those are required follow-up gates.

The fixed footprint is 110,100,480 factor code bytes,
11,108,352 factor scale bytes and 11,010,048 fp32 router bytes
at E128/L24, before safetensors framing. At L24/top-4, selected
factor codes+scales address 3,787,776 bytes/token, versus
5,505,024 for original fp32 factors. The byte gate is
≤4,200,000 selected factor bytes/token; it excludes the
exhaustive router and R8 core.

**Diagnostic quality gate.** Reconstruct the stored R8 core and
reproduce all 48 intact-adapter METH-25 document nats within
0.01 and all 24 stored top-1 streams exactly before scoring.
Then load only the saved 15-level factor artifact and score the
same full documents using EOS-prefix/512-target/512-left-context
BPB, plus 6,144 top-1 positions on the fixed prefixes. Use
20,000 category-stratified bootstrap draws, seed 343434, for a
one-sided 95% upper bound on packed-minus-intact BPB.
Pass only if selected bytes≤4.2 MB/token, pooled BPB penalty≤+0.001,
code≤+0.002, prose/technical≤+0.003, bootstrap upper≤+0.003,
and top-1 agreement≥99%. Also retain the METH-25 donor screen:
pooled ΔBPB≤+0.01, code≤+0.01, prose/technical≤+0.03 versus
original BF16 donor. Report fine-route set changes as a trajectory
diagnostic. A pass permits new independent quality checks and
native LUT activation/kernel work, not model promotion.

**Budget/stop.** Local CPU export and RTX 3060 diagnostic only;
no T4. Limit each stage to 15 minutes and 20 GiB RSS, with
10.5 GiB GPU allocation cap for the audit. If export/reload or
baseline controls fail, stop without interpreting quality.
