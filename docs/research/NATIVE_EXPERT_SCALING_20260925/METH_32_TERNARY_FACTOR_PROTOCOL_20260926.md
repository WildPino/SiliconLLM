# METH-32: stored ternary rank-8 factor compatibility screen

**Uncertainty.** METH-31 measures a fast CPU ternary LUT path in the
Qwen rank-8 shape, but its codes are synthetic. This cell tests
whether the distinctly trained E128 factors from the quality-screened
R8+adapter artifact can be stored in the exact two-trit-per-byte
tile-major code layout consumed by `engine.c`, and whether their
language quality remains close to the original fp32 factors.

**Bound inputs.** Use only the METH-24 packed R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`
and METH-19 factor-0.50 E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
The source model/tokenizer identities are those verified in METH-25.
For the initial diagnostic reuse the 48 METH-25 documents, manifest
SHA-256 `20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77`,
and its published result SHA-256
`025c333a9edb4dcf73f6e421f91cdf54ead8cd11351e3fa17d8a6956d7e56a08`.
These texts are already observed; a pass cannot promote the packed
factors without a new disjoint document and generation audit.

**One weight-only transform.** For each row of each trained A
`[E,8,896]` and B `[E,896,8]` factor, sort absolute weights. For
every retained-count k≥1, set ternary signs on the k largest
weights and one row scale to their mean absolute value; choose
the k minimizing row squared error (equivalently maximize squared
top-k absolute sum / k). Break exact ties toward smaller k; an
all-zero row gets scale 0 and all-zero codes. Pair adjacent input
trits into byte code `(q0+1)*3+(q1+1)` in tile-major `[E,T,Mpad]`
order, padding A output rows from 8 to 32 with zero pairs (code 4).
B has 896 rows, already divisible by 32. Store codes, fp32 row
scales and the unchanged fp32 router in a single safetensors
adapter. Export first, reload and decode it, and pin its hash
before scoring. No activation calibration, finetuning, amplitude
change, route change, or threshold search is allowed.
The exported local artifact
`results/native_expert_scaling/meth32_qwen05b_ternary_lut_adapter.safetensors`
is 77,179,624 bytes, SHA-256
`a99542407aea7f1ae9eb7ac0401f9406e7c0cb1516129955c427538950cf7eee`.
Its [pre-score export ledger](meth32_ternary_factor_export.json) records
120 stored tensors, 55,050,240 code bytes, 11,108,352 scale bytes
and 11,010,048 unchanged router bytes. Export and reload succeeded
before any quality scoring.

**Diagnostic quality comparison.** Reconstruct the stored R8 core
into BF16 PyTorch matrices as in METH-25. Score intact fp32-factor
and reloaded ternary-factor adapters on the same 48 documents with
the METH-17 full-document EOS-prefix/512-target/512-left-context
scorer. Compare pooled/category BPB and each document. Verify that
the intact arm reproduces METH-25's per-document nats within 0.01.
On its 24 fixed 256-token prefixes, compare next-token top-1 for
6,144 positions. Use 20,000 category-stratified document bootstrap
draws with seed 323232 for the one-sided 95% upper bound of the
ternary-minus-intact BPB penalty.

The diagnostic gate passes only if all hold: pooled BPB penalty
≤+0.001, code ≤+0.002, prose and technical/general each ≤+0.003;
one-sided bootstrap upper bound ≤+0.003; next-token top-1
agreement ≥99%; and ternary adapter retains the METH-25 donor
screen (pooled ΔBPB≤+0.01, code≤+0.01, prose/technical≤+0.03
versus BF16 donor). A pass permits a new independent quality audit
and native export work, not a model-quality or 50 tok/s claim.
A fail identifies which factor precision/layout variant must change.
Report exact active bytes and row-error statistics separately from
quality; small squared error alone cannot pass the gate.

**Budget/stop.** Local CPU export and RTX 3060 quality audit only;
no T4. Limit each stage to 15 minutes, 20 GiB RSS and 10.5 GiB
allocated GPU memory for the audit. If the candidate or controls
cannot be reconstructed faithfully, classify the apparatus as
invalid and do not score or tune on these documents.
