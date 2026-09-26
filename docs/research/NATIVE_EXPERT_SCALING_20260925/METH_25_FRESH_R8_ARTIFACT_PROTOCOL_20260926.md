# METH-25: independent quality gate for the stored R8 core and E128 adapter

The packed R8 core is fixed at SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`;
the factor-0.50 E128 adapter is fixed at SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
No quantizer, adapter amplitude, routing geometry or document may be
changed after this protocol. The new 48-document manifest SHA-256 is
`20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77`:
24 code, 16 prose and 8 technical documents, 196,559 selected bytes.
It excludes all METH-17/19 source IDs and screens first/middle/final
256-byte fragments against Qwen calibration/heldout and METH-17/19
selected text, plus a selected-set cross-screen. Some common shorter
phrases and donor pretraining overlap remain possible.

Read the **stored** signed-int8 codes and fp32 row scales from the
R8 artifact. Reconstruct them into BF16 model matrices in PyTorch for
scoring, never re-quantize the donor source. Check that source metadata,
tensor counts, shapes, controls, head tying and hashes match. This
proves the tested quality belongs to the stored bytes but does not
exercise a native int8 kernel. Compare four arms:

1. Original BF16 donor core, experts disabled.
2. Stored R8 core, experts disabled.
3. Stored R8 core plus unchanged factor-0.50 E128 adapter.
4. Stored R8 core plus the same adapter with expert-route identities
   permuted using seed 252525; this is a route-utility control.

Use the METH-17 EOS-prefix, 512-target stride, 512-left-context,
absolute-position full-document BPB scorer. Report pooled/category
paired BPB, 20,000 category-stratified document-bootstrap draws
(seed 252526), and per-document deltas. The **document gate** passes
only if all conditions hold:

- stored-R8 donor pooled ΔBPB ≤ +0.01 versus original BF16 donor;
- stored-R8 adapter pooled ΔBPB ≤ +0.01 versus original BF16 donor;
- one-sided 95% bootstrap upper bound for that adapter Δ ≤ +0.02;
- stored-R8 adapter category Δ ≤ +0.01 for code and ≤ +0.03 for
  prose and technical text;
- permuted route worsens pooled BPB versus intact route by ≥ +0.002.

For a fixed 24-document subset (8/category, seed 252527), compare
the 256-token prefixes' next-token top-1 IDs, 6,144 positions, on
BF16 and stored-R8 donor and on BF16 and stored-R8 adapter. Require
≥95% exact agreement for both arm pairs as a separate ranking gate.
This subset is drawn only from the new manifest. Do not promote
based on documents alone: subsequent task/generation and C parity,
latency and large-E tests are required. Stop at 15 minutes, 10.5 GiB
GPU allocation or 20 GiB RSS on the local RTX 3060.
