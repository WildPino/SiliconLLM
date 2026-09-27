# METH-81: header-derived grouped-Q4 compact-core payload screen

**Question.** METH-62's BF16-attention/R8-head+FFN core fails blind
semantic conservation and charges 548.320 MB of ideal core, router
and selected-factor bytes per token. A more accurate grouped 4-bit
FFN with BF16 attention and tied head may remain under the same
560 MB design allotment. Verify exact candidate payload from the
bound donor tensor shapes before conversion or quality inference.

**Bound source and format.** Read only the header of the SHA-bound
Qwen2.5-0.5B-Instruct donor `model.safetensors` used by METH-59.
Classify the tied embedding/head, 72 FFN matrices, 96 attention
matrices, and 121 one-dimensional FP32 controls by the same METH-24
rule. A grouped-Q4 matrix uses signed symmetric 4-bit codes packed
two per byte, with one FP16 scale for each 64 contiguous input
elements per output row. Price exact row-wise padding if a width is
not divisible by 64. A BF16 override charges two bytes per weight.
Price controls at four bytes per element. Add exactly 8,404,992 bytes:
5,652,480 FP32 product-key bytes and 2,752,512 selected top-four BF16
factor bytes, as in the METH-60 byte ledger. This excludes KV state,
activations, tokenizer and allocator overhead.

**Precommitted arms and decision.** Price (1) all matrices grouped-Q4,
(2) BF16 attention with Q4 tied head/FFN, (3) BF16 tied head with
Q4 attention/FFN, and (4) BF16 tied head+attention with Q4 FFN.
The leading quality candidate is arm 4 if its ideal addressed bytes
are ≤560,000,000 and all tensor classifications/counts match METH-59.
If the gate passes, implement a packed exporter, exact stored
readback, BF16 reconstruction, and fresh development composition
test with the METH-56 quality-audited E128 adapter. If it fails,
do not spend a quality run on this map. No quality, real DRAM
traffic, native parity, or throughput claim follows from this screen.

Read-only local CPU preflight; ≤2 minutes wall time and ≤2 GiB RSS.
No T4 or GPU work.
