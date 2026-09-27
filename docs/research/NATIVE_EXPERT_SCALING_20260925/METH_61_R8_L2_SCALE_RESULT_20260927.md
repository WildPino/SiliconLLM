# METH-61: lower weight L2 does not recover donor prompt decisions

**Decision:** the fixed data-free least-squares R8 scale rule is rejected
for the BF16-attention mixed core. It slightly reduces reconstructed
head/FFN weight error but worsens donor prompt top-1 in every tested
arm. No arm reaches the ≥95% gate, so no modified core was exported.

The [protocol](METH_61_R8_L2_SCALE_PROTOCOL_20260927.md) fixed three
code-refresh iterations plus a final least-squares row scale before
the [runner](../../../benchmarks/donor_adaptation/s1/meth61_r8_l2_scale.py)
read the viewed METH-57 prompts. The BF16 adapter control reproduced
3,997/4,125 positions; the original BF16-attention/R8-head+FFN arm
reproduced METH-60's 3,914/4,125 positions. The
[raw result](meth61_r8_l2_scale_result.json), SHA-256
`8360ee7a9abe9d6ce40eed118ee177287763eb3579380e0508be28a74f66cd46`,
contains all per-prompt counts and organ reconstruction sums.

| BF16-attention core arm | Matching top-1 / 4,125 | Change vs original R8 scales |
|---|---:|---:|
| Original R8 head + FFN | **3,914 = 94.885%** | control |
| L2 scales for head | 3,884 = 94.158% | −30 positions |
| L2 scales for FFN | 3,890 = 94.303% | −24 positions |
| L2 scales for head + FFN | 3,834 = 92.945% | −80 positions |

Head squared reconstruction error falls from 2.67353 to 2.65521;
aggregate FFN error falls from 12.48065 to 12.41629. Better local
weight L2 therefore does not predict better token decisions in this
composition. All arms have the same ideal 548.320 MB/token payload.
The local RTX 3060 run took 12.938 s, peaked at 3.641 GB allocated GPU
memory and ended at 2.918 GB RSS.

This was a viewed-prompt mechanism test, not an independent generation,
task or semantic result. The BF16-attention arm remains below its old
top-1 gate. A new, source-disjoint audit of its actual stored mixed
artifact can instead test the final donor-relative quality requirements
directly; the old top-1 failure must remain visible in that decision.
