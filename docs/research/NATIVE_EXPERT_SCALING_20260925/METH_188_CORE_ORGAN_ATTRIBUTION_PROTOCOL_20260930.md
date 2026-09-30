# METH-188: compact-core organ attribution (frozen protocol)

The METH-187 stored Q6 core plus centered E1280 bank misses the prompt
ranking development gate by 5.318 points. This experiment isolates the
already stored R8 tied head/embedding and grouped-Q6 FFN changes before
choosing which low-rank correction to train. It is a diagnostic on the
**same 24 already viewed METH-121 sources**, not a new quality test.

## Fixed inputs and arms

Bind the Qwen2.5-0.5B-Instruct BF16 source and revision, METH-122 parent
and child checkpoints, METH-121 manifest, METH-186 stored core and
METH-187 result by their existing SHA-256 assertions. Keep centered
E1280 experts enabled in every arm. The arms are:

1. BF16 source head and FFNs (reference);
2. stored R8 tied head/embedding, BF16 FFNs;
3. BF16 tied head/embedding, stored Q6 FFNs;
4. stored R8 tied head/embedding and stored Q6 FFNs (METH-187 repeat).

Attention and controls stay at their BF16 source values. Apply stored
values to the same model in sequence, restoring the exact BF16 tied
head between the two mixed arms. The donor top-1 sequence is recorded
before any core change. Compare source document BPB and donor prompt
top-1 agreement for pooled, code, prose and technical categories.
The reference and fourth arms must reproduce METH-187 per-source
negative log likelihood and prompt match counts exactly. Abort on a
binding or parity mismatch.

## Decision and limits

For each arm, calculate the ranking loss in percentage points versus
the BF16+E1280 reference. If head-only accounts for at least 80% of
the full-core ranking loss while Q6-FFN-only accounts for at most 20%,
prioritize a tied-head correction. Apply the symmetric rule for FFN.
Otherwise use joint correction. Report the interaction separately:
full-core loss minus the two isolated losses. These are planning rules,
not promotion gates; the next stored corrected artifact must still
pass the frozen METH-187 development limits, fresh source-disjoint
quality, and native same-artifact rate. Do not spend new holdout sources
on this attribution.

Run only on the local RTX 3060, with 15 min wall, 10.5 GiB peak GPU
allocation, 20 GiB RSS, and under 1 GB output. Stop on any limit.
No T4 is authorized by this protocol.
