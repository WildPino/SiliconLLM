# METH-83: fresh-source grouped-Q4 core and learned E128 composition

**Question.** METH-82 stores a verified 527.375 MB Instruct donor
core with BF16 tied head/attention and grouped-Q4 FFN. Determine
whether this representation preserves donor-relative likelihood and
prompt decisions when composed with the independently quality-audited
METH-56 E128 product-key adapter. The candidate's ideal core, router
and selected-factor payload is 535.740 MB/token from METH-81; speed
and real DRAM traffic are separate gates.

**Bound arms.** Use the exact BF16 Qwen2.5-0.5B-Instruct donor,
METH-56 update-512 E128 checkpoint and METH-82 Q4 artifact, with
all SHA-256 values checked before inference. Score four arms on the
same inputs: BF16 donor, BF16 donor+E128 adapter, stored-Q4 donor,
and stored-Q4 donor+E128 adapter. Do not retrain or modify factors,
routes or the packed core. Reconstruct only the saved Q4 FFN matrices
to BF16 for the PyTorch development scorer, and verify all 72
reconstructions against the saved format before scoring.

**Fresh development sources.** Before inference, freeze 8 code,
8 PG19 prose and 8 technical/general 4 KB spans with seed
`meth83-q4core-development-83083`. Reuse the bound source commit and
PG19 physical test parquet from METH-72, excluding source IDs and
first/middle/final fragment overlaps from METH-17/19/25/41/45/57/62/72,
calibration/held-out texts, and other selections within this set.
Require ≥4,000 UTF-8 bytes and ≥256 source tokens per document.
Form the existing Qwen chat prompt using each excerpt's first 384
characters. Verify source/text/token hashes.

**Fixed development gates.** For the Q4+E128 arm against BF16 donor,
require pooled document ΔBPB≤+0.02 and each code/prose/technical
category ≤+0.04; prompt-position donor top-1 ≥95% pooled and ≥90%
per category. Also require its pooled prompt top-1 to be no more than
1.0 percentage point below the BF16+E128 arm on the same prompts.
Report all four arms, document rows and prompt counts regardless of
gates. Failure stops this representation before generation/task/semantic
audit and native integration. Passing only licenses a new,
source-disjoint automatic generation, task and blind semantic audit;
it does not prove complete `engine.c` rate, CPU LUT cost, 10× learned
expert quality or multi-family transfer.

Local RTX 3060 only. Stop at 10.5 GiB allocated GPU, 20 GiB RSS,
15 minutes or nonfinite scoring. Preserve partial rows on failure.
No T4.
