# METH-62: direct donor-relative quality of a stored mixed Instruct core

**Uncertainty and decision.** METH-60's BF16-attention/R8-head+FFN
composition addresses an ideal 548.320 MB/token but misses its ≥95%
top-1 proxy by five positions. METH-61's weight-L2 scale correction
worsens that proxy. The actual goal is useful donor-relative quality
and native rate on the same compact artifact. Test the original METH-60
precision map on **new source-disjoint content** with direct document,
generation, task and blind semantic endpoints. The METH-60 top-1 gate
remains failed and cannot be retroactively reclassified. A direct-quality
pass would authorize C integration of this specific mixed artifact;
full-model rate, DRAM traffic and large-E learned quality remain separate.

**Bound model and representation.** Original Instruct source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
METH-56 update-512 product-key checkpoint SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`.
Store the tied embedding/head and every FFN 2-D matrix with the exact
METH-59 R8 per-row code/scale rule, all attention 2-D matrices as BF16,
and every 1-D control as FP32 in one packed-only safetensors file. Do
not reoptimize codes/scales on prompts. Require exact tensor/metadata
readback, source/checkpoint hashes, 169 matrices and 121 controls, and
≤560 million ideal addressed core+product-key-router+top-four BF16
expert bytes. Record physical file bytes separately.

**Frozen external source design.** Before model inference, select 8 code,
8 prose and 8 technical/general 4 KB source spans with seed
`meth62-direct-62062`. Code comes from `.py/.c/.h` files under
`benchmarks/` at Git commit
`882bb43f9118df1e4f79f85a6072111897bede9e`; technical/general
comes from Markdown under `docs/` at the same commit, excluding
`docs/research/NATIVE_EXPERT_SCALING_20260925/`. Prose comes from the
100-row local PG19 test parquet SHA-256
`9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc`.
Rank source IDs by SHA-256 of seed/category/source ID. Derive each
span offset from that hash. Exclude all METH-17/19/25/41/45/57 IDs,
the Qwen calibration/held-out text, and any first/middle/final
256-byte fragment overlap with those prior texts or selected METH-62
spans. Require ≥4,000 UTF-8 bytes and ≥256 document tokens. Save
source commit or parquet physical row, full-source hash, span offset,
text/token hashes, 384-character excerpt and chat prompt IDs. Screen
all 24 excerpts for a two-sentence summary and one specific detail
before inference. If a category cannot supply eight usable sources,
stop and amend this protocol before seeing either arm's responses.

**Direct gates.** On these frozen 24 documents, compare original BF16
Instruct donor with the stored mixed-core+METH-56 student. Require
pooled document ΔBPB≤+0.02 and each category≤+0.04. For 24 greedy
128-token chat responses, require student EOS count≥donor−2 and no
category more than donor+1 in repeated 8-gram 3× or short non-EOS
outputs. On all 1,838 PIQA items, require accuracy delta≥−2
percentage points and paired bootstrap fifth percentile≥−5 points.
Record prompt-position top-1 overall/by category as a diagnostic;
there is **no new top-1 promotion gate** in this direct-endpoint
experiment. The earlier METH-60 top-1 failure remains part of the
artifact record.

**Blind semantic gate.** After generation, make an A/B blinded file,
ordering donor/student by a fixed hash of source ID. A single reviewer
counts unsupported factual claims about the 384-character excerpt,
severe unsupported named-entity/numerical/comparative/causal claims,
and missing requested specific detail, quoting a minimal excerpt span
for each. Ambiguous cases are separate. Freeze and commit verdicts before
unblinding. Require student counts≤donor counts in each category of
finding. Report raw counts and limitations of single-reviewer judgment.

**Cost and stop.** Local RTX 3060 only, ≤10.5 GiB peak allocated GPU,
≤20 GiB RSS, ≤15 minutes for mixed export and ≤30 minutes for automatic
external scoring. Stop on a budget or early automatic gate failure and
preserve partial records. No T4. Only an automatic **and** blind semantic
pass permits native integration of this mixed E128 artifact; neither
would yet prove ≥50 accepted tokens/s or a useful 10× distinct-expert
ladder.
