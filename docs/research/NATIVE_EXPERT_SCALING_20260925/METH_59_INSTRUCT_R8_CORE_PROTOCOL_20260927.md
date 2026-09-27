# METH-59: packed Instruct core composed with the product-key adapter

**Uncertainty and decision.** METH-57 establishes donor-relative E128
quality, and METH-58 exports its learned router and experts to C. The donor's
BF16 core addresses about 988 MB/token and a FP32 tied head alone about
545 MB/token. METH-22 shows a one-byte head/body can fit a *conditional*
traffic budget; METH-23–25 show that the exact R8 per-output-row rule can
preserve document/ranking quality on the different Qwen0.5B **base** donor.
Test whether that same frozen rule works on the bound 0.5B-Instruct donor
with the actual METH-56 product-key adapter. A fail rejects this precision
map for native promotion and triggers a targeted head/body precision
diagnosis; a pass permits an independent quality audit of the combined
stored artifact, then native model integration. It does not itself prove
50 accepted tok/s or large-E quality.

**Bound inputs.** Instruct source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
METH-56 checkpoint SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`;
METH-57 24-document manifest SHA-256
`9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75`.
Use `t2_rules.r8_int8_rtn` on the tied embedding/head and every 2-D
attention/FFN matrix, storing signed one-byte codes and one FP32 scale per
output row. Preserve 1-D controls as FP32. No post-hoc scale tuning. Export
to a packed-only safetensors file with all tensor, metadata and byte-level
reload checks. Require ≤520 million physical file bytes and exactly the
source model's 169 matrices and 121 controls.

**Diagnostic quality gate.** Compare the stored R8 core plus fixed METH-56
adapter to the original BF16 Instruct donor on METH-57's *already viewed*
24 external documents, prompts and 1,838 PIQA items. Also measure R8 donor
alone to isolate core damage. The document gate is pooled ΔBPB≤+0.02 and
each of code/prose/technical≤+0.04; prompt-position top-1 versus original
donor is ≥95% pooled and ≥90% in each category; full PIQA accuracy delta
is ≥−2 points, with paired bootstrap fifth percentile ≥−5 points. For
24 greedy responses, require student EOS≥donor−2, and no category more than
donor+1 for repeated 8-gram 3× or short non-EOS. Save all paired raw rows,
resource use and exact commands. This source set was inspected for METH-57;
even if these diagnostic gates pass, freeze a new source-disjoint semantic
and task audit before calling the compact composition quality-valid.

**Cost and stop.** Local RTX 3060 only, ≤10.5 GiB peak allocated GPU,
≤20 GiB RSS, ≤15 minutes for export and ≤25 minutes for diagnostic scoring.
Stop on a resource or stage-gate failure and preserve partial records. No T4.
The arithmetic one-byte head/body payload is about 496 MB including row
scales/controls on disk; this is not a measured DRAM trace or CPU rate.
