# METH-88/89: B-only adaptation passes fresh development gates

The [frozen METH-88/89 protocol](METH_88_QUANT_AWARE_ADAPTATION_PROTOCOL_20260927.md) continues the existing distinct E128 product-key experts on the exact METH-85 stored grouped-R8 core. The teacher is the BF16 donor plus the METH-56 update-512 expert bank. Only B output factors are updated; the core, router and A factors remain fixed. The bound core SHA256 is `c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a` and the parent expert checkpoint SHA256 is `8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`.

The [training report](meth88_quant_aware_b_adaptation_result.json) records all 128 raw/chat draws and 64 updates. The resulting local checkpoint SHA256 is `3e39c6557afcd072c8a0f232880f9bd3061baa937b43bc88d48ad68f709a36b5`. One apparatus-only first attempt failed before any update because global gradient mode was disabled; explicitly enabling it repaired the run without changing the training rule. The completed run took 56.51 s on the local RTX 3060 and peaked at 3.47 GB allocated GPU memory. A/router identity is checked during evaluation. The METH-85 ideal active-byte count remains 557,106,176; changed B values do not add parameters.

The [METH-89 manifest](meth89_quant_aware_dev_manifest.json), SHA256 `6330eb7e0683bcfdc89da6712d98ba0e994dbea56c91b26230f5ad756ec060f8`, freezes eight code, eight PG19 prose and eight technical sources, excluding prior source IDs and overlapping 256-byte fragments through METH-86. The [raw evaluation](meth89_quant_aware_composition_result.json) binds donor, core, parent/adapted checkpoints and every token sequence. On 98,279 bytes and 4,311 prompt positions:

| Arm | BPB | Donor prompt top-1 |
|---|---:|---:|
| BF16 donor | 1.219246 | 100.000% |
| BF16 + original E128 | 1.214427 | 96.381% |
| Grouped-R8 core, experts disabled | 1.219509 | 94.572% |
| Grouped-R8 core + original E128 | 1.214675 | 94.410% |
| Grouped-R8 core + adapted E128 | **1.214686** | **95.430%** |

The adapted model improves 44 donor top-1 positions versus the original bank on the same core. It remains 0.951 point below BF16+original E128, inside the frozen one-point allowance. Its donor-relative BPB delta is −0.004560, and its 0.004823 BPB benefit over the expert-disabled core passes the ≥0.001 expert-utility gate. All six precommitted pooled/category gates pass. This is **development-only promotion to an independent external audit**; no generation, task, blind semantic, C-path or accepted-token rate claim follows from it. The external source and answerability plan is frozen in [METH-90](METH_90_QUANT_AWARE_EXTERNAL_PROTOCOL_20260927.md).
