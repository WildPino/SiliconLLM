# H5 — H4 rank-48 plus H2I one-byte FFN, cross-composition result

**Date:** 2026-09-16. **Status:** terminal CPU-fp32 diagnostic;
`POSTHOC_TRANSFER_SIGNAL=false`. No training, T4, `engine.c` export, or tok/s measurement.

The [brief and decision rule](../briefs/BRIEF_H5_CROSS_COMPOSITION_DIAGNOSTIC.md) were
committed at `7d6fe61` before the runner loaded a model. The committed runner
[`h5_cross_composition.py`](../../../../benchmarks/donor_adaptation/s1/h5_cross_composition.py)
passed a separate hash/HEAD preflight before the full evaluation. Its write-once
[result JSON](../../../../benchmarks/donor_adaptation/s1/results/h5/h5_cross_composition.json)
has SHA-256 `f66dc4d814617b69e2ebcef9a60031998bc8b20dac9fc75f9bffa848c2883205`.
The result records HEAD `7d6fe61b348afd83058b9de8d9d13e452eb0b8c2`, hashes of
the H4/H2I checkpoints and sidecars, canonical evaluator identities, donor revision,
and held-out token-ID hash. Exactly 24×512 held-out sequences (51,870 scored bytes)
and five E6 prompts ×32 tokens were used for both arms.

| Same Qwen2.5-1.5B donor, same held-out/reference | A: H4 rank-48 q/o | B: A + H2I trained R8 FFN on 8 layers | B − A |
|---|---:|---:|---:|
| BPB | 1.15373752359353 | 1.39916797602759 | **+0.24543045243406** |
| Free exact /160 | 6 | 1 | **−5** |
| Teacher-forced exact /160 | 66 | 55 | **−11** |
| Mean target rank (lower better) | 108.725 | 203.06875 | **+94.34375** |
| Rank≤5 /160 | 112 | 94 | **−18** |
| Arm evaluation wall time | 277.69 s | 290.78 s | not a decode-rate comparison |

Arm A reproduced the canonical H4 terminal BPB within the frozen `1e-5` tolerance,
and reproduced free, teacher-forced, mean rank and rank≤5 exactly. Therefore the
`VOID_APPARATUS` condition did not apply. Arm B installed exactly the H2I v4 trained
R8 masters, fp32 routers, labels and hard top-16 selection on layers
`[3,6,9,12,15,18,21,24]`; the other 20 FFNs stayed donor fp32. No q/o, tokenizer,
eval slice, or reference trajectory changed between arms. The decoded Arm B outputs
show repeated words/phrases and code-token loops; the text is stored in JSON, not
promoted as an independent metric.

The registered priority clauses all fail: BPB did not fall by ≥0.05 (it rose by
0.24543), teacher-forced did not rise by ≥10 (it fell by 11), and free did not
reach ≥15/160 (it fell to 1). The registered prediction that the **joint**
`POSTHOC_TRANSFER_SIGNAL` would not fire is **HIT**. This is an adverse transfer
result for **frozen, independently trained components**; it is not evidence that
fresh *joint training* cannot work. H2I's earlier BPB `0.905344` was measured with
H0's rank-512 q/o representation, not H4's rank-48 representation, and its
reference trajectory differed. The H2I result cannot be substituted as an Arm B
control here.

**Decision:** do not export or GPU-train this frozen H4+H2I composition as though it
were already a deployable model. Any new attempt should begin with a jointly
specified geometry and its own step-zero control, then train/evaluate that exact
geometry. H5 does not reopen H2I's terminal `SCORE-ONLY` gate and does not prove
anything about a 10B transfer or 50 tok/s. The remaining goal is the same:
useful pretrained quality and rate on the *same* `engine.c` artifact.
