# METH-210: exact tied head clears Q8 development and K64 choice gates

The [protocol](METH_210_Q8_EXACT_HEAD_PROTOCOL_20260930.md) and
[runner](../../../benchmarks/native_expert_scaling/meth210_q8_exact_head_diagnostic.py)
were frozen at `f855630`. BF16+E1280 and full tied-R8 Q8+E1280
reproduce every METH-194 document NLL and prompt donor-match count.
The experiment isolates embedding/head precision with explicit tied
or untied parameter pointers; FFNs remain the exact stored METH-193
grouped-Q8 representation, and centered E1280 experts are unchanged.

| Q8 arm | Pooled BPB | Donor prompt top-1 | Difference vs BF16+E1280 |
| --- | ---: | ---: | ---: |
| Tied R8 embedding/head (METH-194) | 1.184919 | 93.391% | -1.113 points |
| Exact embedding, R8 head | 1.184926 | 92.679% | -1.825 points |
| R8 embedding, exact head | 1.184820 | 94.281% | -0.223 points |
| **Exact tied embedding/head (fixed candidate)** | **1.184879** | **93.814%** | **-0.690 points** |

The preselected exact-tied candidate passes every original METH-194
development gate: pooled BPB difference -0.000009, category differences
between -0.000113 and +0.000118, and code/prose/technical top-1 losses
0.805/0.794/0.494 points (<=2), with pooled loss <=1 point. The
embedding/head interaction is not additive: restoring embedding alone
worsens this ranking, while exact head alone has the best diagnostic
ranking. The candidate was not selected after these outcomes.

The original R8 proposal codes with FP16-rounded row scales omit
zero full-BF16 top-1 choices at K=64 across all 4,494 candidate prompt
states. Exact BF16 selected-row recomputation with lowest-ID tie
breaking also gives zero mismatches. Maximum selected-row score
difference from the large head matmul is 0.125; choice parity is a
finite-state finding, not probability or universal reduction parity.
Every likelihood above uses the full head of its arm.

The proposed two-pass E1280 active ledger is 559,794,176 bytes/token,
205,824 below 560 MB, including a conservative exact embedding lookup.
Rounding proposal scales to FP16 saves 303,872 bytes per scan; keeping
the original BF16 head adds 272,269,312 resident bytes. This is a
layout calculation. It does not yet price native precision choices
or add a third-level E12800 router, and no CPU rate was measured.

The [result](meth210_q8_exact_head_diagnostic_result.json) SHA-256 is
`50ae15a1df4803277e24de79e28fd952dd850fb1d3015007ac8258f795a22471`.
Proposal-code SHA-256 is `2277b329bb42fe1241a5ee178129ca9b2ad4fa3f0b9279ece83d197a3c7ed79b`;
FP16-scale bits SHA-256 is `3811080fa83a92bec8f6aa851a4805ed09a6398d1178003247e670f304f63735`.
Runtime was 78.875 s on RTX 3060, peak allocated GPU memory 4.485 GB
and ending RSS 2.293 GB. No T4, training or new quality sources.

**Decision:** create a hash-bound stored core with exact BF16 tied
embedding/head, unchanged Q8 FFNs and the FP16-scale R8 proposal.
Only then test independent quality/generation and implement native
composition with actual precision/traffic accounting. This development
pass is not an accepted >=50 tok/s artifact. The failed shared-key
E12800 router and useful specialist-count scaling remain separate
open requirements; do not silently promote those candidates.
