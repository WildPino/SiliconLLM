# METH-47: stronger KL preserves development chat and passes automatic external gates

The [frozen protocol](METH_47_STRONG_KL_RETENTION_PROTOCOL_20260927.md),
[new development manifest](meth47_retention_dev_manifest.json) and
[continuation runner](../../../benchmarks/donor_adaptation/s1/meth47_strong_kl_continuation.py)
were committed before this run. The [training result](meth47_strong_kl_continuation_result.json),
SHA-256 `189c464f11bbb00245e56da57c63cd32b4e2d6e174cc079071f1ccae7d78f132`,
started from METH-44's bound 16-update checkpoint. Its new-development
chat top-1 was 3687/3814 = 96.670% at resume, 3690/3814 = 96.749%
at update 256, and **3748/3814 = 98.270%** at update 512. Final raw
student-minus-donor BPB was **−0.015638**. Permuting router rows raised
BPB by **+0.013181**; every layer changed all 128 output slots.
The update-512 optimizer/RNG checkpoint has SHA-256
`0c6f09562efff7e78921036fe6ab030d7377150fe9afbe06b3deb1c029e0aa59`.
Training took 736.750 s on the local RTX 3060, peak allocated GPU
2.583 GB and process RSS 3.872 GB, within the declared bounds.

Only after this terminal pass, the
[previously frozen external manifest](meth45_fresh_external_manifest.json)
was opened with the [paired evaluator](../../../benchmarks/donor_adaptation/s1/meth45_instruct_external_audit.py).
The [full external result](meth47_frozen_external_audit_result.json), SHA-256
`9640f086c00c03db7ab7ff28ae13436a3b8fdef18996d2313aab772ddc652c29`,
records all document scores, 12 paired donor/student continuations, all
1,838 PIQA choices and the router control. The independent readback
recomputed the pooled document delta and prompt agreement from raw rows.

| Fixed external gate | Measured donor | Measured student | Result |
|---|---:|---:|---|
| 12-document pooled BPB | 1.360221 | 1.355188 | Student −0.005033; all three category deltas also negative |
| Prompt-position top-1 | — | 2009/2091 = 96.078% | ≥95% pooled and ≥90% per category |
| Greedy chat EOS | 12/12 | 12/12 | No early non-EOS or repeated 8-gram 3× in either arm |
| Full PIQA correct | 1291/1838 | 1290/1838 | −0.0544 accuracy points; paired bootstrap 5th percentile −0.4897 points |
| Permuted-router BPB penalty | — | +0.013181 | ≥+0.002 route-utility gate |

All **predeclared automatic gates pass**. The paired external audit took
506.656 s, peak allocated GPU 2.437 GB and process RSS 3.498 GB. This
is stronger evidence for a 0.5B/E128 Instruct donor adaptation than
METH-44's same-corpus smoke. It is still a BF16 PyTorch donor plus
additive experts, not a compact `engine.c` artifact or a large-E point.

**Post hoc manual generation review, not a registered quantitative gate.**
The 12 saved responses also show limits that automatic repetition and EOS
checks cannot capture. For `s1_tables.py`, the student asserts that C3
has a higher `delta_bpb` than C4 although the shown prompt excerpt gives
no such comparison; the donor does not make that ranking claim. For PG19
row 51, the student calls Salome Sheba's sister and describes Tarver and
Darbey as arriving, neither supported by the excerpt. The donor's answer
on that row also adds unsupported relationships and details. These are
specific semantic concerns, not a measured population error rate.

**Decision:** retain the checkpoint as a candidate for stricter semantic
testing and high-fidelity export diagnostics; do not claim proven
instruction-quality conservation or native promotion yet. The next
quality check should freeze more independent, answerable generation
prompts or an explicit source-grounded adjudication rule before scoring.
The route remains exhaustive E128 and load imbalance observed in
METH-46 has not been resolved. A much larger *distinctly trained*
expert bank, bounded CPU router, actual factor storage and ≥50
accepted batch-1 tok/s on one quality-valid `engine.c` artifact are
still missing.
