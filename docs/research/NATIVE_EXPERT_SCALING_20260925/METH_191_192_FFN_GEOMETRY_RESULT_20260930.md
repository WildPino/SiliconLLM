# METH-191/192: constrained training rejected; direct low-rank weight repair exceeds the byte budget

METH-191's [protocol](METH_191_CONSTRAINED_FFN_CORRECTION_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth191_constrained_ffn_correction.py)
were committed at `44cade7` before execution. It kept A fixed,
trained only B for 64 updates, and projected each `BA` to no more
than 3% of its Q6 FFN base's Frobenius norm. Its fixed 16-sequence
validation split came only from the training corpus, disjoint from
the 64 optimization draws. The objective began at 0.016423 and was
0.071332, 0.056516 and 0.066376 at updates 16, 32 and 64. The best
trained checkpoint was 3.44× worse, failing the precommitted 10%
improvement rule. No checkpoint was exported, no METH-121 development
source was read, and no fresh quality or native rate test was run.
The [raw report](meth191_constrained_ffn_correction_result.json),
SHA-256 `78f1c5608614347bafe3288bf19427802b84cf57d252396aead9de08003e8e4f`,
contains every draw, validation row, and correction norm. Maximum
ratio at update 64 was only 1.272%, so the 3% projection was not the
active limiter. Training took 78.968 s and 6.968 GB peak GPU
allocation on the local RTX 3060.

METH-192's [frozen protocol](METH_192_Q6_RESIDUAL_RANK_PROTOCOL_20260930.md)
and [screen](../../../benchmarks/native_expert_scaling/meth192_q6_residual_rank_screen.py)
were committed at `6af9c9e` before execution. It computed the exact
896-dimensional Gram eigenspectrum of BF16 donor minus stored Q6
residuals for all 72 FFN matrices. Every Gram eigenvalue sum matched
the direct FP32 residual squared norm within the 0.1% gate. The
[raw result](meth192_q6_residual_rank_result.json), SHA-256
`23d2fe14aaa483262f9819729e6307500d63e17104c7fe71c62a2e2a62540d88`,
includes each matrix's optimal rank captures.

| Optimal correction rank | Added BF16 bytes/token | Total ideal bytes/token | Median Q6 residual energy captured |
|---:|---:|---:|---:|
| 64 | 53,084,160 | 534,619,136 | 15.10% |
| 94 | 77,967,360 | 559,502,336 | 20.85% |
| 128 | 106,168,320 | 587,703,296 | 27.00% |
| 256 | 212,336,640 | 693,871,616 | 47.00% |

Rank 94 is the highest uniform rank under 560 MB; rank 95 exceeds it.
The median optimal rank for 95% residual-energy recovery is **782**.
METH-192 fails its frozen ≥50% median capture gate at rank 94.
This rules against direct low-rank **weight reconstruction** as a
promising repair of these Q6 errors under the byte budget. It does
not bound model quality attainable by a task-trained residual or
another quantizer. Runtime was 17.656 s, peak GPU allocation
112.2 MB, and ending RSS 1.673 GB. No T4 or new text source was used.

A direct grouped-Q8 FFN is the next format control: exactly one
additional byte per four weights over packed Q6, or 78,446,592
bytes/token across 313,786,368 FFN weights. If its scales are kept
identical in count and type, the ideal core/router total is
559,981,568 bytes/token, only 18,432 below 560 MB. This leaves
almost no room for growing selected-router metadata as expert count
increases, and actual C memory traffic and quality remain unknown.
