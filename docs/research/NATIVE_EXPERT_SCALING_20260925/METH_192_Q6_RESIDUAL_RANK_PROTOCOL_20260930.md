# METH-192: exact rank budget for Q6 FFN weight residuals

METH-189/190 joint low-rank distillation failed viewed-source quality;
METH-191's constrained B-only retry failed its training-corpus-only
selection. Before further FFN low-rank training, measure what ranks
can recover from the **actual BF16 donor minus stored METH-186 Q6 FFN
weights**. This is a weight approximation screen, not a quality test.

Bind the exact Qwen2.5-0.5B-Instruct revision/source SHA and METH-186
stored core SHA. For every one of the 72 gate/up/down FFN matrices,
reconstruct Q6 BF16 weights exactly as METH-186 does, subtract from
the source BF16 weights in FP32, form the 896-dimensional Gram matrix,
and calculate all eigenvalues. Check that nonnegative eigenvalue sum
matches direct residual squared Frobenius norm within 0.1%. Report
the optimal rank-64, rank-94, rank-128 and rank-256 fractions of
quantization-residual squared energy captured, plus the minimum rank
to capture 95%. Report per organ, all-layer median, minimum, and
energy-weighted fraction. The rank-94 optimal projection is an upper
bound on any rank-94 linear correction's ability to reproduce the
original **weight matrix**, not an upper bound on model quality.

Under the established BF16 A/B storage rule, each added rank costs
829,440 bytes/token for all 72 maps. Rank 94 costs 77,967,360 bytes,
giving 559,502,336 bytes/token including the 481,534,976-byte
core/router ledger; rank 95 exceeds the 560 MB limit. If optimal
rank-94 captures less than 50% median residual energy, stop treating
direct low-rank weight reconstruction as a likely core fidelity path.
This gate does not reject function-trained adapters or alternative
quantization formats by itself.

Use only local weights, no new text sources. Cap at 15 minutes,
10.5 GiB GPU allocation, 20 GiB RSS, 1 GB result disk. No T4 is
authorized by this protocol.
