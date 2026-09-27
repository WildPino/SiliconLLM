# METH-54: low-dimensional product-key CPU router cost

**Motive.** METH-53's exact additive-product geometry achieves
2.722 ms/token at E273,408, but its six-thread 10× scale ratio is
4.800× and fails the frozen 4× gate. Most of its work scans FP32
dimension-896 keys, 89.97 MB/token. Test a separately trainable
rank-64 representation: per-layer `q=P x`, with FP32
`P[64,896]`, `U[A,64]`, `V[B,64]`, pair score
`dot(U[i],q)+dot(V[j],q)`. A/B top-four cross selection remains
exact for this score function. No learned METH-47 routing fidelity
or model-quality claim follows from this cost experiment.

Use the same deterministic synthetic inputs, layers=24, grids
8×16=E128, 128×214=E27,392 and 512×534=E273,408, one and six
CPU threads, scalar-versus-AVX2 check and exhaustive pair-score
oracle: 24×16 input/layer cases at E128, one input per layer at
larger E. Same input and generated parameters across thread arms;
checksum equality required. Report exact projection/key storage,
addressed bytes/token, query projection, key dot scan, pair selection,
full median ms/token, RSS. Warm eight tokens; time four repetitions
of 256, 64 and 32 tokens by grid. Compile with clang 21.1.8
`-O3 -mavx2 -mfma -march=znver2 -fopenmp -lm -lpsapi` on the local
AMD Ryzen 5 3600X. Local CPU only, ≤1 GiB RSS and ≤6 minutes for
the whole sweep; no T4.

**Decision bounds:** require fastest E273,408 router ≤3 ms/token
and six-thread E27,392→E273,408 latency ratio ≤2×, plus all exact
oracle and checksum checks, to retain this representation as a
CPU-cost candidate for joint donor-to-E128 training. This is not
same-artifact `engine.c` timing or a distinct trained large-E ladder.
Any trained/quality test must be separately precommitted before it is
run; the semantic gate cannot be inferred from synthetic keys.
