# METH-53: exact product-key CPU router cost at RAM-scaled E

**Uncertainty.** METH-40's exhaustive rank-64 scan needs 18.453 ms/token
at synthetic E273,547 on this CPU, leaving no credible whole-model margin.
METH-29/30 post-hoc grouped indexes lost route fidelity. Test a different
router *geometry*: trainable additive product keys whose top-4 pair IDs can
be recovered exactly in O(A+B) score reads for E=A×B. This CPU experiment
tests algorithmic correctness and cost before any costly joint training.
It does not transfer the current METH-47 router or prove model quality.

For each of 24 layers define A fp32 key vectors and B fp32 key vectors
of dimension 896. A pair expert `(i,j)` has score `dot(U[i],x) +
dot(V[j],x)` and ID `i*B+j`. Score all A+B keys. Keep the best four
keys on each axis by score, breaking ties by lower index. Evaluate their
16 pairs and select the top four by score, breaking pair ties by lower
ID. Any pair outside this 4×4 product is dominated by at least four
pairs on one axis, so this is the exact top four of the factorized
router under the benchmark's fp32 arithmetic. Verify that claim against
an exhaustive pair-score oracle before timing.

Use a deterministic generated key pool and synthetic D896 inputs for
three grids: 8×16=E128, 128×214=E27,392 (~10B added capacity), and
512×534=E273,408 (~100B). The latter two are near the METH-26 E
targets, not trained expert banks. Each run reads key vectors only;
factor storage is priced separately by METH-51/52. Report exact
key-pool and addressed bytes/token, dot-scan and pair-selection times,
total median ms/token, checksums and RSS for one and six CPU threads.
Use a scalar-versus-AVX2 dot check and exact top-4 pair oracle at E128
on 24 layers × 16 synthetic inputs; at the larger grids check one
input/layer exhaustively. Same generated keys and inputs must be used
for both thread arms of each grid, with equal output checksums.

Warm eight tokens, then four timed repetitions: 256 tokens for E128,
64 for E27,392 and 32 for E273,408. Compile with clang
`-O3 -mavx2 -mfma -march=znver2 -fopenmp -lm -lpsapi` on the local
AMD Ryzen 5 3600X. Local CPU only, ≤8 GiB RSS and ≤10 minutes for
the complete sweep; no T4.

**Decision bounds:** if the fastest measured E273,408 router alone is
>5 ms/token or its E27,392→E273,408 same-thread latency ratio exceeds
4×, reject this fp32 product-key CPU representation before model
training. If both bounds pass and the exact oracle passes, retain it as
a CPU-cost candidate for joint donor-to-E128 training, then a distinct
larger-E quality ladder. The synthetic pair geometry has no semantic or
route-quality claim relative to METH-47, and cannot be added to METH-52
as a same-artifact end-to-end rate.
