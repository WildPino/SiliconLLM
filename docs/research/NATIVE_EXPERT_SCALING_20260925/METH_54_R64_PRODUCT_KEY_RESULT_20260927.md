# METH-54: rank-64 product-key router passes the 10× CPU cost gate

The [frozen protocol](METH_54_R64_PRODUCT_KEY_PROTOCOL_20260927.md)
replaces METH-53's FP32 dimension-896 key vectors with a per-layer
`64×896` query projection and FP32 dimension-64 A/B keys. Each expert
pair `(i,j)` has score `dot(U[i], P x) + dot(V[j], P x)`. The same
16-pair search gives exact top four for this *new* factorized scoring
function; the keys and inputs are deterministic synthetic data.

| Grid | Projection + key bytes addressed/token | 1 thread | 6 threads |
|---:|---:|---:|---:|
| 8×16 = E128 | 5,652,480 bytes | 0.2075 ms | 0.2087 ms |
| 128×214 = E27,392 | 7,606,272 bytes | 0.2932 ms | 0.3164 ms |
| 512×534 = E273,408 | 11,931,648 bytes | 0.4948 ms | **0.4405 ms** |

At the largest grid the six-thread median contains 0.1979 ms query
projection, 0.0890 ms key scan and 0.1336 ms axis/pair selection plus
checksum. End RSS is 16,449,536 bytes. The key bank itself is
6,426,624 bytes and the shared projection is 5,505,024 bytes across
24 layers. Compared with METH-53's 89,972,736 addressed bytes and
2.7223 ms/token at the same large grid, this synthetic representation
reads much less and runs faster, though the two score functions and
route IDs are different.

All six arms pass scalar-versus-AVX2 dot checks (maximum absolute error
1.073e-6), repeat checksum checks, one-versus-six thread route checksum
equality and the exhaustive pair-score oracle. The oracle covers 384
cases at E128 and 24 at each larger grid per thread arm. The
E27,392→E273,408 six-thread latency ratio is **1.392×**. The fastest
large-grid median is **0.4405 ms/token**, below the frozen 3 ms bound;
the ratio is below the frozen 2× bound. **Decision: retain this
representation as a CPU-cost candidate for joint training.**

The [validated summary](meth54_r64_product_key_summary.json), SHA-256
`cd1fb0f1f80f1f273ebcd57f10fee97fa625c14188af05a4fba50aa44d863609`,
contains raw arm hashes and gate calculations. C source SHA-256:
`c73f955b95e9f1870092d873b374ac12450392ef72f72d52ff07f461f7d5791d`;
local binary SHA-256:
`735d7439085facc62e29b5c618c100c88125a64660d6efd04e36db6cba1db4d2`.
Host: AMD Ryzen 5 3600X, six cores/12 logical threads, Windows,
clang 21.1.8. Compile and reproduce with:

```powershell
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks/native_expert_scaling/meth54_r64_product_key_cpu.c -o results/native_expert_scaling/meth54_r64_product_key_cpu.exe -lm -lpsapi
results/native_expert_scaling/meth54_r64_product_key_cpu.exe 512 534 6 32
.venv/Scripts/python.exe benchmarks/native_expert_scaling/summarize_meth54_r64_product_key.py --raw-dir docs/research/NATIVE_EXPERT_SCALING_20260925 --source benchmarks/native_expert_scaling/meth54_r64_product_key_cpu.c --binary results/native_expert_scaling/meth54_r64_product_key_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth54_r64_product_key_summary.json
```

This does **not** establish router fidelity to the METH-47 checkpoint.
It has no jointly trained keys or distinct large-E experts, donor
quality test, `engine.c` LUT integration or same-artifact end-to-end
accepted token rate. The next experiment needs a product-key router
trained from donor parity with E128 distinct factors, explicit load
telemetry and source-grounded semantic evaluation before any larger-E
quality claim. METH-52's selected factor cost is a separate artifact
and cannot be added as a validated whole-model rate.
