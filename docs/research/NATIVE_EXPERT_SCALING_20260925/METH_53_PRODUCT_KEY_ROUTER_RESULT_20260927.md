# METH-53: exact product-key search passes, FP32 CPU scaling gate fails

The [frozen protocol](METH_53_PRODUCT_KEY_ROUTER_PROTOCOL_20260927.md)
tests a *new*, synthetic router geometry. Expert `(i,j)` receives score
`dot(U[i], x) + dot(V[j], x)`. Scanning A+B key vectors and evaluating
the 4×4 Cartesian product gives the exact top four pairs. This is
algorithmic exactness **within this factorized score function**, not
fidelity to the METH-47 learned dense router. The current METH-47
experts/keys were not converted or trained for this geometry.

| Grid | Key pool and addressed bytes/token | 1 thread | 6 threads |
|---:|---:|---:|---:|
| 8×16 = E128 | 2,064,384 bytes | 0.0976 ms | 0.0969 ms |
| 128×214 = E27,392 | 29,417,472 bytes | 1.4900 ms | 0.5671 ms |
| 512×534 = E273,408 | 89,972,736 bytes | 4.4803 ms | **2.7223 ms** |

All six runs pass scalar-versus-AVX2 dot checks (maximum absolute error
8.94e-7), repeat checksum checks and equality between one- and
six-thread route checksums. The exhaustive pair-score oracle agrees
with the 16-pair algorithm on 384 E128 cases and 24 cases at each
larger grid per thread arm. The E273,408 six-thread result splits into
2.5601 ms key dot scan and 0.1492 ms axis/pair selection plus checksum;
end RSS is 94,466,048 bytes. The 89.97 MB addressed key pool implies
33.05 GB/s addressed bytes at 2.7223 ms, not a DRAM-counter reading.

The fastest large-grid rate passes the predeclared **≤5 ms** component
bound. The E27,392→E273,408 latency ratio is **3.007× on one thread**
and **4.800× on six**, so the six-thread result fails the predeclared
≤4× scaling bound. The latter arm is reproducible across four timed
repetitions (2.651–2.741 ms/token). The outcome rejects this **FP32
full-dimension key layout** as the current large-E candidate under
the frozen rule. It does not reject product keys as a trainable
geometry; a smaller representation would be a separate experiment.

The [validated summary](meth53_product_key_summary.json) includes raw
arm file hashes, the arithmetic and the decision. C source SHA-256:
`56bff10df1260ea70ebdcf211e8745c830ca62534e429f149697aa2e4708f322`;
local binary SHA-256:
`805e3252df3d7a225cd5d16dfc39178df6d685112e7f01a8ee1810fb3e5d094a`.
Host: AMD Ryzen 5 3600X, six cores/12 logical threads, Windows,
clang 21.1.8. Compile and reproduce with:

```powershell
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks/native_expert_scaling/meth53_product_key_cpu.c -o results/native_expert_scaling/meth53_product_key_cpu.exe -lm -lpsapi
results/native_expert_scaling/meth53_product_key_cpu.exe 512 534 6 32
.venv/Scripts/python.exe benchmarks/native_expert_scaling/summarize_meth53_product_key.py --raw-dir docs/research/NATIVE_EXPERT_SCALING_20260925 --source benchmarks/native_expert_scaling/meth53_product_key_cpu.c --binary results/native_expert_scaling/meth53_product_key_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth53_product_key_summary.json
```

The keys and input vectors are deterministic synthetic data. No
distinct large-E experts, joint donor training, semantic evaluation,
`engine.c` LUT integration or end-to-end 50 accepted tok/s is tested.
METH-52's trained BF16 factor timing comes from a different artifact;
the two component times cannot be promoted to a same-artifact model
rate. The next bounded-cost diagnostic is a low-dimensional trainable
product-key geometry, followed by joint quality testing if it clears
its own CPU cost gate.
