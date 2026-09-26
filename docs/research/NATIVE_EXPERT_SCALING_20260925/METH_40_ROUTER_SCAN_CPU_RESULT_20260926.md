# METH-40: stored router sketch CPU scan at the 10B/100B E ladder

**Decision:** the predeclared component ceiling passes narrowly at
E273,547 on this CPU: the fastest measured rank-64 sketch projection,
exhaustive scan, top-96 selection and merge takes **18.453 ms/token**
with six threads. This leaves **1.547 ms** inside the 20 ms/token
whole-model target for exact fine rescore, selected-expert LUT,
core inference and all other work. It is therefore a router-cost
candidate under the METH-40 rule, **not** a demonstrated ≥50 tok/s
native model. At this geometry a different, bounded routing design
is likely needed to leave practical room for the rest of the model.

The [frozen protocol](METH_40_ROUTER_SCAN_CPU_PROTOCOL_20260926.md)
binds the E128 METH-36 stored int8 sketch (SHA-256
`133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d`),
L24/D896/rank64/C96, the two E values, 1/6 threads and the
20 ms decision ceiling. The [exporter](../../../benchmarks/native_expert_scaling/meth40_export_router_seed.py)
verified the artifact and wrote a raw seed (5,713,944 bytes; SHA-256
`e0bca061fb9f42b17b22ddf10d17f4da37599e9995cbe698ee19d380aa489262`).
Its [export ledger](meth40_router_seed_export.json) is committed;
the raw binary is a local derived build artifact. The
[benchmark source](../../../benchmarks/native_expert_scaling/meth40_rank64_router_scan.c)
has SHA-256 `3b1ac5c8119239e44938c73110dd1dc56f96352ddf1d022f54a08a9c3e00d2ef`.
The compiled executable has SHA-256
`53710db22afb013355ee9b7befde83de65647f3a0afa9cbe1ea09c6cbb1c5515`.

Windows, AMD Ryzen 5 3600X, 80 GiB physical RAM. Compile command:

```powershell
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks/native_expert_scaling/meth40_rank64_router_scan.c -o results/native_expert_scaling/meth40_rank64_router_scan.exe -lm
```

For each of the four arms, run the executable with
`--seed results/native_expert_scaling/meth40_e128_router_seed.bin`,
`--experts 27355` or `273547`, `--threads 1` or `6`, and
`--out docs/research/NATIVE_EXPERT_SCALING_20260925/meth40_e<E>_t<T>.json`.
The large-E run uses 8 warmup and 4×16 measured tokens; the
small-E run uses 8 warmup and 4×64. Values below are medians of
four repetitions, in ms/token. The independent
[summary verifier](../../../benchmarks/native_expert_scaling/summarize_meth40_router_scan.py)
checks hashes, byte counts, medians, components and cross-thread
checksums; its [machine summary](meth40_router_scan_summary.json)
is the concise audit entry.

| Experts per layer | Threads | Projection | Scan + selection | Merge | Total | Sketch pool |
|---:|---:|---:|---:|---:|---:|---:|
| 27,355 | 1 | 0.214 | 6.217 | 0.168 | 6.598 | 50.148 MB |
| 27,355 | 6 | 0.244 | 3.034 | 0.432 | **3.714** | 50.148 MB |
| 273,547 | 1 | 0.217 | 53.420 | 0.144 | 53.780 | 451.934 MB |
| 273,547 | 6 | 0.249 | 17.864 | 0.334 | **18.453** | 451.934 MB |

The 10× E increase raises the fastest measured router component
by **4.968×**. At E273,547, the scan addresses 420,168,192 code
bytes plus 26,260,512 scale bytes per token; projection addresses
5,505,024 basis bytes. The code-byte-only time at a favorable
40 GB/s is 10.504 ms. This is arithmetic, not a measured DRAM
bandwidth bound. The scalar/AVX2 self-test's maximum absolute
score difference is 0.0001831, and one/six-thread result checksums
agree at each E. The 20 ms component gate passes; no end-to-end
rate or quality gate is claimed.

The expanded rows repeat real E128 sketch rows with deterministic
scale jitter. They are **not independently trained experts**.
Neither exact candidate fine rescoring nor route-replaced quality
at larger E was tested. The current E128 C64 route fails its
prospective top-1 quality gate (METH-37), and C96 has only a
reused-prompt diagnostic (METH-38/39). This CPU result cannot
promote either route or imply that larger choice preserves quality.
It instead narrows the design: retain C96 only as a quality
experiment, and test a learned bounded/sublinear lookup or a
smaller scan payload against independent large-E route targets
before integrating it into `engine.c`.

Raw evidence: [E27,355/t1](meth40_e27355_t1.json),
[E27,355/t6](meth40_e27355_t6.json),
[E273,547/t1](meth40_e273547_t1.json), and
[E273,547/t6](meth40_e273547_t6.json). Their SHA-256 hashes are
recorded in the machine summary. No T4 time was used.
