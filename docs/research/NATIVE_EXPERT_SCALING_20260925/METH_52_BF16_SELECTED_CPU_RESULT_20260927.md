# METH-52: exact-effective BF16 selected path survives a 10× CPU pool

The [frozen protocol](METH_52_BF16_SELECTED_CPU_PROTOCOL_20260927.md)
binds the [METH-51](METH_51_BF16_EFFECTIVE_FACTOR_RESULT_20260927.md)
trained-factor artifact. The [seed exporter](../../../benchmarks/native_expert_scaling/meth52_export_bf16_seed.py)
verified its hash, metadata, shapes and BF16 dtype, then wrote an
88,080,408-byte interleaved E128 raw seed, SHA-256
`356dd993bf084d7cc7ace7d0400288c795b102d245a5cd101bd9d453beec0e32`.
The [C benchmark](../../../benchmarks/native_expert_scaling/meth52_bf16_selected_cpu.c)
uses the actual trained BF16 values. For E2,735 and E27,355 it repeats
E128 factor rows into larger resident pools; those rows are **not** new
learned experts. Inputs and four selected IDs per layer are synthetic.
The timed path includes all eight A dot products, SiLU, all 896 B row
dot products and four-way output accumulation at each of 24 layers.
It omits the router, donor core, tokenizer and generation.

| E | Factor pool | 1 thread median | 6 threads median |
|---:|---:|---:|---:|
| 128 | 88,080,384 bytes | 0.3984 ms/token | 0.3034 ms/token |
| 2,735 | 1,882,030,080 bytes | 0.3927 ms/token | 0.3282 ms/token |
| 27,355 | 18,823,741,440 bytes | 0.4064 ms/token | **0.3211 ms/token** |

All six arms passed the scalar-versus-AVX2 dot check (maximum absolute
error 1.49e-8) and one-versus-six thread repetition checksum equality.
At E27,355, the six-thread median components are 0.0138 ms/token
dispatch/input/checksum, 0.1692 A projection/SiLU and 0.1355 B/output.
The exact addressed factor payload is **2,752,512 bytes/token** for
top-4 at all E values. Dividing these bytes by the fastest median gives
8.57 GB/s of *addressed selected bytes*, not a hardware DRAM-counter
measurement. The largest arm's end RSS was 18,828,685,312 bytes.

The E2,735→E27,355 latency ratio for the same fastest six-thread path
is **0.978×**. Both predeclared component gates pass: 0.3211 ms is
below 3 ms, and the observed 10× pool ratio is below 1.5. The slight
decrease is measurement variation, not evidence that more experts
accelerate computation. Within the 20 ms whole-model target, this
component leaves 19.679 ms for everything else. It cannot be combined
with METH-40's different router seed as a same-artifact timing result.

The [validated summary](meth52_bf16_selected_cpu_summary.json), SHA-256
`63f5ff228fa802ca40d15e366924430c91162041e6d53eadae06f67cb4a7507d`,
links the six raw arm JSON files and recomputes bytes, medians and gates.
The C source SHA-256 is
`51297349ee066ee12e079eac64c10e40566c36421ec912ebb976f94869051ddc`;
the local binary SHA-256 is
`51a34fb12bbcac2d32a53b05731d52724edac2ce0cec364687b1f8bab2ea1a17`.
Host: AMD Ryzen 5 3600X, six physical cores/12 logical threads,
~80 GiB RAM, Windows; clang 21.1.8. Commands:

```powershell
.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth52_export_bf16_seed.py --out results/native_expert_scaling/meth52_e128_bf16_seed.bin --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth52_bf16_seed_export.json
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks/native_expert_scaling/meth52_bf16_selected_cpu.c -o results/native_expert_scaling/meth52_bf16_selected_cpu.exe -lm -lpsapi
results/native_expert_scaling/meth52_bf16_selected_cpu.exe --seed results/native_expert_scaling/meth52_e128_bf16_seed.bin --experts 27355 --threads 6 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth52_e27355_t6.json
.venv/Scripts/python.exe benchmarks/native_expert_scaling/summarize_meth52_bf16_selected.py --seed results/native_expert_scaling/meth52_e128_bf16_seed.bin --binary results/native_expert_scaling/meth52_bf16_selected_cpu.exe --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth52_bf16_selected_cpu_summary.json
```

The benchmark is a **selected BF16 matvec path, not the engine's ternary
LUT**. METH-31 still isolates the synthetic ternary LUT cost, while its
tested real lower-bit factor exports failed quality. E273,547 would
require 188,235,350,016 factor bytes at this precision and was not
allocated on this ~80 GiB host. Neither 10B nor 100B distinct expert
quality, a bounded router, same-artifact native logits or ≥50 accepted
tok/s is established here.

**Decision:** retain BF16 selected factors as a low-cost CPU component
candidate at the measured 10× RAM step. The limiting next work is
donor/core conversion and a bounded CPU router whose choices remain
quality-valid as *distinct* E grows, together with the METH-47
source-grounded semantic gate. Do not promote the component timing to
a whole-model rate.
