# NES-02: 10× expert-count CPU cost stress

**Scope:** throughput-only apparatus, not a trained-capacity or quality test.
The user's target lets the number of experts grow with available RAM, roughly
10× from a 10B to a 100B variant. This cell tests whether the current native
CPU router/selection and LUT path have a large `E` cost that fixed top-8
cannot hide. It does not claim this L6/D256 model represents a 100B donor.

## Question, control and budget

The control is the real [NES-01](NES_01_E128_RESULT_20260925.md) E128 export,
with its trained router/experts. The stress artifact repeats each of its
experts ten times and independently randomizes router rows using the source
layer's weight/bias mean and standard deviation. Its header declares E1280;
top-8, D256, h128, L6, backbone and packed expert format are unchanged.
Copies are **not** distinct learned capacity and the random router cannot be
scored for language quality. Synthetic route IDs on 16 free-generated prefixes
reach 1,128–1,238 distinct experts/layer, confirming broad address coverage
in that separate workload; this is not a route-health claim.

The [synthesis tool](../../../benchmarks/native_expert_scaling/synthesize_e4_capacity.py)
refuses any source other than the NES-01 E128 E4 export SHA-256
`259e1aa73593e8fa8f72a997aaa63d30961342676fa0be587a22d920c804d2cf`.
Reproduction:

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\synthesize_e4_capacity.py --source results\native_expert_scaling\nes01_e128_e4.bin --replicas 10 --seed 0 --out results\native_expert_scaling\nes02_e1280_synthetic_e4.bin
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks\phase60\engine.c -o results\native_expert_scaling\engine_dynamic_e.exe -lm
```

The synthetic file is **3,812,869,184 B**, SHA-256
`25f99a300a26ed90948d04990ed75df9f94ab9a288ef48e6ea0c46fc0a3dcf2e`.
Only the 394 MB source and 3.813 GB synthetic file are needed locally; no GPU
or T4 work. The C binary reads `E` from the header, bounded to 8 through
`INT_MAX/HID_E` so row indexing stays representable. The synthesis tool is
limited to 65,535 because this probe also writes 16-bit route IDs. The dynamic
binary reproduced the previous E32 and E128 fp32 64-token logit streams
**bit for bit**. File load and one-output smoke succeeded at E1280. Windows
file-position checking now uses 64-bit `_ftelli64`/`_fseeki64`, and the writer
records full size/hash. A production large-pool format should omit redundant
fp32 expert reference weights; this stress file intentionally retains them.

The CPU is a Ryzen 5 3600X, six OpenMP threads, clang 21.1.8 with
`-O3 -mavx2 -mfma -march=znver2 -fopenmp`, byte ternary codes and fast SSM
exponential. Four separate 3,000-token input-path timings per arm use the
same validation slice; the first is excluded. The [raw summary](nes02_capacity_stress_baseline_20260925.json)
includes every repeat, log digest, synthetic artifact identity and route
coverage. Its selected code payload is **2,359,296 B/token in both arms** if
each selected weight is addressed once; it is not measured DRAM traffic.
The raw local logs are
`results/native_expert_scaling/nes02_dynamic_e128_timing_6threads.log` and
`nes02_dynamic_e1280_timing_6threads.log`. Each of four separate invocations
per arm uses the command below (substitute the corresponding weight file);
append output and stderr to that arm's log, then run the summary tool with
those logs, synthetic file and generated-route dump.

```powershell
& results/native_expert_scaling/engine_dynamic_e.exe --weights results/native_expert_scaling/nes02_e1280_synthetic_e4.bin --threads 6 --mlp lut --exp fast --timing --time-tok 3000
```

## Unoptimized baseline

| Metric, median of last three | E128 trained | E1280 synthetic | Ratio |
|---|---:|---:|---:|
| Packed ternary code pool | 36 MiB | 360 MiB | 10× |
| Total latency | 1,148.4 µs/token | 1,880.4 µs/token | 1.637× |
| Input-path rate | 870.8 tok/s | 531.8 tok/s | 0.611× |
| Router + selection | 53.6 µs/token | 536.1 µs/token | 10.002× |
| Selected-expert LUT path | 618.6 µs/token | 795.0 µs/token | 1.285× |

The router accounts for about 28.5% of E1280 total latency under this
workload. Its current implementation scores all E rows serially, applies
softmax over E and scans all E candidates eight times for top-8. The LUT path
also slows although top-k and selected code bytes stay fixed. A larger pool
can change cache/TLB behavior, but this timing alone does not identify the
memory level or isolate the repeated weights' influence. No actual 100B
model, distinct E1280 expert training, donor-relative quality or accepted
autoregressive 50 tok/s result is evidenced here.

## Bounded optimization, gates frozen before testing

**Uncertainty:** how much of the E1280 router growth is removable by scoring
independent router rows across six CPU cores, without changing any dot-product
reduction or top-k arithmetic? **Changed coordinate:** parallelize only the
row loop at E≥512 when `--threads` >1; leave E32/E128 serial and all selected
expert/LUT math, weights and route semantics unchanged. This uses the existing
OpenMP build. It costs one source edit/build and four repeats per E128/E1280,
under five minutes CPU; no training or T4 time.

Advance this candidate only if all of these hold:

1. E32/E128 64-token fp32 logit streams and E1280 synthetic 64-token stream
   remain **bit identical** to the unoptimized dynamic-E binary; generated
   route IDs may be checked if any logit differs.
2. E1280 median router+selection time is ≤0.5× its 536.1 µs baseline and
   median total latency ≤0.85× its 1,880.4 µs baseline. Report every repeat;
   a speedup in a single best run is insufficient.
3. E128 median total latency does not regress by more than 5% from 1,148.4
   µs/token. The router row loop should stay serial at E128.

If a gate fails, retain the failure and test a separate algorithmic router
change only after freezing its numerical/quality controls. The E128
free-generation failure in NES-01 still blocks promoting more trained experts,
regardless of a CPU speedup here.

## Parallel-router result

The implementation adds an opt-in `--router-parallel` flag. It applies OpenMP
only to independent router-row dot products at E≥512; the default remains the
serial path. The [result JSON](nes02_parallel_router_result_20260925.json)
contains the four raw repeats per arm, source log hashes, fp32 logit hashes,
and each frozen gate. E32, E128 and E1280 64-token fp32/exact logit streams
are bit identical between serial and parallel modes.
The parallel timing command is the same as above with `--router-parallel`;
its raw logs are `results/native_expert_scaling/nes02_flagged_parallel_e128_timing_6threads.log`
and `nes02_flagged_parallel_e1280_timing_6threads.log`. Final source code,
after the Windows 64-bit position fix, reproduced the E1280 fp32/exact
parallel and serial 64-token logit SHA-256
`b3002d3aca54e3ec0a416d681c732ab15c6a2f72e6b161641c5b37cedde049cc`.

| Metric, median of last three | E128 serial | E128 parallel | E1280 serial | E1280 parallel |
|---|---:|---:|---:|---:|
| Router + selection, µs/token | 53.6 | 56.8 | 536.1 | 355.9 |
| Total latency, µs/token | 1,148.4 | 1,124.9 | 1,880.4 | 1,654.7 |
| Input-path rate, tok/s | 870.8 | 889.0 | 531.8 | 604.3 |

E1280 router cost is **0.664×** baseline, missing the ≤0.5 gate. Total cost
is **0.880×** baseline, missing the ≤0.85 gate. E128 total is 0.980×
baseline, meeting its no-regression gate. Therefore **do not advance this
optimization as the large-E solution**. The flag remains available for a
reproducible experiment; it does not establish the target 50 accepted tok/s
on a 10B/100B artifact.

## Expert-count inversion at 10B and 100B

This is shape arithmetic for the *current* L6/D256/h128/top-8 geometry,
not a proposal that this tiny backbone would preserve 10B/100B quality.
E32 and E128 have 22,516,672 and 79,287,808 distinct trainable parameters,
respectively. Thus the count is `3,592,960 + 591,366 × E`; the increment
includes the dense router row/bias and three expert matrices at each layer.
The first E meeting 10B is 16,904; the first meeting 100B is 169,094. The
larger count is about 10×, as requested, and packed ternary expert pools would
be about **4.99 GB** and **49.87 GB** before metadata, router, core, allocator
overhead or redundant fp32 copies. Available RAM can limit stored capacity.

| Shape-derived term | ≈10B, E16,904 | ≈100B, E169,094 |
|---|---:|---:|
| Dense fp32 router weights addressed/token, `L×E×D×4` | 103,858,176 B | 1,038,913,536 B |
| At an optimistic sustained 40 GB/s, router payload alone | 2.596 ms | **25.973 ms** |
| Selected expert ternary codes at top-8 | 2,359,296 B | 2,359,296 B |

The 100B dense-router payload floor exceeds the entire **20 ms/token** budget
for 50 tok/s before candidate selection, selected experts, mixer, head and
other traffic. The 40 GB/s number is a favorable design yardstick, not a
measured router bandwidth; cache reuse or other hardware can change actual
cost. NES-02's E1280 timing directly shows router growth and a smaller LUT
slowdown despite fixed top-8. Together these results make exhaustive fp32
`O(L×E×D)` routing unsuitable as the assumed large-E path on this CPU.

The next router cell needs a bounded candidate search, hierarchy, or another
sublinear route with a frozen recall/quality gate against the exact E128
router, plus full latency and RAM accounting. A compact packed-only export
is also necessary to turn expert count into a practical RAM dial. Separately,
the E128 free-generation failure needs a quality remedy before training
larger distinct expert pools. Neither this synthetic stress nor the traffic
floor answers whether learned E≈100B can preserve donor-relative quality.
