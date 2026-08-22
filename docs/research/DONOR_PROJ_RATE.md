# Donor-Scale Projection-GEMV Rate — measurement report

**Role:** Builder. **Branch** `research/donor-adaptation`. **Date** 2026-08-20. **No commit, no push.**
**Owns:** `benchmarks/donor_adaptation/gemv_donor_bench.c`, this file. `benchmarks/phase60/engine.c` and
`benchmarks/phase60/gemv_bench.c` were **read and copied from, never modified** (verify with `git status`).

---

## 0. Why this measurement exists

`docs/research/CONTROLLER_STAGE1_AUDIT.md` F1: the donor stage-1 verdict rests on one unmeasured constant,
the fully-streamed fp32 projection-GEMV rate. `docs/PHASE64_BUDGET.md` §1b measured `r(size)` only to
**96 MB** and then *asserted* an asymptote of "34-36 GB/s"; `donor_inventory.py` hardcoded the bottom of that
(`PROJ_STREAMED_FLOOR = 34.0`) and returned "zero donors pass the ≥10 tok/s gate". At 36 the best donor
passes (10.01 tok/s), at 40 it reaches 10.65. Every donor row is an extrapolation past the last measured
point: `Qwen2.5-1.5B` streams **1.550 GB** of fp32 projections+head per token, 16× beyond 96 MB.

**Headline answers.**
1. **The fp32 asymptote is 38.84 ± 0.68 GB/s** — measured, not extrapolated, flat from 64 MB to 1024 MB
   (§4). **Not 34. Not 36.** It does not keep declining, and there is no further cliff past 96 MB.
2. **The donor's actual per-token projection stream measures 37.74 ± 0.18 GB/s = 41.07 ms/token** (§5.2),
   with byte accounting exactly matching the F1 hand-derivation. On this constant alone, `Qwen2.5-1.5B`
   clears the ≥ 10 tok/s gate at **10.22 tok/s** (§5.3). `PROJ_STREAMED_FLOOR = 34.0` is **11% low** and is
   the sole source of the FAIL verdict.
3. **Organ shape stops mattering once streamed** (§5.1): skinny 256×1536 K/V, square Q/O, the 52.5 MB MLP
   slabs and the 890 MB head all land in 36.0-39.0 GB/s. The skinny-matrix worry is falsified.
4. **The engine-integrated ternary MLP at donor dims costs 27.31 ± 0.46 ms/token = 21.25 GB/s** (§10).
   This closes the last wide bracket in the stage-1 model. Integration overhead at donor `D` is
   **+30.5%**, not the 2.4× the [11.40 .. 27.64] bracket assumed — and the tool's 11.398 constant turns
   out to describe the **dReLU row-skip** path, which a SiLU-gated donor cannot use.
5. **The ternary LUT path is BANDWIDTH-bound at donor dimensions, not compute-bound** (§8) — Controller #2's
   "compute-bound by ~16×" does not transfer from `D=256` to `D=1536`. The kernel's own ceiling is 3.3×
   higher at donor `D` than at sandbox `D`.

**Read §2 first — the pre-stated known-positive gate did NOT pass, and why it did not is itself a finding
about `PHASE64_BUDGET.md` §1b rather than about this harness.**

---

## 1. The harness

`benchmarks/donor_adaptation/gemv_donor_bench.c`. Single C file, no weights, no dataset, CPU-only, no GPU.

**Copied verbatim from `benchmarks/phase60/engine.c`** (the kernels under test must be the engine's, not
lookalikes): `now_s`, `hsum256`, `dotf`, `matvec` (the `OMP_PFOR ... schedule(static) if(g_omp_on)` row
partition), `xmalloc`, `acc_add_i8x32`, `matvec_lut_full`, `build_lut_t3`, `bc_tm`, `ref_t3`.
**Methodology copied from `engine.c: run_gemv_sweep()`** — the routine that actually produced the §1b curve,
driven by `benchmarks/phase64/bench_64_1b.sh`: `npass = max(16, 2 GiB / bytes)` per timing window,
min-of-N estimator, one warm call, `x` poisoned each pass so the matvec cannot be hoisted, and the same
synthetic fills (`W[i] = 0.001·((i mod 13)−6)`, `x[i] = 0.01·((i mod 7)−3)`).
**Reporting shape copied from `gemv_bench.c`:** GFLOP/s plus max rel-err against a float64 reference on real
projection dims. `run_expert_rate()` supplies the i.i.d. expert-pool methodology for the ternary path.

The honest `matvec` is called **directly, not through a function pointer**, so the compiler inlines the timing
loop exactly as `engine.c` does. (This was a real defect in the first build — routing through the pointer
blocked inlining and moved the 32 MB cliff point by 30%. Fixed before any number below was taken.)

Modes: `repro` `sweep` `organs` `fullstack` `lut` `lutrepro` `lutstack` `correct` `control`.

### Build and invocation (exact)

```
# 64.1b build policy — the flags that produced the curve being reproduced. NO -ffast-math, ever.
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp \
      benchmarks/donor_adaptation/gemv_donor_bench.c -o bin/gemv_donor_bench.exe -lm

export OMP_PROC_BIND=close OMP_PLACES=cores          # same pinning as bench_64_1b.sh

bin/gemv_donor_bench.exe control                     # planted controls    (§7)
bin/gemv_donor_bench.exe correct                     # float64 table       (§6)
bin/gemv_donor_bench.exe repro     --reps 5          # known-positive gate (§2)
bin/gemv_donor_bench.exe sweep     --reps 5          # 4..1024 MB curve    (§4)
bin/gemv_donor_bench.exe organs    --reps 5          # real donor shapes   (§5)
bin/gemv_donor_bench.exe fullstack --reps 5          # 1.550 GB/token      (§5)
bin/gemv_donor_bench.exe lutrepro                    # known-positive #2   (§8)
bin/gemv_donor_bench.exe lut       --reps 3          # ternary, donor dims (§8)
bin/gemv_donor_bench.exe lutstack  --reps 5          # 578 MB/token        (§8)

# the side-by-side instrument-identity check (§2) — the original routine, unmodified:
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp \
      benchmarks/phase60/engine.c -o bin/engine_64_1b_check.exe -lm
bin/engine_64_1b_check.exe --gemv-sweep
```

**Machine:** AMD Ryzen 5 3600X, 6 cores / 12 threads, 3793 MHz, 80 GB RAM, Windows 11.
clang 21.1.8 (x86_64-w64-windows-gnu). OpenMP present, `omp_get_max_threads()=12`; `t6` = an explicit
`omp_set_num_threads(6)`, exactly as `engine.c` does. Machine state at time of the timing runs: CPU load 10%,
55.5 GB free of 79.9 GB, `webwallpaper32` terminated; Spotify / Chrome / FanControl / a ChatGPT client
resident but idle. Every timed row below carries its own `cv%`; see §3.

---

## 2. THE KNOWN-POSITIVE GATE — read this before any number

### 2.1 Gate A (pre-stated): reproduce §1b to ±8%. **FAIL, 5/8 points.**

`bin/gemv_donor_bench.exe repro --reps 5`:

| size MB | t1 GB/s | (pub t1) | **t6 GB/s** | **(pub t6)** | t6 dev | cv% | verdict |
|---|---|---|---|---|---|---|---|
| 4 | 34.9 | 30.5 | 175.7 | 187.0 | −6.0% | 3.1 | PASS |
| 8 | 34.3 | 31.3 | 196.2 | 185.0 | +6.0% | 1.7 | PASS |
| 16 | 29.2 | 27.2 | 199.5 | 134.4 | **+48.4%** | 4.5 | OUT OF BAND |
| 24 | 26.6 | 27.8 | 188.5 | 60.5 | **+211.5%** | 1.4 | OUT OF BAND |
| 32 | 26.8 | 24.9 | 148.3 | 55.7 | **+166.3%** | 1.6 | OUT OF BAND |
| 48 | 26.2 | 24.5 | 40.0 | 45.5 | −12.1% | 1.6 | OUT OF BAND |
| 64 | 26.0 | 24.9 | 38.4 | 45.3 | −15.2% | 2.4 | OUT OF BAND |
| 96 | 26.4 | 23.3 | **38.3** | **36.5** | **+4.9%** | 3.8 | **PASS** |

The **t1 row reproduces at every one of the 8 points** (worst deviation +13%, most within 5%). The t6 row
reproduces at the two smallest sizes and at **96 MB — the last published point and the sole anchor of the
"34-36 GB/s asymptote" claim**. It does not reproduce in the 16-32 MB cache-transition region, and runs
12-15% low at 48-64 MB.

**Per the brief this is a FAIL and I am reporting it as one. I did not widen the band to make it pass.**

### 2.2 What the failure actually is — the original routine does not reproduce it either

The correct identity question is not "does my harness match a table written on 2026-07-12" but "is my
harness the same instrument as `engine.c`". Tested directly: `engine.c --gemv-sweep`, **unmodified**, rebuilt
with the same flags and run interleaved with the harness on the same machine, same session, same env.

t6 GB/s, three interleaved pairs (`engine.c` / harness):

| size MB | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|
| `engine.c` today (mean of 3) | 188.2 | 188.3 | 190.7 | 197.0 | 120.3 | 40.1 | 37.9 | 38.7 |
| harness today (mean of 3) | 188.5 | 193.1 | 197.1 | 184.2 | 158.9 | 41.1 | 38.1 | 38.3 |
| agreement | 0.2% | 2.5% | 3.4% | 6.5% | **32%** | 2.5% | 0.5% | 1.0% |
| *published 2026-07-12* | *187* | *185* | *134* | *60.5* | *55.7* | *45.5* | *45.3* | *36.5* |

**The two instruments agree with each other at 7 of 8 points, and both disagree with the published table in
exactly the same places.** The one point where they differ from each other, 32 MB, is the L3 cliff, and it is
unstable in both (see §3). So the harness is the instrument; the published mid-region is not reproducible on
this machine today **by the code that produced it**.

### 2.3 Why: the transition region is thread-placement-dependent, the streamed tail is not

`engine.c --gemv-sweep`, t6 GB/s, only `OMP_PLACES` varied:

| OMP_PLACES | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|
| `cores` (bench_64_1b default) | 183.0 | 195.0 | 197.4 | 192.5 | 149.7 | 40.4 | 39.8 | 38.1 |
| `threads` | 153.0 | 158.1 | 127.6 | 43.5 | 38.5 | 38.7 | 37.0 | 38.1 |
| `sockets` | 173.7 | 158.6 | 163.2 | 80.9 | 76.5 | 47.8 | 40.3 | 37.2 |
| unpinned | 158.3 | 159.1 | 160.4 | 120.2 | 91.3 | 46.9 | 39.4 | 37.3 |
| **spread** | 1.2× | 1.2× | **1.5×** | **4.6×** | **3.9×** | 1.2× | 1.1× | **1.02×** |
| *published* | *187* | *185* | *134* | *60.5* | *55.7* | *45.5* | *45.3* | *36.5* |

The 16-32 MB region moves by up to **4.6× on thread placement alone**: 6 threads on 6 physical cores see the
aggregate 2×16 MB L3 and stay resident to 32 MB; 6 threads on 3 physical cores (SMT siblings, one CCX) see
16 MB and break at 24 MB — which is the shape of the published curve. **The published run's placement is not
recorded**, so the mid-region cannot be reproduced by anyone, including `engine.c` itself.

**At 96 MB the spread across all four placements is 37.2-38.1 GB/s — 2%.** The streamed tail is
placement-insensitive, and that is the entire region every donor organ occupies.

### 2.4 What this licenses, stated plainly

- The harness **is** the §1b instrument (§2.2). Same kernels, same estimator, same numbers side by side.
- The published **t1 curve** reproduces fully.
- The published **t6 cache-transition region (16-32 MB) is not reproducible** and should be treated as
  placement-conditioned, not as a property of the machine. **`PHASE64_BUDGET.md` §1b should carry this
  caveat.** No donor conclusion depends on it.
- The published **96 MB anchor reproduces (+4.9%)**, and it is placement-insensitive.
- Therefore the extension past 96 MB in §4 is licensed **as a measurement**, not as an extrapolation — every
  point from 128 MB to 1024 MB is directly measured, four times, and §5 measures the donor's real stream
  end-to-end rather than reading any curve at all.
- **The 37.74 GB/s figure in §5 uses the same harness path as this check** (same `matvec`, same direct-call
  timing loop, same estimator) and does not depend on the §1b table being right.

---

## 3. Variance — what is reportable

Reported per row as `cv%` = coefficient of variation of the per-repetition rates inside one invocation, plus
between-run sd across independent invocations. **Rule applied: >5% is not reportable as a value.**

| region | between-run sd | worst in-run cv | reportable? |
|---|---|---|---|
| 4-24 MB (L3-resident) | 1.1-3.2% | 5.3% | yes, but placement-conditioned (§2.3) |
| **32 MB (the L3 cliff)** | **22.2%** | **21.5%** | **NO — DO NOT QUOTE. Bimodal: 118-170 GB/s.** |
| 48 MB | 1.5% | 4.6% | yes |
| **64-1024 MB (streamed)** | **0.4-2.5%** | 10.9% (one 64 MB rep) | **yes** |
| fullstack 1.550 GB (§5) | **0.5%** | 4.4% | yes |
| organs, streamed | 1-5% | 5.0% | yes |

The machine was quiet enough: 33 of 36 tail measurements sit inside ±1.5% of the mean. **The single
non-reportable number is the 32 MB point**, and it is non-reportable for a physical reason (it is exactly the
aggregate L3 size, so it flips between resident and spilled), not because of background load.

---

## 4. The curve past 96 MB — the asymptote

`bin/gemv_donor_bench.exe sweep --reps 5`, **4 independent invocations**, `OMP_PLACES=cores`. Rate charged
as `M·K·4 B` per pass; in=512 so `M = bytes/2048`, exactly as §1b.

| size MB | t1 GB/s | sd | **t6 GB/s** | **sd** | sd% | t6 GFLOP/s | t6/t1 |
|---|---|---|---|---|---|---|---|
| 4 | 34.9 | 0.29 | 184.8 | 5.97 | 3.2% | 92.4 | 5.30× |
| 8 | 34.7 | 0.25 | 193.3 | 4.42 | 2.3% | 96.7 | 5.57× |
| 16 | 29.7 | 0.45 | 199.2 | 2.13 | 1.1% | 99.6 | 6.71× |
| 24 | 26.9 | 0.31 | 192.2 | 3.80 | 2.0% | 96.1 | 7.14× |
| 32 | 27.0 | 0.14 | *140.6* | *31.17* | *22.2%* | *70.3* | *5.21×* |
| 48 | 26.3 | 0.27 | 41.1 | 0.60 | 1.5% | 20.6 | 1.56× |
| 64 | 26.2 | 0.10 | 38.2 | 0.95 | 2.5% | 19.1 | 1.46× |
| 96 | 26.5 | 0.17 | 39.0 | 0.53 | 1.4% | 19.5 | 1.47× |
| **128** | 26.0 | 0.24 | **39.7** | 0.17 | 0.4% | 19.9 | 1.53× |
| **192** | 26.1 | 0.12 | **39.3** | 0.35 | 0.9% | 19.7 | 1.51× |
| **256** | 26.4 | 0.17 | **38.2** | 0.22 | 0.6% | 19.1 | 1.45× |
| **384** | 26.5 | 0.15 | **38.9** | 0.33 | 0.8% | 19.5 | 1.47× |
| **512** | 26.2 | 0.13 | **39.4** | 0.14 | 0.4% | 19.7 | 1.50× |
| **768** | 25.9 | 0.06 | **38.4** | 0.55 | 1.4% | 19.2 | 1.48× |
| **1024** | 25.8 | 0.17 | **38.3** | 0.16 | 0.4% | 19.2 | 1.48× |

*(32 MB row italicised: not reportable, §3.)*

### The answer

> **Asymptote, t6 = 38.84 ± 0.68 GB/s** (mean ± sd over **36 measurements**: 9 sizes ≥64 MB × 4 independent
> runs; min 36.8, max 39.9, sd 1.7%). **t1 = 26.17 ± 0.28 GB/s.**

**It is not 34. It is not 36. It is ~38.8, and it does not keep declining.** The curve reaches its asymptote
by 64 MB and is flat across the next **16× of size**, to 1024 MB — no downward trend (linear fit over
128-1024 MB is within noise of zero slope). This also disposes of the worry behind the whole exercise:
there is no further cliff waiting past the last measured point.

Context: `PHASE64_BUDGET.md` §1 gives the DRAM aggregate ceiling as **40-44 GB/s**, and §1's engine-integrated
proj-GEMV datum as **~40 GB/s**. 38.8 sits just under the aggregate ceiling — i.e. **the fp32 projection
stream runs at essentially full memory bandwidth at every donor-relevant size**, which is what §1 already
said and what the 34.0 constant contradicted.

---

## 5. Real donor shapes — `Qwen/Qwen2.5-1.5B`

Shapes read from `benchmarks/donor_adaptation/configs/Qwen__Qwen2.5-1.5B.json` (the artefact is the
authority): `D=1536, L=28, V=151936, ffn=8960, n_head=12, n_kv=2, head_dim=128, tie_word_embeddings=true`.

### 5.1 Per organ

`bin/gemv_donor_bench.exe organs --reps 5`, 2 invocations. RESIDENT = one copy hot-looped (shape effect /
upper bound); STREAMED = replicated past 256 MB and cycled so every touch is cold (what a decode sees).

| organ | M×K | MB | RES t1/t6 GB/s | **STR t1/t6 GB/s** | t6 GFLOP/s | cv% |
|---|---|---|---|---|---|---|
| q_proj | 1536×1536 | 9.0 | 27.9 / 161.0 | 25.0 / **37.9** | 80.5 | 1.3 |
| k_proj | 256×1536 | 1.5 | 27.9 / 143.3 | 24.6 / **37.9** | 71.6 | 1.3 |
| v_proj | 256×1536 | 1.5 | 28.2 / 144.7 | 24.3 / **38.2** | 72.3 | 1.7 |
| o_proj | 1536×1536 | 9.0 | 28.3 / 160.6 | 24.2 / **38.1** | 80.3 | 4.8 |
| gate_proj | 8960×1536 | 52.5 | 24.8 / 42.1 | 24.2 / **38.1** | 21.0 | 4.9 |
| up_proj | 8960×1536 | 52.5 | 24.2 / 38.8 | 23.8 / **36.0** | 19.4 | 3.7 |
| down_proj | 1536×8960 | 52.5 | 22.7 / 40.9 | 23.3 / **39.0** | 20.4 | 3.0 |
| **lm_head** | **151936×1536** | **890.2** | 24.3 / 38.3 | — / **38.3** | 19.2 | 2.7 |

**Findings.**
1. **Shape does not matter once the organ is streamed.** Every organ — the 1.5 MB skinny K/V, the 9 MB
   square Q/O, the 52.5 MB MLP slabs, and the 890 MB head — lands in **36.0-39.0 GB/s**, a 8% spread,
   all inside the §4 asymptote's band. The worry that skinny (256×1536) or enormous-and-skinny
   (151936×1536) matrices would break the square-matrix curve is **falsified by measurement**.
2. **The head is not special.** At 890 MB — 9× past any previously measured point — it runs at 38.3 GB/s,
   the same as everything else. It is the largest single organ and it obeys the same law.
3. **Resident rates are irrelevant to the donor.** K/V and Q/O look like 143-161 GB/s in isolation, but a
   decode token walks 1.55 GB, so nothing survives in L3 between layers. **The STREAMED column is the
   donor-relevant one.** Anyone pricing a donor off the resident column would overstate by 4×.

### 5.2 The whole per-token stream — the number the verdict needs

`bin/gemv_donor_bench.exe fullstack --reps 5`, 3 invocations. One decode token's fp32 organs allocated
contiguously and walked **in layer order**: 28 × [q, k, v, o] then the head. This is not a curve lookup and
not an extrapolation — it is the donor's actual per-token projection traffic, timed.

**Byte accounting (planted control, §7 C5): 1,550,057,472 B — EXACT MATCH to the
`CONTROLLER_STAGE1_AUDIT.md` F1 hand-derivation.** The harness charges precisely what the model charges.

| threads | GB/s | ms/token | cv% |
|---|---|---|---|
| t1 | 25.30 ± 0.17 | 61.27 ± 0.40 | 0.7-3.0 |
| **t6** | **37.74 ± 0.18** | **41.07 ± 0.19** | 1.1-4.4 |

> **The proj-GEMV term for `Qwen/Qwen2.5-1.5B` is 41.07 ms/token, i.e. an effective 37.74 GB/s.**

It sits ~3% below the §4 synthetic asymptote (38.8) because the real layer order interleaves the two skinny
K/V organs between the square ones — a small, measured, real cost, not a modelling assumption.

### 5.3 What this does to the stage-1 arithmetic

Substituting **only** this measured constant into the F1 decomposition (everything else untouched: MLP at
the tool's 11.398 GB/s, KV at 40, glue 32.7 µs):

| proj rate | proj ms | MLP ms | KV ms | glue | total ms | **tok/s** | gate ≥10 |
|---|---|---|---|---|---|---|---|
| 34.0 (the tool's hardcoded floor) | 45.59 | 50.90 | 5.87 | 0.03 | 102.40 | 9.77 | FAIL |
| 36.0 (top of §1b's stated asymptote) | 43.06 | 50.90 | 5.87 | 0.03 | 99.86 | 10.01 | PASS |
| **37.74 (MEASURED, this document)** | **41.07** | 50.90 | 5.87 | 0.03 | **97.87** | **10.22** | **PASS** |
| 40.0 (budget §1/§2 top) | 38.75 | 50.90 | 5.87 | 0.03 | 95.55 | 10.47 | PASS |

**`PROJ_STREAMED_FLOOR = 34.0` is 11% below the measured rate and is the sole source of the FAIL verdict.**
On the measured constant the best donor clears the gate at **10.22 tok/s**. This settles the bracket
question F1 raised; it does **not** settle the donor route — F7 (the sealed 128K contract) and F8 (4-bit KV
is unbuilt) are untouched by this measurement and remain outstanding.

---

## 6. Correctness vs float64 — every shape

`bin/gemv_donor_bench.exe correct`. Two metrics, because one of them is a trap:

- `rel = max |y−yr| / (|yr|+1e-30)` — **blows up under cancellation and is reported, not used as the gate.**
  On the §1b synthetic fill (`W` cyclic mod 13, `x` cyclic mod 7) many row sums land near zero, so `rel`
  reaches 1.15e+01 while the kernel is bit-for-bit as accurate as fp32 permits. Quoting `rel` here would be
  the wrong number.
- `norm = max |y−yr| / Σ_k|W·x|` — condition-normalised. **This is the gate**, bounded by `sqrt(K)·eps_fp32`.

**Row coverage is 100%** — every shape came in under the checker's cost budget, so stride = 1 everywhere;
no sampling anywhere in the table (including the 151936-row head and the 524288-row 1024 MB point).

| shape | M×K | size | rows | t1 rel | t1 **norm** | t6 rel | t6 **norm** | bound | verdict |
|---|---|---|---|---|---|---|---|---|---|
| q_proj | 1536×1536 | 9.0 MB | 1536/1536 | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| k_proj | 256×1536 | 1.5 MB | 256/256 | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| v_proj | 256×1536 | 1.5 MB | 256/256 | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| o_proj | 1536×1536 | 9.0 MB | 1536/1536 | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| gate_proj | 8960×1536 | 52.5 MB | 8960/8960 | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| up_proj | 8960×1536 | 52.5 MB | 8960/8960 | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| down_proj | 1536×8960 | 52.5 MB | 1536/1536 | 1.53e−05 | **4.31e−09** | 1.53e−05 | 4.31e−09 | 1.13e−05 | PASS |
| **lm_head** | **151936×1536** | **890.2 MB** | **151936/151936** | 1.15e+01 | **4.53e−09** | 1.15e+01 | 4.53e−09 | 4.67e−06 | PASS |
| sweep 4…1024 MB (15 shapes) | M×512 | 4-1024 MB | all rows | 5.40e−07 | **9.30e−09** | 5.40e−07 | 9.30e−09 | 2.70e−06 | PASS ×15 |

Worst normalised error anywhere: **9.30e−09, i.e. 290× inside the `sqrt(K)·eps` bound.**
**t1 and t6 agree to the last bit on every shape** — the row partition changes which thread computes a row,
never the row's value, so threading cannot be hiding an error.

### Ternary LUT path — bit-exact, against **two independent** references

Reference A = scalar sum over the packed codes (the kernel's definition). Reference B = `ref_t3` over the
*original* ternary weights, which also validates the `bc_tm` packing.

| M×K | codes MB | mismatch vs A | mismatch vs B | rows | verdict |
|---|---|---|---|---|---|
| 384×256 | 0.05 | 0 | 0 | 384 | BIT-EXACT |
| 1536×1536 | 1.12 | 0 | 0 | 1536 | BIT-EXACT |
| 8960×1536 | 6.56 | 0 | 0 | 8960 | BIT-EXACT |
| 1536×8960 | 6.56 | 0 | 0 | 1536 | BIT-EXACT |
| 2048×512 | 0.50 | 0 | 0 | 2048 | BIT-EXACT |

---

## 7. Planted controls — real output, both directions

`bin/gemv_donor_bench.exe control`. Verbatim, trimmed only of blank lines.

```
C0  BASELINE (honest kernel, q_proj 1536x1536, 1536 rows checked)
    max rel-err  = 1.150e+01   (worst row 5)
    max norm-err = 4.526e-09   (worst row 10)   sqrt(K)*eps_fp32 = 4.672e-06
    FIRE THRESHOLD for every control below := 10 x baseline norm-err = 4.526e-08

C1  DOES THE INSTRUMENT SEE ONE ELEMENT AT ALL?  (target W[768][768] = -0.00300000003, x[768] = 0.0199999996)
    C1a NEGATIVE leg - nothing corrupted, float64 reference recomputed:
        max |ref64(pristine) - ref64(pristine)| over all 1536 rows = 0  -> exactly 0: reference deterministic (silent, as it must be)
    C1b POSITIVE leg - W[768][768] moved by exactly ONE ULP (dw = 2.3283064365386963e-10):
        row 768 observed delta = 4.656612768993984e-12
               predicted dw*x = 4.656612768993984e-12
               agreement 0.000e+00 relative -> FIRES, at exactly the predicted magnitude
        all other 1535 rows: 0 changed, max delta 0 -> silent everywhere else (no cross-talk)
    C1c the fp32 KERNEL under the same 1-ULP corruption: 0/1536 output floats changed; norm-err 4.526e-09 vs threshold 4.526e-08
        -> fp32 comparator is BLIND to a 1-ULP single-element corruption. This is PHYSICS, not a defect:
           the induced delta (dw*x ~ 1e-11) sits orders below the fp32 accumulation noise of a K=1536 dot.
           The float64 reference (C1b) DOES see it, which is what proves every element of W is consumed.

C2  MINIMAL-SIGNIFICANT-CORRUPTION LADDER (fp32 kernel vs pristine float64 reference)
    W[768][768] *= (1+delta), everything else untouched. Fire threshold = 4.526e-08
    delta        rel-err     norm-err    verdict
    0e+00        1.150e+01   4.526e-09   silent  (negative leg: delta=0 must NOT fire)
    1e-07        1.150e+01   4.526e-09   silent
    1e-06        1.150e+01   4.526e-09   silent
    1e-05        1.150e+01   6.047e-09   silent
    1e-04        1.150e+01   6.894e-08   FIRES
    1e-03        1.150e+01   7.033e-07   FIRES
    1e-02        1.150e+01   7.045e-06   FIRES
    1e+00        1.150e+01   7.046e-04   FIRES
    1e+06        1.714e+05   7.046e+02   FIRES
    -> minimal corruption of ONE element out of 2359296 that this comparator resolves: delta = 1e-04

C3  THE PLAUSIBLE ARTEFACT: a kernel that streams fewer bytes than the harness charges.
    kernel        bytes read   charged   inflation  rel-err     norm-err    caught?
    honest           9.0 MB      9.0 MB   1.00x      1.150e+01   4.526e-09   silent (correct)
    halfrows         4.5 MB      9.0 MB   2.00x      1.150e+01   7.059e-03   CAUGHT   (odd rows never computed)
    ktrunc           4.5 MB      9.0 MB   2.00x      6.013e+07   8.925e-03   CAUGHT   (every row summed over K/2 only;
                                                                            every output written, looks plausible)
    and the reason it matters: honest 143.00 GB/s vs ktrunc 301.97 GB/s (2.11x inflation on the SAME charged bytes)

C4  TERNARY LUT PATH - bit-exact comparator, both directions (minimal corruption here = 1 code unit)
    C4a NEGATIVE : uncorrupted codes              -> 0/1536 rows differ -> silent (bit-exact)
    C4b MINIMAL  : ONE code byte 4->5 of 1179648       -> 1/1536 rows differ (row 768, expected 768) -> FIRES, at exactly the right row
    C4c GROSS    : same byte 4->0                  -> 1/1536 rows differ -> FIRES

C5  BYTE ACCOUNTING (the harness must charge exactly what the donor model charges)
    fp32 proj+head / token  =   1550057472 B  vs CONTROLLER_STAGE1_AUDIT F1 = 1550057472 B -> MATCH
    ternary MLP codes/token =    578027520 B  vs CONTROLLER_STAGE1_AUDIT F1 =  578027520 B -> MATCH
    negative control on the SAME line: F1's total ternary term is 580206592 B because it INCLUDES
    2179072 B of per-row fp32 scales this codes-only harness does not stream. 578027520 + 2179072 = 580206592 -> MATCH; the gap is exactly the scales term (audit F2)
```

### What the controls establish, and one honest negative

| control | direction | result |
|---|---|---|
| C1a | negative (nothing wrong) | reference deterministic to the bit — **silent**, correctly |
| C1b | positive (1 ULP) | **FIRES**, delta matches `dw·x` to 0.000e+00 relative, zero cross-talk |
| C1c | positive (1 ULP), fp32 leg | **does NOT fire** — reported, not hidden. See below. |
| C2 δ=0 | negative | **silent**, no false positive |
| C2 δ≥1e−4 | positive | **FIRES**, monotone through 1e+06 |
| C3 honest | negative | **silent** |
| C3 halfrows / ktrunc | positive | **CAUGHT**, both |
| C4a | negative | **silent**, bit-exact |
| C4b (1 code unit) / C4c (gross) | positive | **FIRES**, at exactly the predicted row |
| C5 | both | byte accounting MATCH; the deliberate 2,179,072 B mismatch resolves to the scales term |

**The honest negative (C1c).** The brief asked for a one-ULP corruption to be *detected by the rel-err check*.
It is not, and I am not going to dress that up. A 1-ULP move of one weight perturbs its row by ~4.7e−12
against a row conditioning of ~1e−2 — roughly **1e−9 relative, which is ~100× below the fp32 accumulation
noise floor of a K=1536 dot (4.5e−9)**. No fp32-vs-float64 comparator can resolve it; a threshold that did
would fire on every honest run. Two things are done instead, and together they cover what the control was
for:
1. **C1b proves the 1-ULP change is real and fully propagated** — the float64 reference sees it, at exactly
   the predicted magnitude, on exactly one row and no other. That is what establishes that *every element of
   W is actually consumed*, which is the property a bandwidth bench must have.
2. **C2 finds the minimal corruption the fp32 comparator does resolve: δ = 1e−4 on one element in 2,359,296.**
   That is the honest sensitivity figure, measured rather than asserted.

**C3 is the control that matters most here** and it was designed against this project's own failure mode
(`feedback_planted_controls`: *"here one fails with the PLAUSIBLE artefact"*). A GB/s harness dies by charging
for bytes the kernel never read. `ktrunc` reads half of each row, writes **every** output, produces an
entirely plausible-looking vector, and reports **301.97 GB/s against the honest 143.00 — a 2.11× inflation**.
The comparator catches it at 8.9e−03, six orders above threshold. Every rate in this document is therefore
guarded by a check that is demonstrated to fire on exactly the artefact that would fake it.

---

## 8. The ternary pshufb-LUT path at donor dimensions

> **INSTRUMENT NOTE, added 2026-08-22 (BRIEF_D5 AMENDMENT 1, see sec.12.4 / A1.2):** every GB/s figure in
> this section was produced by a `min-of-reps` estimator -- i.e. the *maximum* of the per-rep rates. That
> is a maximum order statistic: its expected value rises with the number of reps used, by construction,
> with no change in the underlying rate. **Numbers in this section were taken at different `reps` and were
> never comparable to each other on that account**, and every one of them is upward-biased by an amount
> this section did not quantify. The bias was only discovered and measured during BRIEF_D5 (sec.12.4): a
> 25-rep per-rep trace of the *same* shape and thread count gave `reported (min) = 23.44 GB/s` against
> `per-rep mean = 21.57 GB/s`, a difference of 8.7%, growing with `reps` (22.36 @ 5 reps -> 25.13 @ 30
> reps, same population). **This is not retro-edited here** -- the numbers below stand as measured and
> published -- but no comparison across them, or against a number from another section, should be read
> as tighter than roughly this order of magnitude without checking the `reps` behind each side.

**The open question (Controller #2, Q3.2):** the dense LUT path measures 11.40 GB/s at t6 at the sandbox's
`D=256`, and Controller #2 argued it is **compute-bound by ~16×, not bandwidth-bound**. If that holds at
donor `D`, the rate should be roughly **flat** in working-set size. If it does not, the rate should **fall**
with size like the fp32 curve.

Designed so it can come out either way: **one fixed kernel shape**, working set swept from cache-resident to
512 MB. Nothing about the kernel, the LUT, or the access pattern changes down a column — only how many
distinct blocks the pool holds. A compute-bound kernel is physically incapable of caring.

### 8.1 Known-positive #2 — this one *does* reproduce

`bin/gemv_donor_bench.exe lutrepro` — `engine.c: run_expert_rate()` replicated exactly (512 MB pool,
10922 × 48 KB experts, 48 i.i.d. touches/token), 2 invocations:

| | published 64.1b(2) | measured | dev |
|---|---|---|---|
| t1 GB/s | 7.45 | 7.58, 8.24 | +2% … +11% |
| t6 GB/s | 17.0 | 18.59, 18.92 | +9% … +11% |
| µs/expert (t6) | 2.88 | 2.60, 2.64 | −9% |
| t6/t1 | ×2.29 | ×2.30, ×2.45 | — |

Within ~11% on every figure, same ratio. **The ternary half of the §1b apparatus reproduces cleanly** — which
also confirms the divergence in §2 is specific to the fp32 cache-transition region, not a general property of
this machine or this rebuild.

### 8.2 The answer: BANDWIDTH-bound at donor working sets

`bin/gemv_donor_bench.exe lut --reps 3`, t6 GB/s, i.i.d. gather. Block shape fixed within each column:

| pool MB | **D=256 sandbox**<br>384×256 (48 KB) | **donor gate/up**<br>8960×1536 (6.56 MB) | **donor down**<br>1536×8960 (6.56 MB) | **donor attn**<br>1536×1536 (1.12 MB) |
|---|---|---|---|---|
| 1 block (resident) | **29.3** | **99.4 / 98.0** | **75.3 / 74.5** | **96.4 / 97.9** |
| ~3-13 | 30.9 | 77.2 | 74.7 | 89.2 |
| ~16-26 | 30.4 | 92.2 | 54.2 | 79.3 |
| ~32-59 | 25.8 | 47.9 | 35.9 | 53.8 |
| ~63-125 | 21.5 / 19.6 | 34.5 | 29.6 | 37.7 / 30.3 |
| ~256 | 18.8 | 30.2 | 27.1 | 28.2 |
| **512** | **18.4** | **30.6** | **26.4** | **27.3** |
| **resident ÷ streamed** | **1.6×** | **3.2×** | **2.8×** | **3.6×** |

*(The "1 block" row appears twice for the 6.56 MB shapes because a 1 MB pool request cannot hold one block;
the duplicate is a free repeatability check and the two agree to 1-2%. All cv ≤ 7%, mostly ≤ 4%.)*

> **Verdict: at donor dimensions the dense LUT path is BANDWIDTH-bound, not compute-bound.**
> Holding the kernel completely fixed and only growing the working set from 1.1 MB to 512 MB drops the rate
> **3.6×** (96.4 → 27.3 GB/s). A compute-bound kernel cannot do that. **Controller #2's Q3.2 reading is
> falsified at donor `D`.**

### 8.3 Why the sandbox said otherwise — the claim does not transfer across `D`

The sandbox measurement was taken at `D=256`, and **that is where the reasoning broke**. The measured
*compute ceiling* — the fully-resident rate, one block, no memory pressure — is:

| shape | resident t6 GB/s = the kernel's own ceiling |
|---|---|
| 384×256 (the sandbox expert) | **29.3** |
| 1536×1536 (donor attn) | **96.4-97.9** |
| 8960×1536 (donor gate/up) | **98.0-99.4** |
| 1536×8960 (donor down) | **74.5-75.3** |

**At donor `D` the kernel is 2.5-3.4× faster than at `D=256`.** At `D=256` the ceiling (29 GB/s) sits close
enough to the streamed rate (18.4) that the path *looks* compute-limited — resident÷streamed is only 1.6×.
At donor `D` the ceiling moves to ~97 GB/s while the streamed rate stays ~27, and the path is plainly
memory-limited. **A conclusion measured at `D=256` about which resource binds does not survive the move to
`D=1536`** — the same lesson as Phase 61's "microbench compute-bound does not compose to engine
memory-bound", in the other direction.

*Mechanism — flagged as HYPOTHESIS, not asserted* (`feedback_verify_public_claims`): the small-shape penalty
is consistent with per-call OpenMP fork/join plus a short outer loop (`384/32 = 12` row-blocks against
`1536/32 = 48`) amortising badly at 1.68 µs/call. This was **not** isolated by measurement here and must not
be reported as established.

### 8.4 One decode token of the donor's ternary MLP

`bin/gemv_donor_bench.exe lutstack --reps 5`, 3 invocations. 28 × (gate, up, down) packed tile-major and
walked in layer order. **Byte accounting: 578,027,520 B — EXACT MATCH to the F1 codes term** (§7 C5).

| threads | GB/s | ms/token | cv% |
|---|---|---|---|
| t1 | 8.63 ± 0.04 | 66.97 ± 0.28 | 1.8-2.1 |
| **t6** | **27.64 ± 0.91** | **20.93 ± 0.71** | 1.6-2.4 |

Sequential (this) and i.i.d. (§8.2, 27.3 GB/s at 512 MB) agree to 1% — at 6.56 MB granularity the gather
pattern costs nothing, consistent with probe-4's granularity finding.

Note the streamed ternary rate (27.6) is **~71% of the streamed fp32 rate (38.8)**, so the path does not
reach full DRAM bandwidth either. It is bandwidth-*bound* without being bandwidth-*saturating*. Isolating
that residual is out of scope here and is flagged as open.

### 8.5 Consequence — and the necessary caveat

`donor_inventory.py` charges donors **`DENSE_LUT_GBS = 11.398`**, derived from the *engine-integrated* dense
MLP at `D=256`. The kernel-pure rate at donor shapes is **27.6 GB/s — 2.4× faster.**

**The caveat is load-bearing and must travel with the number.** 11.398 is engine-integrated (it carries
dequant, dReLU, per-row scale multiply, combine); 27.6 is kernel-pure. They are not the same quantity, and
**this measurement cannot tell you where between them the integrated donor rate lands.** 64.1b(2) found the
analogous gap on the MoE path to be ~8.4 µs/expert of overhead around the kernel. The honest statement is a
bracket, `MLP ∈ [20.93 ms kernel-pure, 50.90 ms as currently charged]`, and closing it needs an
engine-integrated measurement at donor dimensions, which is **not** in this brief.

**This does not change the §5.3 verdict flip, and that matters:** the gate is cleared by the proj measurement
**alone**, at the tool's own pessimistic MLP rate (10.22 tok/s, §5.3). The LUT finding only widens the margin.
For completeness, with both measured constants:

| KV assumption | proj ms | MLP ms | KV ms | total ms | **tok/s** | gate ≥10 |
|---|---|---|---|---|---|---|
| tool's charge (34.0 proj, 11.4 LUT), 4-bit KV | 45.59 | 50.90 | 5.87 | 102.40 | 9.77 | FAIL |
| **measured proj only**, tool's LUT, 4-bit KV | **41.07** | 50.90 | 5.87 | 97.87 | **10.22** | **PASS** |
| **both measured**, 4-bit KV (unbuilt, F8) | **41.07** | **20.99** | 5.87 | 67.96 | **14.71** | **PASS** |
| **both measured**, **fp16 KV — the only precision built** | **41.07** | **20.99** | 22.40 | 84.49 | **11.84** | **PASS** |

The last row is the one worth noticing: **on measured rates the best donor clears the gate even with fp16 KV**,
which was F8's single largest objection to F1's correction. That objection is now answered on measurement
rather than on a bracket — subject to the §8.5 kernel-pure caveat, and subject to everything in §9.

---

## 10. The engine-integrated ternary MLP at donor dimensions (the bracket that decided the gate)

> **INSTRUMENT NOTE, added 2026-08-22 (BRIEF_D5 AMENDMENT 1, see sec.12.4 / A1.2):** every GB/s figure in
> this section was produced by a `min-of-reps` estimator -- i.e. the *maximum* of the per-rep rates. That
> is a maximum order statistic: its expected value rises with the number of reps used, by construction,
> with no change in the underlying rate. **Numbers in this section were taken at different `reps` and were
> never comparable to each other on that account**, and every one of them is upward-biased by an amount
> this section did not quantify. The bias was only discovered and measured during BRIEF_D5 (sec.12.4): a
> 25-rep per-rep trace of the *same* shape and thread count gave `reported (min) = 23.44 GB/s` against
> `per-rep mean = 21.57 GB/s`, a difference of 8.7%, growing with `reps` (22.36 @ 5 reps -> 25.13 @ 30
> reps, same population). **This is not retro-edited here** -- the numbers below stand as measured and
> published -- but no comparison across them, or against a number from another section, should be read
> as tighter than roughly this order of magnitude without checking the `reps` behind each side.

**Why this section exists.** After §4–§8 the projection term was effectively a point
([41.48 .. 40.70] ms) while the ternary MLP term still spanned **[20.99 .. 50.91] ms — a 30 ms bracket
against a 2.13 ms margin.** Its low end was *engine-integrated at `D=256`*; its high end was
*kernel-pure at donor `D`*. Different quantities. The missing number was the **integrated** rate at
donor `D`, and nothing else could decide the gate while it was open.

**Harness:** `gemv_donor_bench.exe mlpint` / `mlpctl`. `mlp_dense_int()` is `engine.c mlp_dense()`
(LUT path) copied **operation for operation** and parameterised by `(D, HID, L)`. Integrated means it
carries everything the kernel-pure figure omits: the int8 activation quantisation, the LUT build, the
per-row fp32 scale multiplies, the dReLU gate, and the **second** quantise + LUT build before the down
matvec. `rmsnorm` is excluded exactly as `engine.c` excludes it (`forward_token` lines 374-379 put it
in the `norm` bucket, not `mlp`).

### 10.1 Known-positive — against the LIVE ENGINE, not a historical table

The repo *does* carry the artefacts (`results/phase60/e1_model.bin`, `results/phase55/ids.u16`), so the
real engine's own `--timing` decomposition could be run rather than trusted:

```
bin/engine_64_1b_check.exe --weights results/phase60/e1_model.bin --timing \
    --mlp lut --skip {on|off} --exp {exact|fast} --threads {1|6}
```

**First finding: the published `LUT-MLP dense 313→207` is the `--skip on --exp fast` (E3.5) config.**
That was not documented and it matters. Running E3.5 today reproduces the *whole* §1 decomposition:

| organ (t6, µs/tok) | scan-recur | proj-GEMV | SWA-attn | **LUT-MLP** | head | norms+glue |
|---|---|---|---|---|---|---|
| `PHASE64_BUDGET` §1 published | 46 | 272 | 65 | **207** | 8 | ~7 |
| engine E3.5 today (mean of 3) | 36.5 | 222.8 | 48.5 | **166.3** | 6.3 | 7.3 |

Every organ reproduces to within −20%, and the *shape* matches exactly. **So `DENSE_LUT_GBS = 11.398`
is derived from the dReLU row-skip path** — a path that streams *fewer* bytes than the 2,359,296 it is
charged for. Charging a **SiLU-gated donor** that rate is wrong twice over: wrong `D`, and wrong path.

**The donor-relevant config is `--skip off`** (no dReLU sparsity is available for free on a SiLU-gated
donor; that is a quality change P61/probe-2 price separately). Harness vs live engine, `skip=off`, t6:

| | runs | mean µs/tok | sd | agreement |
|---|---|---|---|---|
| **engine `--skip off`, t6** | 95.6, 92.5, 94.1, 86.4, 92.0, 84.4 | **90.8** | 4.5 (5.0%) | — |
| **this harness, `skip=0`, t6** | 89.9, 103.1, 97.6, 121.7 | **103.1** | 13.3 (12.9%) | **+13.5%** |

**Stated plainly: agreement is ~14% on the means, with overlapping ranges, and both are noisy at this
size** (the sandbox MLP is a 2.4 MB working set and ~90-120 µs of work). I am not going to call that a
tight match. It is good enough to license the donor point below — which is 300× larger and, as the next
table shows, **11× tighter** — but it is not the ±5% the projection known-positive achieved, and the
donor figure should be read with that in mind.

### 10.2 The measurement

`bin/gemv_donor_bench.exe mlpint --reps 3`, **4 independent invocations**, `OMP_PLACES=cores`.
Bytes charged = 580,206,592 B/token (codes 578,027,520 + per-row fp32 scales 2,179,072 — both
byte-exact against the F1 hand-derivation, §10.3 M4).

| config | D | ffn | L | skip | t1 µs/tok | t6 µs/tok | in-run cv | **t6 GB/s** |
|---|---|---|---|---|---|---|---|---|
| sandbox | 256 | 1024 | 6 | 0 | 173-175 | 89.9-121.7 | 2.2-23.7% | 19.8-24.7 |
| sandbox | 256 | 1024 | 6 | 1 | 263-284 | 232-284 | 1.7-6.3% | 8.5-10.4 |
| **DONOR** | **1536** | **8960** | **28** | **0** | 68.7-73.1 ms | **27.31 ms** | **0.7-1.9%** | **21.25** |
| DONOR | 1536 | 8960 | 28 | 1 | 78.4-80.0 ms | 62.6-72.3 ms | 0.3-3.9% | 8.0-9.3 |

> **THE NUMBER: the engine-integrated dense ternary MLP at donor dimensions costs
> 27.31 ± 0.46 ms/token at t6 = 21.25 ± 0.36 GB/s.**
> 4 runs, between-run sd **1.7%**, in-run cv 0.7-1.9%. Comfortably reportable.

**The bracket collapses.** Against the two ends it was spanning:

| | rate | ms/token | vs measured |
|---|---|---|---|
| tool's charge (`DENSE_LUT_GBS`, integrated at `D=256`, **skip-on path**) | 11.40 GB/s | 50.91 | **over-charges by 1.86×** |
| **MEASURED: integrated at donor `D`, skip-off** | **21.25 GB/s** | **27.31** | — |
| kernel-pure at donor `D` (§8.4) | 27.64 GB/s | 20.93 | under-charges by 1.31× |

**The integration overhead at donor `D` is +30.5%, not the 2.4× the bracket assumed.** That is the
whole finding: the quant / LUT-build / scale-multiply / dReLU envelope is a fixed cost per layer that
scales with `D + ffn`, while the matvec work scales with `D · ffn`. At `D=256` the envelope is a large
fraction of the block; at `D=1536` it is a third. **A conclusion about which resource binds, measured at
`D=256`, does not transfer to `D=1536`** — the same lesson §8.3 drew for compute-vs-bandwidth, now
repeated for integrated-vs-kernel-pure.

**Robustness note.** With `skip=0` the block is branch-free and data-independent, so this rate does
**not** depend on the synthetic weights' distribution or on activation sparsity. Only `skip=1` is
data-dependent, which is why its numbers are reported but not used.

### 10.3 Planted controls — and one that had to be rebuilt

`bin/gemv_donor_bench.exe mlpctl`. **The first version of this control did not fire, and that was my
error, not a property of the path.** Two defects, both worth recording because both are easy to ship:

1. It compared the corrupted output against the **float64 reference**, whose own fp32-vs-f64 rounding
   floor (5.5e−08) masked the corruption entirely.
2. It corrupted a row **dReLU had already zeroed** — unobservable *by construction*. Measured: dReLU
   leaves only **2280 of 8960 rows active (25.4%)**, so a blindly-chosen row is ~75% likely to be dead.

Rebuilt to compare **bit-exactly against the pristine fp32 output** (no floor) and to target the
**active row that sets the re-quantisation scale**. Real output:

```
M1 ACCURACY GATE -- whole integrated path (fp32) vs float64 reference, D=1536 HID=8960
   max abs-err 2.910e-10 over max|ref| 5.328e-03 -> relative 5.462e-08   -> PASS

M2 INSTRUMENT SENSITIVITY -- corrupted fp32 output vs PRISTINE fp32 output, BIT-EXACT
   dReLU left 2280 of 8960 rows active (25.4%); scale-setting row = 6390 (gh=0.021865).
   ladder A: N ternary codes corrupted inside gate row 6390 (of 768 codes in the row)
     N=0    ->    0/1536 outputs changed -> silent (negative leg, as required)
     N=1    -> 1536/1536 outputs changed -> FIRES
     ...    (monotone through N=768)
     -> minimal corruption resolved: 1 code of 768 in the row (1 of 6881280 in the matrix)
   ladder B: the per-row fp32 scale gate_sc[6390]
     1 ULP  (0.0126 -> 0.0126000009) -> 1330/1536 outputs changed -> FIRES
     delta 0e+00    ->    0/1536 outputs changed -> silent (negative leg, as required)
     delta 1e-07    -> 1330/1536 outputs changed -> FIRES
     -> minimal scale corruption resolved: delta = 1e-07
M3 THE PLAUSIBLE ARTEFACT -- down-matrix half-tile zeroing (a block reading half its codes):
       1533/1536 outputs changed -> CAUGHT
M4 BYTE ACCOUNTING
   sandbox codes/token  =    2359296 B vs engine.c-derived 2359296 B -> MATCH
   sandbox scales/token =      55296 B vs audit F2        55296 B -> MATCH
   donor   codes/token  =  578027520 B vs F1 hand-deriv 578027520 B -> MATCH
   donor   scales/token =    2179072 B vs F1 hand-deriv   2179072 B -> MATCH
```

**Both directions, on every ladder.** Note this path resolves **one code byte in 6.88 million** and a
**1-ULP** scale change — far sharper than the fp32 GEMV comparator of §7, because the comparison is
bit-exact self-referential rather than fp32-vs-f64. The `max rel change` column is unreliable under
cancellation (near-zero denominators produce 1e+25 values); **the changed-output count is the signal.**

### 10.4 Consequence

`DENSE_LUT_GBS = 11.398` should be replaced by **21.25 GB/s** for a SiLU-gated donor at `D=1536`,
and the `[11.40 .. 27.64]` bracket withdrawn — it spanned two things neither of which was the answer.
The MLP term for `Qwen2.5-1.5B` is **27.31 ms**, and the last wide bracket in the stage-1 model is
closed. What it does *not* close: `OLMoE`'s routed-expert path, which is a different kernel
(`matvec_lut_rows` over gathered experts) and is measured only kernel-pure (§8.2).

---

## 11. Status, and what is NOT established

**Status: COMPLETE.** All six items of the brief are measured: known-positive gate (§2), variance (§3),
asymptote past 96 MB (§4), real donor organ shapes (§5), the ternary compute-vs-bandwidth question (§8),
correctness + planted controls (§6, §7), plus the engine-integrated MLP measurement the audit made top priority (§10).

**Timing conditions.** The owner's quiet-machine precondition was met before any timed number here was
recorded: CPU load 10%, 55.5 GB free of 79.9 GB, `webwallpaper32` terminated; Spotify / Chrome / FanControl /
a ChatGPT client resident but idle. Every timed row carries its `cv%`. The one row that failed the ≤5% rule
— **32 MB, between-run sd 22.2%** — is marked non-reportable in §3 and §4 and is used for nothing. No other
row exceeded 5% between runs.

**What this measurement does NOT establish.**

- **Nothing about quality.** Synthetic weights, rate only. It says nothing about whether any donor can be
  converted to the engine at all, which is the actual programme question.
- **The MLP bracket is not closed** (§8.5). 27.6 GB/s is *kernel-pure*; the tool's 11.398 is
  *engine-integrated at `D=256`*. These are different quantities. The integrated donor rate lies somewhere
  between and needs a separate measurement. **The §5.3 verdict flip does not depend on this** — it clears the
  gate at the tool's own pessimistic MLP charge.
- **F7 and F8 are untouched.** No donor passes the sealed 128K SKU-B contract; 4-bit KV is unbuilt. §8.5's
  last row shows the gate is cleared even at fp16 KV, which answers F8's largest objection — but F8's other
  two assumptions (zero organ overlap, a linearly-scaled 7 µs glue term for a model 6× deeper than the
  anchor) are inherited unchanged and unmeasured.
- **`PHASE64_BUDGET.md` §1b's 16-32 MB t6 points are not reproducible** and should be annotated as
  placement-conditioned (§2.3). Any S1/S2/M1 grid row that reads `r(size)` in the 16-34 MB band inherits that
  caveat. **The donor rows do not** — they all sit past 48 MB, where the curve is placement-insensitive to 2%.
- **The 48/64 MB published points (45.5 / 45.3) did not reproduce** (measured 40.0 / 38.4, −12% / −15%) under
  any placement tried. Unexplained; flagged rather than rationalised. It does not affect the donor rows,
  which are read from directly measured points, not interpolated.
- **The §8.3 mechanism is a hypothesis, not a result.** The small-shape penalty is *consistent with* per-call
  OpenMP fork/join over a short outer loop; that was not isolated by measurement and must not be repeated as
  established (`feedback_verify_public_claims`).
- **One machine.** 3600X reference floor only. Per `feedback_portability_no_hardfit` these are runtime
  parameters of a hardware class, not constants of the architecture.
- **Not audited by anyone but me.** This is a Builder artefact and has had no Controller pass.

**Recommended follow-ups, in value order.**
1. Amend `donor_inventory.py`: `PROJ_STREAMED_FLOOR 34.0 → 37.74` (measured, §5.2) and re-run `analyze`.
   Restate every downstream "zero donors pass" claim.
2. Annotate `PHASE64_BUDGET.md` §1b with the placement caveat (§2.3) and the measured >96 MB extension (§4).
3. Scope Controller #2's Q3.2 compute-bound claim to `D=256` and correct audit F6's stated mechanism (§8.2-8.3).
4. Measure the *engine-integrated* dense LUT rate at donor dims to close the §8.5 bracket.

---

## 12. BRIEF D5 — the ternary LUT rate at donor projection widths (IN PROGRESS)

`docs/research/BRIEF_D5_LUT_RATE_AT_DONOR_WIDTH.md`. §10 above closed the *engine-integrated* MLP rate at
`D=1536` (21.25 GB/s) but that number was measured at **our** model width. §8.3 already found once that a
conclusion about which resource binds does not transfer across `D`; D5 asks the same question of the
21.25 GB/s constant itself, at real donor widths. **This section reports what is built and what one finding
already follows from shape arithmetic alone. No timing number in this section exists yet — see §12.4.**

### 12.1 Harness

`benchmarks/donor_adaptation/gemv_donor_bench.c`, new `d5` mode (and `benchmarks/donor_adaptation/build_d5_json.py`
to turn its tagged stdout into the deliverable JSON without hand-transcription). Reuses the apparatus that
produced every number in §8 and §10 verbatim: `MlpStack`/`mlp_alloc`/`mlp_time` (the engine-integrated path)
and `matvec_lut_full`/`build_lut_t3` (the kernel-pure LUT path) are untouched; the new code is orchestration
and capture around them, not new numeric kernels. Runs, in order: the four planted controls (brief §5) —
STOPping immediately if control 1 fails — then `rate(D,threads)` for `D∈{1536,2048,4096,5120,8192}` (ffn:D
ratio fixed at 3.5, the real Llama-3-70B ratio, so the `D=8192` point coincides with the real donor organ
measured separately), a kernel-pure fp32-vs-ternary comparison at the identical shape for every `D` in that
sweep (the direct compute-vs-bandwidth discriminator — see §12.3), and the real Llama-3-70B-class organ
shapes (q/o, k/v, gate/up, down) via the same kernel-pure sweep used in §8.2.

### 12.2 New permanent feature: the machine-quiescence gate

Every timing mode (all but `correct`, which does no wall-clock measurement) now scans resident processes
(`CreateToolhelp32Snapshot` + `GetProcessMemoryInfo`, `-lpsapi`) before running and **refuses** (`exit(3)`)
if any process holds ≥1 GB working set, unless `--force-unclean` is passed — which is itself printed into
the run header so it cannot be dropped when a number is later quoted. This is the standing law of §2.3 of
this document (`PHASE64_BUDGET.md`'s 16–32 MB points are unreproducible because environment went unrecorded
and swung 4.6× on placement alone) applied to the harness itself rather than to one experiment. **Verified
firing, live**, before any `d5` timing code path ran: with a concurrent D4 run resident at 24–56 GB (it grew
over several checks), `d5 --reps 2` printed the process list and refused at the top of `main()`, before
`mlp_alloc` or any other allocation in the new code was reached. Re-invoking `correct` (exempt, no timing)
in between confirmed the scan is live per-invocation, not cached.

### 12.3 One finding that needs no timing — and the caution that goes with it

**Derived, not measured** (pure byte arithmetic on the ternary block-packing formula `EB = (K/2)·((M+31)&~31)`,
already used unmodified throughout §8):

> At Llama-3-70B-class organ shapes (`d_model=8192`, `d_ffn=28672`), a single ternary-packed block of
> `gate`/`up`/`down` is **112 MB** and `q`/`o` is **32 MB** — each exceeds the 16 MB L3 cliff (§3, `project_probe3_cache`)
> on its own. Those organs are therefore **always in the streamed regime** at donor scale; `l3_resident` is
> false for them regardless of pool size. Only `k`/`v` at **4 MB**/block can ever be L3-resident.

This is true independent of whether any timing loop ever completes, which is why it is written down now
rather than held for the sweep.

**What it does not establish.** Exceeding L3 means the data must come from DRAM. It does **not** by itself
mean the kernel is *bandwidth*-bound — those are different claims, and this document has already drawn that
distinction once (§8.2 vs §8.3: compute-bound at `D=256`, bandwidth-bound at `D=1536`, a reversal driven by
`D`, not by working-set size alone). The two banked numbers point toward the LUT path *still being
compute-bound even while streamed* at donor width: the ternary path sustains **21.25 GB/s** (§10.2,
integrated) against an fp32 DRAM ceiling of **38.84 GB/s** (§4) on the same silicon — if DRAM can deliver
~39 and the LUT only draws ~21, the kernel, not the bus, may still be the limit. **This is exactly the open
question D5's §12.1 fp32-vs-ternary matched-shape sweep is built to answer, and it has not run yet.** Do not
read "always streamed" as "therefore bandwidth-bound" anywhere downstream of this paragraph until §12.4 is
filled in with a measured ratio.

**The stake.** `ADAPTER_MEMO_01_SPEED_BUDGET.md` §2.4's 5-trits-per-byte packing question (0.2 vs 0.5
bytes/weight, ~2.5× on bytes moved) is conditional on exactly this. Still compute-bound → denser packing is
roughly a 10% loss (fixed per-call overhead amortised over less work per byte). Bandwidth-bound → denser
packing is roughly the full ~2.5× lever, and the budget's ~24 tok/s figure could move toward ~61. The
`ratio_fp32_over_ternary` column in the (not yet run) fp32-vs-ternary sweep decides which, per `D`.

### 12.4 Status

**The machine cleared. Control 1 ran — repeatedly, under several configurations — and did not reproduce
the banked point, in a way that looks like real drift rather than a noisy row.**

**First reading, as reported to the Principal before the break:** `d5 --reps 5`, `OMP_PROC_BIND=close
OMP_PLACES=cores`: `t6 = 22.361 GB/s`, `dev = +1.111` against banked `21.25 ± 0.36`, **`cv = 10.75%`**.
By the harness's own printed criterion (`cv% > 5% => the machine was not quiet, do not use the row`) that
single row is not a measurement either way — it neither passed nor failed control 1, and the Principal's
read that the ±0.36 tolerance could not survive contact with a ±2.4 GB/s-scale instrument is correct as
far as it goes. But it does not go far enough: chasing the variance down produced ten independent `t6`
draws, not one, and nine of the ten are elevated, not merely noisy.

**Every configuration tried, same shape (`D=1536, HID=8960, L=28, skip=0`):**

| config | reps | t6 GB/s | cv% | dev vs 21.25 |
|---|---|---|---|---|
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 5 | 22.361 | 10.75 | +1.111 |
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 30 | 25.125 | 3.50 | +3.875 |
| `OMP_PROC_BIND=close OMP_PLACES=cores` (per-rep instrumented) | 25 | 23.439 | 5.27 | +2.189 |
| `OMP_PROC_BIND=close OMP_PLACES=cores`, invocation 1 of 4, matching the original protocol exactly | 3 | 20.203 | 13.56 | -1.047 |
| same, invocation 2 of 4 | 3 | 24.762 | 4.66 | +3.512 |
| same, invocation 3 of 4 | 3 | 24.763 | 2.89 | +3.513 |
| same, invocation 4 of 4 | 3 | 24.755 | 7.58 | +3.505 |
| `OMP_PLACES=cores` only (no explicit `OMP_PROC_BIND`, matching sec.10.2's documented invocation literally) | 3 | 23.461 | 3.08 | +2.211 |
| same | 3 | 24.709 | 4.33 | +3.459 |
| no thread-placement env vars at all | 3 | 22.998 | 6.68 | +1.748 |

Mean of all ten draws: **23.66 GB/s**, +11.3% above banked. **Nine of ten exceed 21.25; none fall inside
the ±0.36 band.** This is not the signature of a single unlucky high-variance row — the reading given to
the Principal before the break was, if anything, the most conservative of the ten.

**1. Diagnosing the variance -- three separable findings, not one cause.**

- **OpenMP placement was verified, not assumed.** A standalone probe (`omp_get_num_places()` /
  `omp_get_place_proc_ids()` / a Win32 `SetThreadAffinityMask` round-trip to read back each thread's live
  affinity mask) confirms `OMP_PLACES=cores` enumerates 6 places of 2 logical processors each (one
  physical core's SMT pair per place -- `[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]`), and that at `nt=6` each of
  the 6 OpenMP threads is bound to a distinct place (`0x3, 0xc, 0x30, 0xc0, 0x300, 0xc00` -- no overlap).
  **Six threads land on six distinct physical cores; SMT-sibling contention between our own threads is
  ruled out**, not merely assumed away.
- **`OMP_PROC_BIND` is not the explanatory variable.** Re-running control 1 with `OMP_PLACES=cores` alone
  (no explicit `OMP_PROC_BIND`, i.e. exactly what sec.10.2 documents having used) and with neither variable
  set at all produced the same elevated range (22.9-24.7 GB/s) as the fully-specified configuration. The
  documentation gap the constraints section warns about (unrecorded placement swinging a number 4.6x) is
  real as a *general* risk but is **not what is happening here** -- three different placement configurations
  converge to the same elevated band.
- **`t6`'s per-rep dispersion is structurally higher than `t1`'s, for a mundane reason.** A per-rep debug
  trace (25 reps, `t1` and `t6` separately) shows `t1` is tight (7.75-8.28 GB/s) except for one rep that
  stalled to 130.6 ms against a typical ~71 ms -- a single transient OS-level interruption, not a trend.
  `t6` has no such single outlier but a wider intrinsic spread every rep (19.18-23.44 GB/s, cv ~5-6% even
  in a clean sample). This is consistent with a barrier-synchronised 6-thread reduction: each rep's time is
  set by the slowest of 6 concurrently-running physical cores, so the probability that *some* core takes a
  transient hit (DPC, scheduler quantum, interrupt) scales with core count. Six threads, six times the
  surface area, versus one.
- **The `min-of-reps` estimator (inherited unmodified from `engine.c run_gemv_sweep`, by design -- see the
  comment at the estimator's definition) is a maximum order statistic: its expected value is monotonically
  non-decreasing in `reps`.** With real per-rep dispersion at `t6` (~5-6%), sampling more reps mechanically
  pulls the reported "best" rate toward the population's upper tail -- 22.36 at `reps=5`, 23.44 at
  `reps=25`, 25.13 at `reps=30`, drawn from the *same* per-rep distribution (mean of the 25-rep trace ~
  21.57 GB/s, close to banked). **Comparing a `reps=N` reading against a tolerance calibrated at a
  different `N` is not apples-to-apples**, independent of any drift question. But this alone does not
  explain the finding below.

**2. What the banked `21.25 ± 0.36` actually was (DONOR_PROJ_RATE.md sec.10.2, re-read against today's
data).** `gemv_donor_bench.exe mlpint --reps 3`, **4 independent invocations**, `OMP_PLACES=cores` (no
`OMP_PROC_BIND` recorded). In-run cv 0.7-1.9%. Between-run sd **1.7%** (-> +-0.36 at +-1 sigma). An independent
Controller replication (`ctrl3_donor_bench.exe mlpint --reps 3`, different build, different session) landed
at 21.55-21.63 GB/s, agreeing with the Builder's figure. **Reproducing that exact protocol today --
`reps=3`, 4 independent invocations, same env -- gives 20.203 / 24.762 / 24.763 / 24.755: mean 23.62 GB/s,
between-run sd 8.36%, roughly 5x the historical dispersion, and a mean +11.1% above banked.** Matching `N`
does not rescue the reproduction. **The comparison was apples-to-apples in protocol; the instrument itself
now returns a different number under that protocol than it returned before.** That is the finding -- not an
estimator artifact, not an unrecorded-placement artifact, both of which were tested and ruled out above.
What changed on the machine between the original banked session and today (OS/driver update, AMD chipset
power-management change, BIOS/AGESA revision, thermal/ambient conditions not caught by the >=1 GB
quiescence-gate threshold) is **not established** -- the effect is real and repeated, the cause is not yet
known, and this document will not guess at it.

**3. Control 1, plainly: FAIL, and more decisively than the single row first reported.** Not one
high-variance reading -- ten draws across three placement configurations and four rep-counts, nine of which
exceed banked, with the matched-protocol replication (the fairest test available) landing at +11.1% with
5x the historical between-run dispersion. **Per the brief and per standing instruction, this is a STOP: no
new tolerance has been adopted, 22.361 (or any other single reading) has not been adopted as a replacement
banked value, and nothing past control 1 has been run.** A defensible tolerance is the Principal's call
once amended in a dated correction to the brief (pre-registration is not edited in place); building one
here is out of scope for the Builder. Candidates worth the Principal's attention when that correction is
drafted: (a) re-baseline `21.25` on this machine, today, under the exact matched-`N`, multi-invocation
protocol above, rather than compare across sessions; (b) if cross-session comparison is kept, widen the
tolerance to reflect the instrument's *demonstrated* between-run dispersion (~8% today) rather than the
historical 1.7%, which this session shows does not hold; (c) either way, pin `reps` and invocation count in
the brief itself so the comparison is never again made across different `N`.

**Housekeeping folded into this touch of the file (not a numeric change):** a diagnostic-only per-rep
trace, gated behind `GEMV_D5_DEBUG_REPS` and printed to stderr, was added to `mlp_time()` in
`gemv_donor_bench.c` to produce the per-rep evidence above; it does not alter `*us_tok`, `*cv`, or any
control's pass/fail computation. Three stale doc-path references (`gemv_donor_bench.c:9`,
`gemv_donor_bench.c:1170`, `build_d5_json.py:202`) were updated to the paths the reorganisation moved them
to; no numeric kernel was touched.

**Not yet measured, still pending a tolerance the apparatus can actually satisfy:** controls 2-4; the
`rate(D,threads)` curve; the L=2-vs-L=28 sensitivity check; the fp32-vs-ternary matched-shape ratio at each
`D` (sec.12.3); the timed rate at the real Llama-3-70B organs. **No line above this one may be read as a rate
measurement -- only the shape arithmetic in sec.12.3 is established, and control 1's ten draws establish that
the 21.25 GB/s constant does not currently reproduce on this machine, not what the machine's true rate is.**
## 12.5 Results, under Amendment 1 (2026-08-22)

**Status: measured, not yet audited by the Controller. Per standing instruction, nothing in this section
may be quoted anywhere outside this document until that audit is done.** This section reports the D5
sweep run (`gemv_donor_bench.exe d5 --reps 5`) plus follow-up measurements taken while attacking the
Principal's own reading of that sweep, per the Principal's explicit request not to let the raw log's
numbers pass into the record unchallenged.

### 12.5.1 Reproducibility manifest

| field | value |
|---|---|
| git revision | `ad5cf075b4c66ba9dbc4e001023dbba4e507658b` |
| build | `clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp benchmarks/donor_adaptation/gemv_donor_bench.c -o bin/gemv_donor_bench.exe -lm` (no `-ffast-math`) |
| compiler | `clang version 21.1.8` (llvm-mingw, x86_64-w64-windows-gnu) |
| CPU | AMD Ryzen 5 3600X, 6 physical cores / 12 logical (SMT2) |
| thread placement | `OMP_PROC_BIND=close OMP_PLACES=cores` -- verified live (sec.12.4) to bind each of 6 OpenMP threads to a distinct physical core, no SMT-sibling sharing |
| machine state | quiescence gate: 0 processes >=1 GB resident, clean |
| sweep command | `OMP_PROC_BIND=close OMP_PLACES=cores GEMV_D5_A1_VERDICT=PASS bin/gemv_donor_bench.exe d5 --reps 5 > benchmarks/donor_adaptation/results/d5_raw.log 2>&1` |
| `reps` (sweep) | 5, single invocation, pinned for the whole run (A1.5) |
| control-1 verdict source | external, NOT this invocation -- see sec.12.4: 14 independent `d5c1` invocations, grand mean 22.07 GB/s, between-invocation sd 4.01%, PASS at +-3sigma. `GEMV_D5_A1_VERDICT=PASS` carries that externally-computed verdict into this run's `D5SUMMARY` line rather than re-deriving it from one process (a single process cannot compute a between-invocation sd) |
| follow-up probe (this section) | new `d5g` mode, `--D --HID --L --nt` generic six-stat probe, same build/env, run separately from the pinned sweep (noted per-measurement below) |
| raw log | `benchmarks/donor_adaptation/results/d5_raw.log`, untracked, holds its own `STOP. No commit, no push.` line |

### 12.5.2 The four controls

| control | verdict | numbers |
|---|---|---|
| 1 (A1.3, external) | **PASS** | grand mean 22.07 GB/s over 14 invocations, between-invocation sd 4.01% (+-2.66 GB/s at 3sigma), dev vs banked 21.25 = +0.82 GB/s -- see sec.12.4 |
| 2 (forced slowdown) | **PASS** | L3-resident 95.73 GB/s -> streamed(512MB) 27.69 GB/s, drop ratio **3.46x** (need >=2.0x) |
| 3 (forced speedup) | **PASS** | L3-resident 95.73 GB/s (need >=50, target ~100 per the banked L3 probe) |
| 4 (fp32 cross-check) | **PASS** | mean 38.717 GB/s, sd 0.401, banked 38.84+-0.68, audited 39.87+-0.09, tolerance +-2.04 (3sigma) |

All four fire. Control 1's mechanism changed under Amendment 1 (external multi-invocation aggregation
replacing the single-shot +-0.36 gate); controls 2-4 are unaffected by that amendment -- they are
directional/threshold checks against known-positive behaviour, not comparisons against the drifted
absolute constant, and their own reported numbers are self-consistent with ample margin.

### 12.5.3 `rate(D, threads)` -- Part A, and the L-sensitivity flag the Principal caught

**Raw sweep (`L=2`, `ffn:D` ratio fixed at 3.5, min-of-5 estimator, thread-placement as above):**

| `D` | `HID` | ws (MB) | t1 GB/s | t6 GB/s | t6 cv% |
|---|---|---|---|---|---|
| 1536 | 5376 | 23.6 | 6.25 | 52.30 | 3.47 |
| 2048 | 7168 | 42.0 | 5.46 | 27.51 | 1.78 |
| 4096 | 14336 | 168.0 | 5.28 | 17.27 | 1.25 |
| 5120 | 17920 | 262.5 | 5.59 | 18.16 | 1.39 |
| 8192 | 28672 | 672.0 | 2.42 | 10.69 | **9.13** |

**The Principal is right to refuse the L-sensitivity check as a licence, and it is worse than the single
number in the log suggested.** The printed check (`L2_t6_gbps=29.500, L28_t6_gbps=21.137, ratio=1.396`)
has two problems, both checked directly rather than argued from a distance:

1. **It was run at the wrong shape.** It used `D=1536, HID=8960` -- the *control-1 / Qwen2.5-1.5B* shape --
   not `D=1536, HID=5376`, the shape the D-sweep actually uses at that point (`HID = 3.5D`, the Llama-3-70B
   ratio). A licence measured on one shape was applied to a sweep run on a different one at every point.
2. **It was measured with the retired min-of-reps estimator**, the same instrument defect as control 1
   (sec.12.4 / A1.2).

Both were re-measured directly with the new `d5g` mode (mean/median/sd, not min), at the D-sweep's *actual*
shapes, at the two ends of the sweep:

| `D` | `HID` | `L=2` mean GB/s | `L=28` mean GB/s | ratio (mean-based) | ratio (old, min-based) |
|---|---|---|---|---|---|
| 1536 | 5376 (actual dsweep shape) | 37.34 | 24.54 | **1.522x** | -- (not measured at this shape) |
| 1536 | 8960 (the check's shape) | 31.31 | 24.75 | 1.265x | 1.396x |
| 8192 | 28672 (actual dsweep shape, = donor width) | 11.39 | **10.38** | **1.097x** | -- |

**Reading this plainly: the gap is real (survives the switch to the unbiased mean estimator at both
shapes tested), it is *not* uniform across the sweep, and it is smallest exactly at the point that matters
most.** At `D=1536` the true `L=28` rate is roughly 66% of what `L=2` reports (1.52x inflation) -- a large,
material gap, not a rounding difference. At `D=8192` the gap nearly closes (1.10x) -- consistent with
every layer's per-block working set (352 MB) being so far past the 16 MB L3 cliff that cycling through 28
of them instead of 2 barely changes the access pattern, whereas at `D=1536` each layer's block (12.4 MB)
sits *just under* the L3 cliff, where the layer *count* materially changes whether the working set behaves
as cache-assisted or fully cold. **`D=2048/4096/5120` were not re-measured at `L=28`; their true rates are
unknown, not interpolated, and should not be read as sitting between these two bookends without a direct
measurement.** The dsweep table above is therefore an **upper bound** on the true `L=28` curve, not a
faithful stand-in for it, with the size of the overstatement ranging from confirmed ~10% (at `D=8192`) to
confirmed ~52% (at `D=1536`) at the two points actually checked. **This is stated here explicitly so the
inflated rows are not quietly carried forward; they are marked, not fixed.**

**The one number that matters most is the one where the licence problem bites least.** For "does 21.25
transfer to donor width," the load-bearing figure is the *directly measured* `D=8192, L=28` integrated
point: **mean 10.38 GB/s** (min 11.15, `reps=5`, same shape/threading as control 1) -- not the sweep's
`L=2` proxy (10.69, itself close by coincidence at this specific `D`, for the reason just given).

### 12.5.4 fp32-vs-ternary ratio, in both units -- attacking the Principal's inference directly

**Raw ratio (kernel-pure, matched gate/up shape at each `D`, both arms min-of-5, same reps -- from the log):**

| `D` | ternary t6 GB/s | fp32 t6 GB/s | `ratio_fp32_over_ternary` |
|---|---|---|---|
| 1536 | 23.19 | 38.13 | 1.644 |
| 2048 | 22.97 | 38.45 | 1.674 |
| 4096 | 20.66 | 38.35 | 1.856 |
| 5120 | 20.52 | 39.45 | 1.922 |
| **8192 (donor width)** | **11.34** | **38.71** | **3.412** |

The Principal's message quoted the first three (1.644/1.674/1.856) and read them as the answer. **The
full curve is worse for us, not better: `D=5120` and especially `D=8192` -- the point that actually matches
donor width -- are not in that quote, and `D=8192`'s ratio (3.412) is the largest in the table, not a
middling one.** Corrected for the record: quote the whole curve, not the first three rows.

**Is the GB/s comparison like-for-like? No -- and stating it in the other unit does not save the packing
argument, it corroborates the same conclusion from the opposite direction.** Ternary (current packing,
`T=K/2` -- confirmed from the kernel: one packed code byte per 2 weights) moves **0.5 bytes/weight**; fp32
moves **4 bytes/weight**, an 8x difference. Converting both arms to **weights/s** (`GB/s x 1e9 / bytes_per_weight`):

| `D` | ternary Gweights/s | fp32 Gweights/s | ratio (ternary/fp32, weights/s) |
|---|---|---|---|
| 1536 | 46.39 | 9.53 | 4.87x |
| 2048 | 45.94 | 9.61 | 4.78x |
| 4096 | 41.32 | 9.59 | 4.31x |
| 5120 | 41.04 | 9.86 | 4.16x |
| **8192** | **22.69** | **9.68** | **2.35x** |

**Two genuinely different questions, two genuinely different (and correct) units, and they agree.**
Weights/s answers "which kernel does more useful work per second" -- ternary wins by 2.3-4.9x, which is
the entire reason to use ternary at all, and is not in dispute. **Bytes/s vs the machine's demonstrated
DRAM ceiling (fp32's ~38.7 GB/s) is the unit the bandwidth-vs-compute question is actually asked in**,
because bandwidth-bound-ness is a statement about bytes moved per second hitting a *physical* ceiling,
independent of what those bytes encode. Denser packing's entire value proposition is reducing bytes
moved per weight -- **which only buys anything if the kernel is currently constrained by bytes/s.** The
weights/s table does not change that: it shows ternary is doing more *work*, not that it is closer to
*saturating memory bandwidth* -- and by the bytes/s measure it is at 29-33% of the demonstrated ceiling at
every `D`, falling further behind (not closing the gap) as `D` grows. **Both readings point the same way:
compute-bound, worsening at donor width, not improving.**

**A decisive discriminator beyond the ratio, already sitting in this log at zero additional cost:
thread-scaling.** A bandwidth-bound kernel saturates a shared, finite DRAM channel quickly -- adding cores
buys little. A compute-bound kernel, each core working its own LUT/ALU throughput, scales close to
linearly until *something else* (not the shared bus) limits it. The `t1`->`t6` scaling factor for both
arms, every `D`, already in the fp32-vs-ternary rows above:

| `D` | ternary t1->t6 scaling | fp32 t1->t6 scaling |
|---|---|---|
| 1536 | 3.78x | 1.54x |
| 2048 | 4.01x | 1.61x |
| 4096 | 2.40x | 1.53x |
| 5120 | 3.09x | 1.63x |
| 8192 | 4.78x | 1.60x |

**fp32 is pinned at ~1.5-1.6x scaling from 1 to 6 threads at every single `D` -- the textbook signature of
a kernel that saturates DRAM bandwidth with 2-3 threads and gets almost nothing from the rest.** Ternary
scales 2.4-4.8x at every `D` -- well short of ideal 6x, but 1.6-3x better than fp32's ceiling-bound
behaviour, consistent with headroom left on the memory bus, i.e. **compute, not bandwidth, is what is
limiting it.** This corroborates the ratio finding with an independent signature, at no additional
measurement cost. **A stronger, causal (not just correlational) test would strip the LUT-gather/accumulate
out of the kernel while streaming the identical bytes** (a compute-ablation variant of `matvec_lut_full`)
and check whether the rate jumps (compute-bound: yes) or holds (bandwidth-bound: no). Cost: a new kernel
variant plus a short run, estimated 30-60 minutes of harness work -- not built, flagged as the next thing
to build if the Controller wants a third, causal signature rather than two corroborating correlational ones.

### 12.5.5 Real Llama-3-70B organ shapes -- width or shape?

**Kernel-pure, largest streamed pool tested per organ (~470-537 MB, all well past the 16 MB L3 cliff):**

| organ | `M x K` | single-block MB | t1 GB/s | t6 GB/s |
|---|---|---|---|---|
| q/o | 8192x8192 | 32.0 | 2.66 | **8.92** |
| k/v | 1024x8192 | 4.0 | 6.76 | **29.23** |
| gate/up | 28672x8192 | 112.0 | 2.39 | **11.05** |
| down | 8192x28672 | 112.0 | 2.56 | **10.55** |

**Shape, not `d_model` width, drives the rate.** k/v shares `K=8192` (the same "width") with q/o but is
**3.3x faster** (29.2 vs 8.9 GB/s) -- the difference is its output dimension `M=1024`, 8x smaller than
q/o's `M=8192`. gate/up and down have transposed shapes (`28672x8192` vs `8192x28672`) but the **same**
block size (112 MB) and land within 5% of each other (11.05 vs 10.55) -- consistent with block size (not
which axis is `M` vs `K`) being the driver once both are deep in the streamed regime. **q/o is the one
genuine anomaly**: it has the *smallest* block of the three organs that exceed L3 (32 MB, a third of
gate/up's and down's 112 MB) yet is the *slowest* (8.92 vs ~10.5-11.0). This does not fit a simple
"bigger block, comparable rate" story and is flagged as unexplained, not resolved -- worth a follow-up if
q/o's shape matters for the programme (it is the attention path, so it likely does).

**These organ numbers are kernel-pure, like Part-B always has been -- they are not directly comparable to
the banked 21.25/22.7 GB/s integrated constant.** §10's own finding stands: integration overhead (quant,
LUT build, per-row scale, dReLU) adds ~30.5% at donor `D`. The correct like-for-like comparison for "does
the constant transfer" runs through Part A / the fresh `d5g` point below, both integrated, both `L=28`,
both the same estimator -- not through this table.

### 12.5.6 The actual answer to D5's question

> **Does the 21.25 GB/s constant transfer to donor width? No. Is the donor-width path bandwidth-bound or
> compute-bound? Compute-bound.**

**The transfer question, answered at matched methodology (engine-integrated, `L=28`, mean estimator, same
thread placement):**

| point | mean GB/s |
|---|---|
| `D=1536` (banked shape) | **22.07** (14-invocation grand mean, sec.12.4) |
| `D=8192` (donor width, directly measured `L=28`, this section) | **10.38** |
| **fall** | **2.13x** |

This is close to, and independently arrived at from, the Principal's own reading of the raw log (~2x at
`llama70b_down`) -- but arrived at through the corrected like-for-like route (integrated-vs-integrated,
`L=28`-vs-`L=28`, mean-vs-mean) rather than the raw log's kernel-pure organ number, and the two agree.
**The constant does not transfer. It falls by roughly half at donor width.** Every tok/s figure in
`ADAPTER_MEMO_01` that is built on 21.25 GB/s is now known to be built on a number that is not the donor
figure -- by a factor of about 2, in the pessimistic direction.

**The bandwidth-vs-compute question, answered by two independent, mutually corroborating signatures, at
every `D` tested, worsening (not improving) toward donor width:**

- `ratio_fp32_over_ternary` is **>1 at every `D`** (1.64 to 3.41) and *grows* with `D` -- the ternary
  kernel gets *further* from the fp32 DRAM ceiling as it gets wider, not closer.
- Thread-scaling: fp32 is capped at ~1.5-1.6x (1->6 threads) at every `D` -- saturating DRAM already at 2-3
  threads. Ternary scales 2.4-4.8x -- not linear, but consistently, substantially better than fp32's
  bandwidth ceiling allows, at every `D`.

**Consequence for the packing question (`ADAPTER_MEMO_01` sec.2.4): on this evidence, denser (5-trit)
packing does not pay.** The pre-registered reading (sec.12.3, brief sec.4) was explicit that `ratio > 1`
means compute-bound and packing buys roughly a 10% loss from fixed per-call overhead amortised over less
work per byte, not the ~2.5x bandwidth lever. **That is the reading this measurement supports, at every
`D` swept, most strongly at donor width.** This is a negative result, arrived at before anything was spent
building the denser packing path -- which is the point of having run this brief before that work started.

### 12.5.7 What this changes, and what it does not

- **`ADAPTER_MEMO_01` sec.2.1's caveat is not discharged and not merely tightened -- it is resolved in the
  pessimistic direction.** The constant does not transfer; it should be replaced with a donor-width figure
  roughly half its value for any tok/s computation that spans donor `D`, not adjusted by a correction
  factor on 21.25 itself.
- **sec.2.4's packing question is answered negatively** on two independent, corroborating signatures
  (ratio and thread-scaling), consistently across the whole `D`-sweep, most strongly at the donor-width
  point itself.
- **What remains open:** the `D=2048/4096/5120` points' true `L=28` rates (bounded above by the `L=2`
  sweep, unmeasured directly); q/o's anomalous slowness relative to its block size; a causal (not just
  correlational) compute-ablation test, costed above at 30-60 minutes, not yet built.
- **This section is a measurement and an audit-in-progress, not a decision.** Per standing instruction,
  nothing above enters `ADAPTER_MEMO_01` or any other document until the Controller has reviewed it.
