# BRIEF E4 — is the attention organ latency-bound? A 2x2 on the Q·K reduction

Pre-registered 2026-09-06, **pushed before the arms exist**. Successor to
`probes/E3_ENGINE_AT_TARGET_SCALE.md` §4.6, which named this experiment and marked its own reading
**corroborated, not proven**.

---

## 1. What this asks, and what it does NOT ask

E3 measured the `attention` organ at **3.72–4.27 cycles per FMA per thread across all twelve
measured points (±7%)**, invariant to shape, context length, KV cache size over a 16x range, GQA
factor and vocabulary. Two explanations survive that observation:

- **(a) latency.** `for(i<HD) d += qh[i]*kt[i]` (`donor_engine.c:552`) is a floating-point
  **reduction**. Built `clang -O3 -mavx2 -mfma -ffp-contract=on` with **no `-ffast-math`** (forbidden
  since Phase 35), the compiler may neither reassociate nor vectorise it, so the loop runs one scalar
  FMA per FMA *latency* — ~4–5 cycles on Zen2. The measured constant is that latency.
- **(b) bandwidth.** The loop reads the KV cache and is limited by the load path, and the constant is
  a coincidence of this machine's bytes-per-FMA.

E4 discriminates them. **It is not an attempt to make today's `T10` faster** — at 300 context the
whole `attention` organ is 9.862 ms of a 323.6 ms token, so deleting it entirely would buy **+3.1%**
(3.090 → 3.19 tok/s), and at 800 context **+8.1%**. Anyone reading this for a headline tok/s number
is reading the wrong experiment.

**What it is for.** `f = rope + attention + norm+glue` is the cost **no weight format can remove**,
and `1000/f` is the ceiling on every future sparse/MoE/ternary win. E3 measured that ceiling at
**38.3 tok/s at 800 context on a 10 B shape — below the 50 tok/s goal.** At `T10` @800,
**attention is 25.385 of the 26.088 ms of `f`, i.e. 97.3% of it.** The goal is bounded by this loop
and by nothing else on the non-weight side. That is the whole reason to touch it.

---

## 2. Arms — one binary, runtime flag, so no arm differs by a build

`--attn {serial,ilp4,avx1,avx4,serial2}`, dispatched **outside** the `t` loop (per head), so the
branch is `L*NH` per token and cannot appear in the measurement.

| arm | Q·K inner product | isolates |
|---|---|---|
| `serial` | **A0**, unchanged: `float d=0; for(i<HD) d+=qh[i]*kt[i]` | baseline |
| `ilp4` | **A1**: 4 independent scalar accumulators, summed at the end | ILP alone, no SIMD |
| `avx1` | **A2**: one `__m256` accumulator, `_mm256_fmadd_ps`, horizontal sum | SIMD width alone |
| `avx4` | **A3**: 4 independent `__m256` accumulators | both |
| `serial2` | **G1**: A0 run **twice**, `d = (d1+d2)*0.5f` | planted positive (see §4) |

`HD` is 64 or 128 in every shape measured, so all four are exact multiples of 8 and 32 with no
remainder loop. **The `A·V` loop (`out[i] += w*vt[i]`) is not touched**: it accumulates over `i`
independently, so the compiler already has 16 independent chains there and it is not the suspect.

---

## 3. Predictions, with their arithmetic, stated before any arm is built

At `T10` @800: `L=48, NH=32, HD=128`, mean position 400.5, 6 threads.
Dot-product FMAs per token = `48*32*128*400.5 = 78.7 M`; per thread `13.1 M`.

| | reasoning | predicted `attention` ms @800 |
|---|---|---|
| A0 measured | — | **25.385** (E3 §3.2) |
| A1 `ilp4` | chain /4, but scalar FMA throughput caps at ~2/cycle: `13.1M/2 = 6.5M` cycles ≈ 1.6 ms compute | **7–11** |
| A2 `avx1` | iterations /8, chain still one deep: `13.1M/8 × 5 = 8.2M` cycles ≈ 2.0 ms compute | **6–10** |
| A3 `avx4` | compute ≈ 0.4 ms; **the load path becomes the floor** | **5–8** |

**The predicted floor is memory, not zero.** The dot loop reads `78.7 M × 4 B = 315 MB` of `K` per
token (GQA re-reads included; unique `K` is 79 MB). At even 50 GB/s that is **6.3 ms**, so a perfect
kernel lands near 6–8 ms and **~3–4x is the most this change can give**, not 8x or 20x. Stating this
now so that a 3.5x result is not written up as a disappointment or a 6x as a miracle.

**If (b) is true instead:** all three arms land within 10% of 25.385 and none of the above happens.

---

## 4. Gates — every one of them fixed here, before the arms exist

**G1 — planted positive; the instrument must fire (`feedback_planted_controls`).**
`serial2` computes the same dot product twice and returns `(d1+d2)*0.5f`. In IEEE754 `d+d` is exact
and `*0.5f` is exact, so **`serial2` is bit-identical to `serial` while doing 2x the work.** Required:
the `attention` organ rises by **≥1.7x** at `T10` @300 **and** `--bpb` is **bit-identical** to A0.
If the organ does not move, the timer is not attached to the loop being changed and **E4 is VOID** —
this is the gate E3's §2.2 taught, at the one place it can still fail.

**G2 — parity, on a real donor, not on noise.**
`qwen25-05b_tqh.bin` + the pinned `ids_qwen25-05b_tqh.bin` slice (24x512), `--bpb`, one run per arm.
Accumulation order changes, so **bit-identity is not required and not expected** for A1/A2/A3.
Required: `|ΔBPB| ≤ σ_seed = 0.005` against A0, **and the actual magnitude is reported**, not just the
pass. E1 traced the engine-vs-PyTorch delta of `1.53e-05` to accumulation order; anything far above
that is a bug, not a rounding difference, and will be treated as one.

**G3 — timing discipline (`feedback_perf_parallelization`).** Idle machine, ≥3 reps, median with IQR,
one timer at a time. A contended timing is not a timing.

**G4 — the baseline must reproduce itself.** A0 re-measured with the new binary must land within the
E3 IQR of `9.862` @300 and `25.385` @800. If it does not, the binary changed something outside the
flag and no arm may be compared to E3's table.

---

## 5. Decision rule — fixed now

Judged on the **`attention` organ at `T10` @800** (the largest, and the one that bounds `f`), best arm
vs A0, median of 3:

| | rule | what follows |
|---|---|---|
| **LATENCY-CONFIRMED** | best arm ≤ **0.50x** A0 (≥2.0x faster) | E3 §4.6's reading is proven; `f` and the `1000/f` ceiling are re-derived at every shape and `SPEED_LEDGER` §13.4 is upgraded from *corroborated* to *measured* |
| **MIXED** | between 0.50x and 0.90x | report the achieved floor **in GB/s of touched bytes**; the remaining distance is the load path |
| **BANDWIDTH-BOUND** | best arm ≥ **0.90x** A0 | E3 §4.6's reading is **withdrawn**; the lever is an **int8 KV cache** (the cache is `[L][maxseq][NKV*HD]` **fp32** and has never been touched), and that becomes E5 |

**Reported regardless of the label:** the new `f` and `1000/f` at `T10` @300 and @800, and the new
end-to-end tok/s — including the fact that it moves by at most a few percent, which does not change
under any outcome.

---

## 6. What is NOT claimed

Nothing about quality beyond G2's parity. Nothing about whether the GQA re-reads reach DRAM or L2 —
E3 did not measure it and neither does this. Nothing about contexts beyond 800. Nothing about shapes
other than the six E3 measured.

---

## 7. Reproduction

```
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm
python e3_bench.py --weights D:/_ktmp/e3/T10_th.bin --bench 800 --reps 3 --extra --attn avx4
donor_engine.exe --weights D:/_ktmp/e1/qwen25-05b_tqh.bin --bpb D:/_ktmp/e1/ids_qwen25-05b_tqh.bin --attn avx4
```
