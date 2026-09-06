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

---

## 8. AMENDED after run 1 — **run 1 is `VOID`**, two gates failed, and both are my errors

Appended 2026-09-06, **before run 2 exists**. Run 1's arms are in `results/e4/arms.json` and its
analysis in `results/e4/analysis.txt`; nothing below deletes them.

### 8.1 G1 failed at its threshold, and the threshold is what was wrong

Required ≥1.7x. Measured **1.601x @300 and 1.602x @800** — the same number twice, from two
independent context lengths.

**Why 1.7 was the wrong number.** `serial2` doubles the **dot loop**, not the organ. §4 wrote the
threshold as though the dot loop were the whole organ. Solve the two equations instead:

| | @300 | @800 |
|---|---|---|
| `serial` = `X + R` | 9.123 | 24.270 |
| `serial2` = `2X + R` | 14.603 | 38.876 |
| **dot loop `X`** | **5.480** | **14.606** |
| **the rest `R`** (softmax `expf` + the A·V loop) | **3.643** | **9.664** |
| `R` as a share of the organ | **39.9%** | **39.8%** |

Two context lengths, 2.7x apart in organ time, give the same 39.9% split. So the instrument fired
exactly as its mechanism predicts; **the arithmetic in my threshold did not include `R`.**

**That reasoning is not allowed to rescue the run.** It is a re-reading of a failed gate after seeing
it fail, which is the move E3 §2.5 permits only once and only with the ground truth failing beside
it. So the gate is not re-read: it is **replaced by one that tests the model instead of asserting
it** (§8.3), and run 1 keeps the label `VOID`.

### 8.2 G4 failed because I moved the baseline, and this one is a real defect

Required: `--attn serial` reproduces E3 within its IQR (which is **0.005 and 0.000 tok/s** at `T10`).

| point | E3 | run 1 `serial` | |
|---|---|---|---|
| `T10` @300 | 3.090 tok/s, attention 9.862 | 3.240, **9.123** | **+4.9% tok/s, −7.5% attention** |
| `T10` @800 | 2.960, 25.385 | 3.090, **24.270** | +4.4%, −4.4% |
| `S05` @300 | 55.700, 1.096 | 57.580, 1.033 | +3.4%, −5.7% |
| `S05` @800 | 49.280, 2.847 | 52.710, 2.642 | +7.0%, −7.2% |

**Cause, found by reading my own diff.** Extracting the dot product into a function also hoisted the
KV address out of the `t` loop: E3 computes `s->kcache + ((l*maxseq + t)*KVO) + kvh*HD` at every
position, run 1 computes `kbase` once per head and walks `kbase + t*KVO`. That is strength reduction
of the address arithmetic — a **second, unannounced change** riding inside the arm labelled
"unchanged". It is worth 4–7% on its own, bit-identically, and it means **no ratio in run 1 is
against E3's baseline.**

This is the same failure as E3 §2.3 (a configuration named in prose but not in a flag), one level
down: *an arm named "unchanged" must be unchanged, and the way to know is to run it against the
artifact it claims to reproduce* — which is exactly what G4 is for, and G4 caught it.

### 8.3 Run 2 — fixed here, before it exists

Two arms are added; **no threshold, prediction or decision rule from §3–§5 is altered**, and the
§5 rule will be applied against the restored baseline.

| new arm | what it is | what it is for |
|---|---|---|
| `serial_e3` | the dot loop **and** E3's exact per-position address arithmetic | the true baseline; every ratio is re-taken against it |
| `serial3` | the dot loop **three** times, `d1 + (d2−d1) + (d3−d1)` | turns G1 from an assertion into a **test** |

**G1′ — the planted control becomes a falsifiable model, not a threshold.** `X` and `R` above were
solved from two points, so they fit those two points by construction. A third point cannot be fitted:
with `X`, `R` already fixed, `serial3` must land on `3X + R`.

| | predicted `attention` ms, stated before the arm exists |
|---|---|
| `T10` @300 | **20.083** |
| `T10` @800 | **53.482** |

**Required: within ±3% of those, at both context lengths, and `--logits` bit-identical to `serial`.**
If it lands there, the organ is linear in dot-loop count, `X` and `R` are *measured* rather than
assumed, and the instrument is proven on a known positive at a point it was not fitted to. If it does
not, the linear model is wrong, §8.1's explanation of G1 collapses with it, and **E4 stays `VOID`**.

**G4′ — unchanged in force, restated in target.** `serial_e3` must reproduce E3's `9.862` @300 and
`25.385` @800 within E3's dispersion. Additionally reported, because run 1 makes it measurable:
`serial_e3 − serial` is the **price of the per-position address arithmetic alone**, predicted
**≈0.74 ms @300** and **≈1.12 ms @800** from run 1's difference.

**What run 1's arms are allowed to be used for in the meantime:** nothing that carries a label. They
are listed in the probe as measured, with `VOID` on them, exactly as E3 run 1 was.

---

## 9. AMENDED after run 2 — G4' was a malformed gate, and §5's threshold sits inside the noise

Appended 2026-09-06 **before the interleaved measurement of §9.2 exists**.

### 9.1 G4' cannot be failed or passed, because its reference is not stable

G4/G4' compare a number measured now against a number E3 published yesterday. E3's own binary was
rebuilt from `d7977c8^` (its `--logits` output is sha256-identical to `serial_e3`) and timed **in the
same session** as `serial_e3`:

| point | E3 published | **E3's own binary, now** | `serial_e3`, now | old vs new | **old vs its own published number** |
|---|---|---|---|---|---|
| `T10` @300 | 3.090 | **3.040** | 3.030 | −0.33% | **−1.6%** |
| `T10` @800 | 2.960 | **2.880** | 2.880 | +0.00% | **−2.7%** |
| `S05` @300 | 55.700 | **53.370** | 53.400 | +0.06% | **−4.2%** |
| `S05` @800 | 49.280 | **48.360** | 48.390 | +0.06% | **−1.9%** |

Two facts, and they point the same way:

1. **`serial_e3` is E3's engine.** Four points, maximum disagreement **0.33%**, three of them ≤0.06%
   — measured, on top of the bit-identity. G4''s *purpose* is satisfied.
2. **E3's own binary cannot reproduce E3's own published table**, by −1.6% to −4.2%. And run 2
   measured `serial_e3` at `T10` @300 as 3.240 two hours before this table measured it as 3.030 —
   a **6.5% swing in the same code on the same machine**, while the within-run IQR stayed at 0.005.

> **Law: a within-run IQR is not a reproducibility interval.** This machine's within-sweep dispersion
> is 0.000–0.015 tok/s and its **between-sweep dispersion is 5–7%**, an order of magnitude larger. Any
> gate that compares a measurement to a number from another session is measuring the calendar. The
> only valid form is **re-measure the reference in the same sweep** — which is why run 2 re-timed all
> seven arms under one binary, and why that decision is the reason the probe survives.
>
> Third instance in this programme: E3 §2.5's reference moved 2.9% in twenty minutes, T2's ledger
> timing was contended by 35%. It is now a standing rule, not an observation.

**Consequence for E4:** G4' is recorded as **MALFORMED**, replaced by the same-session comparison
above, which passes at 0.06%. No result in E4 depends on any cross-session number.

### 9.2 §5's threshold is inside the instrument's dispersion, so it cannot decide

Run 2, `T10` @800, best arm `avx4` vs `serial_e3`: **12.058 / 23.978 = 0.503x**. §5 draws
`LATENCY-CONFIRMED` at **≤0.50x**. The two differ by **0.6%**, and the seven arms of that block were
measured sequentially across ~100 minutes of a machine whose between-point drift is several percent —
the same code read 23.978, 24.270 and 24.463 across three separate measurements (±1%).

> **Law: a threshold placed inside the instrument's dispersion cannot decide anything.** §5 wrote
> 0.50 without knowing the dispersion, which was not measured until §9.1.

**Fixed before the measurement:** `serial_e3` and `avx4` are re-timed at `T10` @800
**interleaved A/B/A/B/A/B**, one rep each, so every pair straddles the same few minutes and the drift
is differenced out. The ratio is taken **per adjacent pair** and the median of the three pairs decides:

| ratio, median of 3 interleaved pairs | label |
|---|---|
| ≤ 0.50x | **LATENCY-CONFIRMED** |
| 0.50x – 0.90x | **MIXED** |
| ≥ 0.90x | **BANDWIDTH-BOUND** |

unchanged from §5 in every threshold. **The spread of the three pairs is reported**, and if it
straddles 0.50 the label is `MIXED` and the probe says the rule could not separate them — that is the
outcome §9.2 exists to make sayable rather than to avoid.

### 9.3 What does not depend on any of this

G1' (§8.3) passed at four points against predictions fixed before the arm existed — `+1.43%` and
`−0.04%` against the brief's own pre-registered `T10` numbers — and every one of those is a *ratio
inside one sweep*. G2's parity is deterministic. **`X` and `R` are measurements**, and the reading
they give does not move with the machine's mood:

- the dot loop went **14.647 → 2.242 ms, 6.53x**, and with it from **5.4 → 35.1 GB/s of unique K
  bytes** — from 6.5x below this machine's DRAM limit to sitting on it. That is latency, and it is
  no longer latency.
- **`R` (softmax + A·V) is 81.4% of the best arm's organ** and no arm here touched it. An int8 K
  cache would now cut the dot loop's unique bytes 4x, worth **~1.16x on the organ** — so E3 §4.6's
  named next step is **superseded before it is run**, and the lever is `R`. `R`'s composition is
  measured only as a total; decomposing it is E4's owed follow-on, not a claim here.
