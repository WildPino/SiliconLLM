# E4 — the attention reduction was latency-bound, and fixing it moves the ceiling that E3 said was the wall

**Brief:** `briefs/BRIEF_E4_ATTENTION_ACCUMULATORS.md` — §1–§7 at `25a6d29` (before any arm existed),
§8 at `6115642` (run 1's `VOID`, before run 2), §9 at `b3e94a7` (before the interleaved measurement).
**Predecessor:** `probes/E3_ENGINE_AT_TARGET_SCALE.md` §4.6, which marked its own reading
*corroborated, not proven* and named this experiment. `INDEX` §4 carried it as **item 0**.

---

## 0. Verdict

**`LATENCY-CONFIRMED`.** Run 1 was `VOID`. Run 2 supplied the arms; run 3 (interleaved) supplied
the label.

- The `Q·K` inner product was **6.53× below its own memory limit**, and is not any more. Isolated
  from the rest of the organ by a planted control that was *proven at a point it could not be fitted
  to*, the loop went **14.647 → 2.242 ms/token** at a 10 B shape and 800 context, and from
  **5.4 → 35.1 GB/s of unique K bytes** — from far below this machine's DRAM to sitting on it.
- **E3's hardest negative result is overturned.** E3: *"at 800 context a 10 B shape cannot pass
  38.3 tok/s even with a free weight path."* The ceiling `1000/f` is now **78.5 tok/s**.
  **50 tok/s at 800 context is a weight-side problem again**, which E3 had established it was not.
- **The 50 tok/s active-weight budget at 800 context goes from 0 to 259 M**, and at 300 context from
  359 M to 523 M within the same sweep (+46%).
- It costs nothing: **|ΔBPB| = 3.03e-06**, ~1650× under σ_seed, on a real donor.
- **The lever named by E3 is superseded before being run.** An int8 KV cache is now worth **~1.16×**
  on the organ. The floor is `R` — softmax + the `A·V` loop — which is **81.4% of the best arm's
  attention organ** and which nothing here touched.
- **End-to-end tok/s moves +4.2%**, and that was never the point (§1).

---

## 1. What this asked, and what it was not for

E3 measured the `attention` organ at 3.72–4.27 cycles per FMA per thread across twelve points (±7%),
invariant to shape, context, KV size over a 16× range, GQA and vocabulary — the signature of a serial
FP reduction that `clang` may not reassociate without `-ffast-math` (forbidden since Phase 35). Two
explanations survived: **latency** or **bandwidth**. E4 discriminates them.

The brief said in advance that this is not a tok/s experiment: at `T10` @300 the whole organ is 9.8 of
a ~310 ms token, so deleting it entirely buys ~3%. What it is for is **`f = rope + attention +
norm+glue`**, the cost no weight format can remove, and `1000/f`, the ceiling on every future
sparse/MoE/ternary win. At `T10` @800, attention was **97.3% of `f`**.

---

## 2. Gates

### 2.1 G2 — parity, on a real donor, and it is free

`qwen25-05b_tqh.bin` + the pinned `ids_qwen25-05b_tqh.bin` (24×512, 51 870 scored bytes), `--bpb`.

| arm | BPB | ΔBPB vs `serial` | |
|---|---|---|---|
| `serial` | 4.635627680 | — | |
| `serial2` | 4.635627680 | +0.000e+00 | **bit-identical** |
| `ilp4` | 4.635623126 | −4.554e-06 | 1098× under σ_seed |
| `avx1` | 4.635623934 | −3.746e-06 | 1335× under σ_seed |
| `avx4` | 4.635624653 | −3.027e-06 | **1652× under σ_seed** |

For scale: E1 traced the whole engine-vs-PyTorch delta to `1.53e-05` and attributed it to
accumulation order. These are smaller than that. **Reassociating this reduction is free at the
deliverable**, which is the only place quality is defined.

### 2.2 G1 (run 1) — **FAILED**, and the threshold was what was wrong

`serial2` computes the dot product twice and returns `(d1+d2)*0.5f`, exact in IEEE — **2× the work,
bit-identical output** (sha256-equal over 30.4 M logits). Required ≥1.7×; measured **1.601× @300 and
1.602× @800**.

The brief wrote 1.7 as though the dot loop were the whole organ. It is not. Solving the two equations
gives the decomposition, and the two context lengths agree:

| | @300 | @800 |
|---|---|---|
| `serial` = `X + R` | 9.025 | 24.463 |
| `serial2` = `2X + R` | 14.895 | 39.110 |
| **dot loop `X`** | 5.870 | **14.647** |
| **the rest `R`** = softmax + `A·V` | 3.155 | **9.816** |

That reasoning was **not allowed to rescue the run** — it is a re-reading of a gate after seeing it
fail. Run 1 keeps the label `VOID` and the gate was replaced by one that *tests* the model.

### 2.3 G1′ (run 2) — **PASS**, at a point that could not be fitted

`X` and `R` are solved from the 1× and 2× times, so they fit those two by construction and prove
nothing. `serial3` runs the loop three times and returns `d1+(d2−d1)+(d3−d1)` (bit-identical,
verified by sha256): it must land on `3X + R`, and it cannot be fitted there.

| point | predicted `3X+R` | brief §8.3, fixed before the arm existed | **measured** | error |
|---|---|---|---|---|
| `T10` @300 | 20.765 | 20.083 | **20.370** | −1.90% (vs brief **+1.43%**) |
| `T10` @800 | 53.757 | 53.482 | **53.458** | −0.56% (vs brief **−0.04%**) |
| `S05` @300 | 2.177 | — | **2.124** | −2.43% |
| `S05` @800 | 5.622 | — | **5.501** | −2.15% |

Four points, two shapes, two head dimensions (`HD` 64 and 128), tolerance ±3%. **`X` and `R` are
measurements from here on**, and every mechanism claim below rests on that and not on an assumption.

### 2.4 G4 / G4′ — **MALFORMED**, and the reason is worth more than the gate

G4 asked that `--attn serial` reproduce E3's published table. It failed by +3.4% to +7.0%. My first
diagnosis (§8.2 of the brief) blamed the KV address arithmetic, which the refactor had hoisted out of
the `t` loop. **That diagnosis is withdrawn**, priced directly:

| | `serial_e3` − `serial` |
|---|---|
| `T10` @300 | +0.139 ms (1.5%) |
| `T10` @800 | **−0.485 ms (−2.0%)** |
| `S05` @300 / @800 | +0.018 / +0.012 ms |

±2%, **sign not even consistent** — it cannot produce a uniform +5%. So instead of arguing, E3's own
engine was rebuilt from `d7977c8^` (its `--logits` output sha256-identical to `serial_e3`) and timed
**in the same session**:

| point | E3 published | **E3's own binary, now** | `serial_e3`, now | old vs new | **old vs its own published number** |
|---|---|---|---|---|---|
| `T10` @300 | 3.090 | **3.040** | 3.030 | −0.33% | **−1.6%** |
| `T10` @800 | 2.960 | **2.880** | 2.880 | +0.00% | **−2.7%** |
| `S05` @300 | 55.700 | **53.370** | 53.400 | +0.06% | **−4.2%** |
| `S05` @800 | 49.280 | **48.360** | 48.390 | +0.06% | **−1.9%** |

1. **`serial_e3` *is* E3's engine** — four points, worst disagreement 0.33%, three of them ≤0.06%,
   on top of bit-identity. G4′'s purpose is satisfied.
2. **E3's own binary cannot reproduce E3's own published table.** And run 2 read `serial_e3` at
   `T10` @300 as **3.240** two hours before this table read the same code as **3.030** — a **6.5%
   swing**, while the within-run IQR stayed at **0.005**.

> **Law: a within-run IQR is not a reproducibility interval.** Within-sweep dispersion on this
> machine is 0.000–0.015 tok/s; **between-sweep dispersion is 5–10%**. Any gate comparing a
> measurement to a number from another session is measuring the calendar. The only valid form is to
> **re-measure the reference in the same sweep**.
>
> Third instance in this programme: E3 §2.5's reference moved 2.9% in twenty minutes; the ledger's
> original 18.6 G-w/s was contended by 35%. It is a rule now, not an observation.

**Everything in §3 and §4 below is a ratio taken inside one sweep.** That is why the probe survives
this, and it is the only reason.

### 2.5 §5's threshold sat inside the dispersion, so run 3 was interleaved

Run 2 gave `avx4 / serial_e3` = **0.503×** at `T10` @800 against a `LATENCY-CONFIRMED` line at
**≤0.50×** — 0.6% apart, from arms measured across ~100 minutes.

> **Law: a threshold placed inside the instrument's dispersion cannot decide anything.**

§9.2 fixed the remedy before running it, with **every threshold unchanged**: `serial_e3` and `avx4`
interleaved **A/B/A/B/A/B**, one rep each, ratio taken per adjacent pair, median of three decides,
spread reported.

| pair | `serial_e3` | `avx4` | ratio |
|---|---|---|---|
| 1 | 27.287 | 13.585 | 0.4979 |
| 2 | 27.698 | 12.964 | 0.4680 |
| 3 | 25.054 | 12.215 | 0.4875 |

**median 0.4875, spread 2.98%, all three below 0.50 → `LATENCY-CONFIRMED`.**

The `serial_e3` arm itself moved **10.55%** across those same three pairs (25.054 → 27.698). The
ratio moved 2.98%. That single comparison is the whole case for interleaving, measured.

---

## 3. Arms — run 2, all seven under one binary, one sweep

`--bench 800`, `T10` (10.603 B active weights/token), 6 threads, 3 reps, median.

| arm | tok/s | IQR | **attention ms** | vs `serial_e3` | `f` ms | **1000/f** |
|---|---|---|---|---|---|---|
| `serial_e3` | 3.100 | 0.005 | **23.978** | 1.000× | 24.678 | **40.5** |
| `serial` | 3.110 | 0.005 | **24.463** | 1.020× | 25.141 | 39.8 |
| `serial2` (2× the dot loop) | 2.970 | 0.000 | **39.110** | 1.631× | 39.783 | 25.1 |
| `serial3` (3× the dot loop) | 2.860 | 0.005 | **53.458** | 2.229× | 54.132 | 18.5 |
| `ilp4` | 3.190 | 0.000 | **16.148** | 0.673× | 16.827 | 59.4 |
| `avx1` | 3.220 | 0.010 | **13.238** | 0.552× | 13.919 | 71.8 |
| **`avx4`** | **3.230** | 0.005 | **12.058** | **0.503×** | **12.735** | **78.5** |

`--bench 300`, `T10`: `serial_e3` 9.164 → `ilp4` 5.925 → `avx1` 4.810 → **`avx4` 4.540 (0.495×)**,
`1000/f` **101.6 → 191.7**.

`S05` (`HD`=64, 0.494 B): @300 1.033 → **0.630 (0.610×)**; @800 2.644 → **1.593 (0.602×)**. The same
sign at a 21× smaller shape and the other head dimension — E2's law holds (*a small donor gives the
sign, not the magnitude*): the sign replicates, the magnitude does not (0.60× vs 0.50×).

---

## 4. What the numbers say

### 4.1 ILP and SIMD width, separated — and it was mostly width

Using the **measured** `R` = 9.816 ms to subtract the part no arm touched (`T10` @800):

| arm | organ ms | **dot loop** = organ − `R` | speedup on the dot loop |
|---|---|---|---|
| `serial` | 24.463 | 14.647 | 1.00× |
| `ilp4` — 4 scalar chains, no SIMD | 16.148 | 6.332 | **2.31×** |
| `avx1` — 8 lanes, one chain | 13.238 | 3.422 | **4.28×** |
| `avx4` — 8 lanes × 4 chains | 12.058 | **2.242** | **6.53×** |

Breaking the dependency chain **without** vectorising already buys 2.31×, which by itself falsifies
the bandwidth explanation: no change in bytes moved, 2.31× less time. Vectorising alone buys 4.28×.
Doing both buys 6.53×, so the last 1.53× is the chain depth that 8 lanes did not already hide.

### 4.2 The loop was 6.5× below its own memory limit, and now sits on it

Both byte conventions, never mixed (`feedback_charged_vs_moved_bytes`). `T10` @800: **K touched**
315.0 MB/token (GQA re-reads included), **K unique** 78.7 MB/token.

| dot loop | ms | unique GB/s | touched GB/s |
|---|---|---|---|
| `serial` | 14.647 | **5.4** | 21.5 |
| `avx4` | 2.242 | **35.1** | 140.5 |

35 GB/s of unique bytes is this machine's DRAM read rate. 140 GB/s of touched bytes is not achievable
from DRAM, so **the GQA re-reads are being served by cache** — E3 explicitly declined to claim this
and it now falls out of the arithmetic. The loop is bandwidth-bound *now*; it was **6.5× away from
being so** before, which is what latency-bound means.

### 4.3 The brief's own floor prediction was wrong, in the direction this programme's law predicts

§3 predicted `avx4` would land at **5–8 ms** on the organ because "315 MB at 50 GB/s is 6.3 ms". The
dot loop measured **2.242 ms** — *faster than the predicted floor*. The prediction charged **touched**
bytes to DRAM when only **unique** bytes reach it. A ceiling is a denominator, and that is where the
wrong unit hid. The organ predictions (`ilp4` 7–11, `avx1` 6–10, `avx4` 5–8) all missed high
(16.148 / 13.238 / 12.058) for the same reason G1's threshold missed: they treated the organ as the
dot loop.

### 4.4 `R` is the new floor, and it retires the lever E3 named

`R` = softmax + the `A·V` loop = **9.816 ms**, untouched by every arm, and now **81.4%** of `avx4`'s
organ. Consequence, arithmetic:

> An **int8 KV cache** cuts the dot loop's unique bytes 4×: 2.242 → ~0.6 ms, organ 12.058 → ~10.4 ms
> = **~1.16×**. E3 §4.6 named it as the next lever if the loop came back bandwidth-bound. It did come
> back bandwidth-bound — **and the lever is worth 16%.**

**`R`'s composition is measured only as a total.** Decomposing it is owed (§7), not claimed here. What
is claimed is that it is now the binding constraint on `f`, and `f` is 97% attention.

### 4.5 `f`, the ceiling, and the budget

Within one sweep, `T10`, `serial_e3` → `avx4`:

| | @300 | @800 |
|---|---|---|
| `f` ms | 9.846 → **5.216** | 24.678 → **12.735** |
| **ceiling `1000/f`** | 101.6 → **191.7 tok/s** | **40.5 → 78.5 tok/s** |
| `r_w` (weight organs, unchanged by design) | 35.40 → 35.38 | 35.54 → 35.69 |
| **budget at 50 tok/s** = `r_w × (20 − f)` | 359 M → **523 M** | **0 → 259 M** |

The budget rows carry the ±5% between-sweep dispersion of §2.4 in their absolute value; the **ratios**
(+46% at 300, 0→259 M at 800) are within-sweep and do not.

### 4.6 The goal

E3 closed with: *"at 800 context a 10 B shape cannot pass 38.3 tok/s even with a free weight path —
50 tok/s at 800 context is unreachable by any weight-side work whatsoever."*

That sentence is now **false**, and it was made false by 40 lines of C that change no weight, no
format and no arithmetic anyone can measure at the deliverable (ΔBPB 3e-06):

| | E3 | **E4** |
|---|---|---|
| ceiling at 800 context, 10 B | 38.3 tok/s | **78.5 tok/s** |
| 50 tok/s at 800 context | impossible | **a weight-side problem again** |
| active-weight budget @800 | 0 | **259 M** |

End-to-end the engine went **3.100 → 3.230 tok/s (+4.2%)** at `T10` @800. Both numbers are true and
the second is the unimportant one — the dense path is still ~16× from 50 tok/s, and it is still the
weights. What changed is that the **non-weight** wall stopped being the thing in front of it.

### 4.7 The twelve-donor screen, re-priced — `f` had eaten the budget

E3's screen fitted `f = A·L·NH·HD·pos + B·L·D` on its twelve points. E4 changed `A` by **2.04×**
(`3.09445e-07 → 1.51461e-07` ms per FMA-unit), so the screen was **re-measured, not rescaled**: all
six shapes under `--attn avx4` in one sweep, with `f` and `r_w` both taken from it
(`results/e4/shapes_avx4.json`, `speed/e3_budget_by_shape.py --arms`).

Residuals first, because they are the licence: **0.3–0.8% at the 7–10 B shapes that decide**, −0.2%
to +7.8% at 1.5–3 B, and **−23% at `S05`** — with attention halved, the two-term model no longer
fits the small end. No donor in the screen is below 1.5 B, so that residual prices nothing, but the
fit is now unfit for a shape it used to fit.

| donor, head as % of budget | @300 was → is | @800 was → is |
|---|---|---|
| **Mistral-7B-v0.3** (smallest vocabulary on disk) | 31% → **23%** | **141% → 33%** |
| OLMo-2-7B | 94% → 71% | **432% → 102%** |
| **Qwen3-8B** | 152% → **110%** | **2763% → 169%** |
| Phi-3-mini | 21% → 17% | 44% → 22% |
| Qwen3-1.7B | 59% → 53% | 81% → 60% |
| Qwen2.5-1.5B | 42% → 39% | 53% → 42% |

E3's headline at 800 context — *"the screen stops being about the head at all; every 7–8 B dense
donor is over the line regardless of tokenizer, and Mistral-7B, the smallest vocabulary on disk, is
at 141%"* — **is withdrawn. It was about `f`.**

What survives is **Qwen3-8B**, which still breaks at 110% @300 and 169% @800 for the reason it always
did: a 151,936-token vocabulary on a 4096-wide model is a 622 M head, charged in full every token.
**The head is set by the tokenizer, not the model** — the one claim that has now survived three
re-pricings.

---

## 5. The label

Brief §9.2, verbatim, on the interleaved median: `0.4875 ≤ 0.50` → **`LATENCY-CONFIRMED`**.
Run 1: **`VOID`** (§2.2, §2.4).

---

## 6. Departures from the brief

1. **Run 1 is `VOID`** and both its failing gates are reported in full rather than re-run silently.
2. **G1 was replaced, not reinterpreted** (§2.2 → §2.3). The replacement is a test the original was
   not: a third point that cannot be fitted, with its target published first.
3. **§8.2's diagnosis is withdrawn** (§2.4). I blamed the address arithmetic; priced, it is ±2% with
   inconsistent sign. The real cause was the machine, and finding that required running E3's own
   binary rather than reasoning about mine.
4. **G4/G4′ are recorded `MALFORMED`, not failed** — their reference is not stable enough to be a
   reference (§2.4). This is the one place in E4 where a criterion is set aside, and it is set aside
   by a measurement that falsifies its premise, with the replacement passing at 0.06%.
5. **§5's threshold could not decide** and run 3 was added to measure the same ratio properly
   (§2.5). No threshold changed.
6. **§3's predicted floor was wrong** and is reported as wrong (§4.3), with the reason.
7. **Not measured, not claimed:** the composition of `R`; anything beyond 800 context; any shape
   other than `T10` and `S05` under these arms; whether the `A·V` loop has the same headroom (it
   accumulates over `i` and was excluded by design).

---

## 7. What this closes and what it owes

**Closes.** E3 §4.6's *corroborated, not proven* reading, now proven by a 2.31× that moves no bytes.
E3's 38.3 tok/s ceiling at 800 context. The int8-KV lever, retired at ~1.16× before being built. The
question of whether GQA re-reads reach DRAM — they do not (§4.2).

**Owes.**
1. ~~**Decompose `R`.**~~ **Done — E5, `probes/E5_DECOMPOSE_R.md`, verdict `OVERHEAD-DOMINATED`.**
   `R` = `S` 25.3% + `Y` 20.6% + `P` 54.0% at `T10` @800; on this probe's own `R` = 9.816 that is
   **`S` = 2.485, `Y` = 2.020, `P` = 5.301 ms**. The third component this probe did not name is the
   **largest** one. Two corrections back to here: E5's `X` = 2.253 ms reproduces §4's 2.242 to
   **0.5%** by a method sharing no arm with it — the strongest check this probe's headline number
   has had; and the softmax, which §7 assumed was the cheap half of `R`, is 25.3% of it and came in
   48% above the band E5 pre-registered. `P` is **not** the OpenMP fork: that was priced directly at
   **1.4% of `P`**. What `P` is remains unmeasured, and is now item 0 of the INDEX.
2. ~~Re-run E3's twelve-donor budget screen against the new `f`.~~ **Done** (§4.7). `A` was wrong by
   2.04×; the screen was re-measured rather than rescaled, and `e3_budget_by_shape.py` now derives
   `r_w` from whichever sweep it is given rather than from a hardcoded table.
3. ~~`f` beyond 800 tokens of context; still nothing measured bounds it.~~ **Done.** E7 §8.2
   measured it on a real 7.07 B donor: **linear, 0.0319 ms per token of actual context**, the
   300→800 and 800→1600 intervals agreeing to **0.2%**, no knee. Cells admitted by the
   `ffn`-invariance witness. Note also E7 §9: the readings above are the **`avx4`** arm, but the
   engine's *default* is `serial` (`donor_engine.c:116`) — E4's runners pass `--attn` explicitly,
   so this table is unaffected, but `e3_bench.py` does not.
4. ~~Every absolute tok/s in `SPEED_LEDGER` and E3 carries a ±5% between-sweep band that its
   published IQR does not show (§2.4).~~ **Done.** `SPEED_LEDGER.md` carries the band as a banner
   binding the whole file plus a line at each headline absolute (§10, §12.3, §13.5 — the last is
   marked withdrawn rather than banded); `probes/E3_ENGINE_AT_TARGET_SCALE.md` gained an amendment
   banner carrying both the band and this probe's withdrawal of its ceiling. The **ratios** are
   unaffected and are named as such in both places. No re-run.

---

## 8. Reproduction

```
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm
bash e4_run2.sh          # all seven arms, one binary, both shapes, both contexts
bash e4_interleave.sh    # A/B/A/B/A/B at T10 @800
bash e4_g4prime.sh <dir> # E3's own binary, rebuilt from d7977c8^, in this session
python e4_analyse2.py    # every table in §2-§4
```

Results: `benchmarks/donor_adaptation/results/e4/` — `parity.txt`, `arms.json` (run 1, `VOID`),
`arms2.json`, `g4prime.json`, `interleave.json`, `analysis.txt`, `analysis2.txt`.
`e4_analyse.py` is kept byte-for-byte as it was when it issued run 1's `VOID`.
