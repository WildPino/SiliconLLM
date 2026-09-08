# E7 — a real 7.6 B donor, end to end, and the synthetic proxy finally checked against it

**Status: CLOSED, `REAL-WEIGHTS-CONFIRMED`.** Brief: `briefs/BRIEF_E7_REAL_LARGE_DONOR.md`
(§1–6 pushed at `cb03580` before the export ran; §7, the memory amendment, at `46496db`;
§9, the extension, at `8252031`; §10, the attention-kernel amendment, at `9103c26` — each before
the run it governs).
**Every gate passed.** Extended twice since: §8 here closes E4's owed item 3, and §9 records a
defect in which kernel the engine runs by default.

| | |
|---|---|
| donor | `Qwen/Qwen2.5-Coder-7B` — **7.617 B parameters, ≈ 7.072 B active per token** |
| shape | D 3584, F 18944, L 28, heads 28/4, head_dim 128, vocab 152064, untied |
| fp32 arm | `qwen25-coder7b_f32.bin`, **30,462,466,100 bytes**, sha256 `6f17d9c372ee0fcf…`, `--fold none --load-dtype bfloat16` |
| packed arm | `qwen25-coder7b_p.bin`, **5,722,636,340 bytes**, sha256 `d31c5047cb331f15…`, `--quant packed --rule R0 --head-ternary --fold layers`, 197 matrices, mean ternary zero-fraction 0.3572 |
| data | `engine/results/e7/{parity.json, generate.json, bench.json}` |
| runner | `engine/e7_real7b.py` (`--stage parity | generate | bench`) |

---

## 1. The gap this closes, and the one it does not

Before E7 the programme's honest headline was: **"10 B at 50 tok/s" had a measured speed and no
measured model.** Every target-scale number in E3, E4 and E5 came from `T10`, a *synthetic shape
file* — right dimensions, untrained weights — and E6's demonstration that a donor generates text
stopped at 1.5 B.

E7 moves that to **7.07 B active weights of real trained parameters**, which is **67% of the
target size**. It does not reach 10 B: nothing bigger exists offline (`Qwen3-8B`,
`Qwen3-30B-A3B` and `Qwen3-Next-80B-A3B` are **1 MB config stubs**, not weights).

## 2. The model works at 7.6 B

| gate | test | result |
|---|---|---|
| **G-L** | the engine consumes the file exactly, header matches `config.json` | **PASS** — `layout OK: consumed exactly 30462466100 bytes`. **15× the largest artifact this runtime had ever loaded**, and the size class where E1 found a 32-bit `ftell` bug that made everything over 2 GB silently unloadable |
| **G-P** | `--logits` vs PyTorch fp32, 8 positions | **PASS** — rel L2 **1.070e-05**, top-1 **1.0000** |
| **G-G** | `--generate` greedy vs PyTorch greedy, E6's five frozen prompts × 32 tokens | **PASS — 160/160, no divergence** |
| **G-C** | planted control: the packed arm must not also pass | **PASS — 0/160** |

What the fp32 arm says, on our runtime, character for character with PyTorch:

    "The three laws of motion were formulated by"
      -> " Isaac Newton in his work Philosophiae Naturalis Principia Mathematica, first
          published on July 5, 1687. Newton used them to"

    "Water boils at"
      -> " 212 degF or 100 degC and ice melts at 32 degF or 0 degC. If the temperature of"

    "def fibonacci(n):"
      -> "\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:
          \n        return fibonacci(n"

## 3. The speed, against a band written down first

`--bench` on the packed arm, **3 repetitions per cell**, dispersion reported:

| cell | reps | median | spread |
|---|---|---|---|
| 300 tokens of context | 3 | **4.460 tok/s** | **0.45%** |
| 800 tokens of context | 3 | **4.580 tok/s** | **0.00%** |

**The brief predicted 4.66 tok/s and fixed the band at 4.0–5.4 before the donor was exported.**
Measured 4.460 — **−4.3%**, inside the band. The prediction was nothing but E3's delivered weight
rate carried onto a different shape, so what passed is the *weight-rate model*, on trained weights,
at a size it had never been tested at.

**A real 7 B donor runs 11.2× short of 50 tok/s** (`50 / 4.460`), on an engine whose attention
organ has already been made 6.5× faster (E4) and then fully decomposed (E5). E7's contribution to
the goal is therefore negative in the useful sense: it removes the possibility that the 3.1 tok/s
at `T10` was an artefact of synthetic weights.

## 4. The synthetic proxy was honest — checked, not assumed

Everything E3, E4 and E5 measured at target scale used untrained weights. That was defensible
(a matvec does not care what the numbers are) and it was never checked. Now it can be:

| | active weights/token | tok/s @300 | **delivered weight rate** |
|---|---|---|---|
| `T10`, synthetic 10.6 B (E3) | 10.603 B | 3.090 | **32.8 G-weights/s** |
| **Coder-7B, real 7.07 B (E7)** | 7.072 B | **4.460** | **31.5 G-weights/s** |

**−4.0%**, which is *inside* this programme's own reproducibility band for an absolute rate
(±5%, `SPEED_LEDGER` §13's law). The two are indistinguishable by the instrument that measured
them. **The synthetic shape files were a sound speed proxy**, and E3/E4/E5's target-scale numbers
survive the check. This is the first time that has been tested rather than argued.

## 5. What the ternary arm did, and what it does not prove

The packed arm agreed with PyTorch on **0 of 160** tokens and answers every prompt with the token
`ERCHANTABILITY`, repeated to the end:

    "The capital of France is" -> "ERCHANTABILITY <=>\nERCHANTABILITYmaleERCHANTABILITY..."

**This is not a clean statement that ternary damage grows with scale.** E6's arms were exported
under **R3** (calibrated) with `--fold layers`; E7's packed arm is **R0** (`bitlinear`, no
calibration) because §7 of the brief forbade a fold under the low-memory loader. R0 is the weakest
rule in the family and T2 measured the rule as worth **+3.309 → +1.260 BPB**. So the honest
reading is: **R0 at 7 B collapses completely**, and how much of that is the rule versus the scale
is a separate experiment nobody has run.

## 6. The exporter had to change, and the change was gated

The first export was **killed by the system for low memory**: `AutoModelForCausalLM(dtype=float32)`
on a bf16 checkpoint materialises ~30 GB before a byte is written. `qwen_export.py` gained
`--load-dtype {float32,bfloat16}`; under `bfloat16` the donor is held at its own checkpoint
precision and each tensor is widened to fp32 at write time, inside `w_fp32` and `quantize`, so the
rules still see fp32.

**Control, threshold fixed before it ran: export `Qwen2.5-0.5B --quant fp32 --fold none` under both
load dtypes and compare sha256.**

    float32  load -> ebed02f7ec8917d825d031e0ab00677617f54cc87e7cc5a9e1b9b0cd860d5740
    bfloat16 load -> ebed02f7ec8917d825d031e0ab00677617f54cc87e7cc5a9e1b9b0cd860d5740

Equal — and **equal to E1's own `qwen25-05b_f32.bin`**, byte for byte, 1,976,131,124 bytes. The
low-memory path is the same exporter, and it reproduces an E1 artifact exactly.

`--load-dtype bfloat16` **refuses** `--fold` and `--rule R3`, because folding and calibration are
arithmetic rather than storage. That is why §5's arm is R0/fold-none and why **E7 reports no BPB**.

## 7. A measurement note worth keeping — **REFUTED by §8.1, kept as written**

The 800-context cell is **faster** than the 300-context cell here (4.580 vs 4.460), the opposite of
`T10`, where E3 read 3.090 @300 and 2.960 @800. Two-point arithmetic on the wall times
(67.276 s and 174.723 s) gives a marginal cost of **214.9 ms/token** and a **fixed ~2.8 s inside
the timed region** — plausibly first-touch of a 5.7 GB weight array, amortising over more tokens.
At `T10` (48 layers × 32 heads) `f` grows fast enough to overwhelm that; here (28 × 28) it does
not. **This is a two-point fit, not a measurement**, and §8.1 tested it and **refuted it**. The
inversion is 2.7%, inside this programme's own ±5% band for an absolute rate; there was no
mechanism to model. Left standing verbatim because the brief called it a fit, not a result, and
because the cost of the error — one 20-minute third point — is the whole argument for saying so
out loud.

Note also that wall time per repetition was dominated by **loading**, not computing: after writing
a 30 GB file the page cache is cold and each `--bench` re-reads 5.7 GB. It does not touch the
number — the `BENCH` line times only the decode loop — but it is why a 11-minute measurement took
an hour.

## 8. Extension — the third point, and `f` beyond 800

Pre-registered at `briefs/BRIEF_E7_REAL_LARGE_DONOR.md` §9, pushed at `8252031` **before** the run.
Verdict at §9-bis (`9103c26`).

### 8.1 G-X1 — the fixed cost does not exist

`--bench 1600`, packed arm, 3 reps: **4.05 / 4.23 / 4.31 → median 4.230**, spread 6.1%.

| | |
|---|---|
| §7's fit predicted | **4.616 tok/s**, band 4.50–4.68 |
| the call fixed in the brief | *above 4.580 the fit survives, below it is refuted* |
| measured | **4.230** — below the threshold, below the band, **−8.4%** off the prediction |

**REFUTED.** There is no fixed cost inside the timed region, so no short `--bench` cell in the
SPEED_LEDGER reads low. The 300→800 inversion §7 modelled is **2.7%** — inside the ±5% band this
programme applies to every absolute rate. It was noise, and it should not have been modelled.

### 8.2 G-X2 — `f` is linear to 1600 tokens (E4's owed item 3, closed)

Profiled cells are admissible only where the **`ffn`-invariance witness** holds: the FFN organ
cannot depend on context length, so a cell whose `ffn` sits above the uncontended ~175–180 ms
plateau was measured under load and its attention reading is discarded. **Six of nine survived.**

| context | `ffn` (the witness) | **attention ms/token** |
|---|---|---|
| 300 | 176.935 | **4.932** |
| 800 | 177.832 / 175.730 | **13.092 / 12.713** |
| 1600 | 176.193 / 179.518 / 180.119 | **25.462 / 25.636 / 25.951** |

Discarded: `ffn` 237.124 / 228.602 / 213.422, which would have read attention as **8.031 / 6.428 /
16.067** and manufactured a knee between 300 and 800 that does not exist.

| interval | Δ attention | per token of `--bench N` | **per token of actual context** |
|---|---|---|---|
| 300 → 800 | 7.97 ms | 0.01594 | **0.0319 ms** |
| 800 → 1600 | 12.78 ms | 0.01597 | **0.0319 ms** |

`--bench N` averages the organ over positions 0…N−1, so mean context is N/2 and the third column
is the physical slope. **The two intervals agree to 0.2% across a 5.3× range in context: `f` is
linear to 1600 with no knee.** Nothing had bounded `f` above 800 before today.

### 8.3 An anomaly that was not one

E7's attention (12.90 ms @800) against E4's `T10` figure (12.058 @800) looks like a ~2.2× anomaly:
Coder-7B has 28 layers × 28 heads = **784 head-layers**, `T10` has 48 × 32 = **1536**, so E7 should
cost **0.510×** of `T10`, not the same. The comparison was wrong, not the engine — **E4's 12.058 is
the `avx4` arm and E7 ran the default, which §9 shows is `serial`.** Against E4's own `serial` row:

    predicted   24.463 ms x (784 / 1536)  =  12.49 ms
    measured    12.90 ms                             ->  +3.3%

An organ cost predicted across two different models and two different head counts, from head-layer
count alone, correct to 3.3%.

## 9. The engine's default attention kernel is not the one E4 won with

`donor_engine.c:116` reads `static int g_attn = ATTN_SERIAL;`. That is E4's **second-slowest** arm
(`T10` @800: `serial` **24.463 ms**, `serial_e3` 23.978, `avx4` **12.058**). E4's 6.53× win on the
`Q·K` dot loop is reachable only through an explicit `--attn avx4` and **was never made the
default**. E4's own runners pass `--attn` on every point, so E4's table is sound; `e3_bench.py`
defaults it to the empty string and never passes it, and neither did `e7_real7b.py`. **Every number
in §§1–8 above, and E3's `T10` baseline, is on `serial`.**

The arm transfers to a real donor. Profiled, witness-admissible:

| context | `serial` | `avx4` | ratio | E4's `T10` ratio |
|---|---|---|---|---|
| 300 | 4.932 | **2.336** | **0.474×** | — |
| 800 | 12.90 | **5.737** | **0.445×** | 0.493× |

**What it is worth, stated before the rate was measured** (brief §10, G-Y2): attention is **2.2% of
the token at 300 and 10.5% at 1600**; the FFN is **73–80%**. Halving a 2.2% organ cannot move an
11.2× gap. **§9 is a correctness note about which kernel ran, not a speed result.** E4 already
published the rate consequence at `T10` — `avx4` 3.230 vs `serial` 3.110 tok/s, **+3.9%**, inside
±5% — so §§13–16 of the SPEED_LEDGER need **no numeric correction**. What they need, and are owed,
is a line naming the kernel that produced them.

**Not proposed: changing the default.** `serial` is the arm every prior probe's baseline was taken
on; flipping `g_attn` would silently re-base E1–E7. The defect is documentation, and the fix is a
runner that passes `--attn` explicitly plus a ledger line that names the kernel — not an edit that
makes old numbers unreproducible.

## 10. The kernel's win is real, and no un-profiled instrument here can see it

Pre-registered as G-Y1 at brief §10 (`9103c26`) and G-Y1b at brief §11 (`db53f63`), each pushed
before the run it governs. Verdicts at brief §10-bis and §11-bis (`f1b81ad`).

### 10.1 G-Y1 — the null was called, and the other cell did not decide

`--bench {300, 1600} --attn avx4`, 3 reps, un-profiled:

| cell | `serial` | **`avx4`** | Δ | §10 predicted | band |
|---|---|---|---|---|---|
| 300 | 4.460 (spread 0.45%) | **4.450** (2.5%) | −0.22% | +1.2% | 4.35–4.70 ✅ |
| 1600 | 4.230 (spread **6.1%**) | **4.290** (1.2%) | +1.42% | +5.7% | 4.30–4.70 ❌ by 0.23% |

**300 is the null §10 called in advance**, and it lands where §10 said. **1600 did not decide:**
the `serial` baseline's own spread (6.1%) exceeds the effect (5.7%) and its maximum, 4.31, sits
*above* the `avx4` median. E4's law — *a threshold inside the instrument's dispersion cannot
decide* — disposes of it, and the 0.23% band miss refutes nothing.

### 10.2 G-Y1b — **VOID**, and pairing did not rescue it

Arms interleaved per repetition at 1600, E4's own method, because §13 fixes between-sweep
dispersion at 5–10% while ratios do not carry it:

| pair | `serial` | `avx4` | ratio |
|---|---|---|---|
| 1 | 4.17 | 3.96 | 0.950 |
| 2 | 4.04 | 4.42 | 1.094 |
| 3 | 4.11 | 4.00 | 0.973 |
| | spread 3.2% | spread **11.5%** | **median 0.973** |

§11 fixed the band at **1.00–1.09** and declared a paired median outside it VOID. **0.973 is
outside. G-Y1b is VOID and no number from it enters anything.**

**The cause is not named, because it was not measured.** One CPU-load sample immediately after the
run read 46%; three samples minutes later read 7 / 13 / 14%, with nothing of this programme's own
running either time. A single instantaneous sample is not a load measurement, and *"the machine
was busy"* is precisely the **plausible** artefact this programme fails on. What is certain is the
consequence: **within-arm dispersion reached 11.5%, roughly double the largest effect attention
can produce at 1600 (6.4%). The un-profiled 1600 cell cannot resolve this question at all.**

### 10.3 The defect that is worth more than the question: `--bench` has no contention witness

The `ffn`-invariance witness that made §8.2 trustworthy **exists only under `--profile`.** On the
plain `--bench` path there is nothing to check. **Every un-profiled rate this programme has
published rests on the operator's belief that the machine was idle, with no instrument able to
contradict it.** New, general, and now the first owed item (§11.4).

### 10.4 The consequential outcome is EXCLUDED, from data already in hand

§11's third named outcome was: *the profiler's organ split is not additive to the wall*, which
would read every table in `SPEED_LEDGER` §12–§14. Four witness-admissible 1600-context profiled
cells (`ffn` all inside the 176–181 plateau) settle it:

| cell | attention | `ffn` | TOTAL | wall | TOTAL vs wall | **TOTAL − attention − `ffn`** |
|---|---|---|---|---|---|---|
| `serial` A | 25.462 | 176.193 | 242.130 | 242.083 | **+0.019%** | 40.475 |
| `serial` B | 25.636 | 179.518 | 247.313 | 247.265 | **+0.019%** | 42.159 |
| `serial` C | 25.951 | 180.119 | 247.713 | 247.668 | **+0.018%** | 41.643 |
| **`avx4` D** | **11.315** | 178.458 | 231.005 | 230.936 | **+0.030%** | **41.232** |

**The organs sum to the wall to 0.03% in every cell**, and **the remainder after removing the two
organs that moved is arm-invariant** — `avx4`'s 41.232 sits *inside* the `serial` arm's own
40.475–42.159 scatter. Swapping the attention kernel changed the attention organ and nothing else.

On the two cells with the closest `ffn` (B and D): attention falls **14.32 ms**, `ffn` differs
**−1.06 ms**, so the wall should fall **15.38 ms** — it falls **16.33 ms**. **Agreement to
0.95 ms, 0.4% of the token.**

**So the `avx4` win does reach the wall, by roughly 6–7% at 1600 context.** That figure comes from
profiled walls, so **E7 publishes no `avx4` rate** (§12's law: profiled runs are for the organ
split only — the profiler once manufactured a 9.6% anomaly). The honest position: *the organ win
is measured, its arithmetic consequence is consistent to 0.4%, and no un-profiled instrument on
this machine can resolve 6–7% at the 1600 cell.*

### 10.5 What did not move

§9's accounting, fixed before any of this ran, is exactly what happened. Attention is **2.2% of
the token at 300 and 10.5% at 1600**; the FFN is **73–80%**; **E7 remains 11.2× short of 50
tok/s.** Two runs, ~80 minutes, and the goal moved by nothing — which both §10 and §11 said in
advance they would be worth, and is why being wrong in them was cheap.

## 11. The contention witness — `--bench` can now say whether the machine was quiet

§10.3 found the defect and owed the fix first. Pre-registered as §12 of the brief (`55bd8c6`),
**failed its own gate**, rebuilt and re-registered as §13 (`02baefa`), verdict §13-bis (`57ed5c8`).
**`WITNESS-CONFIRMED`.**

### 11.1 What it is

`donor_engine.c`: a second macro pair `TICW`/`TOCW`, used at **the FFN block and nowhere else**,
times **layer 0** on the plain `--bench` path and appends `ffn~ <x> ms/tok` to the `BENCH` line.
**On by default**; `--no-witness` restores the old hot path exactly.

**Why the FFN alone** — it is 73–80% of the token and *cannot depend on context length*, which is
the entire content of the `ffn`-invariance witness that made §8.2 and ledger §17 admissible.
**Why default-on** — a witness a runner must remember to pass is the same defect as §9, where
`--attn` is a flag, no runner passed it, and every E3 and E7 number is a `serial` reading as a
result. **Why the tilde** — `ffn~` is extrapolated ×`L` from one layer and is labelled so; against
the profiler's own all-layer `ffn` of 12.091 it reads 11.556 / 11.711 / 12.202, ~4%.

### 11.2 It failed the first time, at 48 timestamps per token

| | first build | **rebuilt** |
|---|---|---|
| timestamps per token | `2L` = 48 | **2** |
| `now_s()` | `QueryPerformanceFrequency` **+** `Counter`, every time | **`Counter` only** |
| **G-W1 paired median** (gate ≥ 0.990) | **0.9877 ❌** | **0.9996 ✅** |

The first build cost **1.2%** against a prediction of **0.008%** — **150× off**, so the mechanism
was not the one costed. §12 had already fixed the consequence in advance: *rejected as built, made
cheaper, not quietly kept.* The second prediction, **0.995–1.000**, was derived from the
**measured** 1.2% divided by the **structural** 24× rather than from a nanosecond count, and it
contains the result.

**A filtered median is recorded but NOT counted.** The failed build's own readings flag three of
ten pairs as off-plateau, and excluding them lifts its median to 0.9958 — above the gate. **Using
the instrument under test to select the data that acquits it is circular and inadmissible.** It is
written down so nobody later finds it and thinks it was hidden.

### 11.3 The three gates, as passed

| gate | test | result |
|---|---|---|
| **G-W1b** | 10 interleaved pairs, `--bench 300`, smallest/fastest shape (worst case for a per-token timer) | **PASS** — median **0.9996**, band 0.99–1.01, and inside the 0.995–1.000 predicted |
| **G-W2b** | `--logits` sha256 with and without the witness | **PASS** — identical, `d4960bc7…0424`, 4,861,952 bytes |
| **G-W3b** | **planted control**: it must FIRE under deliberate load | **PASS** — idle `ffn~` **11.631**, six busy threads **16.225**, **+34.3%** over the idle arm's maximum across ten reps |

**G-W3b is the one that decides whether the other two mean anything.** *An instrument must be shown
to fire on a known-positive before its nulls count* — and making it 24× cheaper did not cost it
its sensitivity.

Unasked for: the `ffn~` plateau spans **11.552–12.083, a 4.6% spread**, against rate spreads of
4.2% (witnessed) and 6.2% (unwitnessed). **The witness's dispersion is comparable to the rate's**,
which is what makes a reading off the plateau mean something.

### 11.4 What it does not do

**It validates nothing retroactively.** Every rate published before `02baefa` — E1 through E7,
`SPEED_LEDGER` §§12–19 — was taken without it and still rests on the operator's belief that the
machine was idle. **§10.2's VOID stands; G-Y1's 1600 cell stands undecided.** The witness makes the
*next* measurement checkable, not the last one.

**Declared cost:** `now_s()` is cheaper, so **profiled walls taken after `02baefa` are not
comparable to profiled walls taken before it.** Organ splits and slopes are unaffected — a uniform
per-`TIC` cost cancels in a difference, which is all §8.2 and §10.4 ever used.

**And it buys no speed.** E7 is **11.2× short of 50 tok/s** and §11 moves that by nothing.

## 12. Owed

1. **A real 10 B.** Still the open item. E7 reached 67% of the size; the rest needs a download.
2. ~~**`f` beyond 800 tokens** — E4's owed item 3.~~ **CLOSED by §8.2: linear, 0.0319 ms per token of context, no knee to 1600.**
3. **R3 at 7 B**, if anyone wants a clean scale statement about the ternary damage.
4. The 11.2× gap to 50 tok/s is **entirely the weight path**, now on trained weights.
5. ~~**FIRST: give `--bench` a contention witness.**~~ **DONE, §11, `WITNESS-CONFIRMED`.**
   All three gates pass, including the planted control. It validates nothing retroactively.
6. **A 1600-context `avx4` rate.** §10.4 says it should be 6–7% above `serial`; nothing on
   this machine has resolved it, and item 5 is the prerequisite.

---

## 13. CORRECTION, filed 2026-09-07 while pre-registering E15 — the packed arm is `--fold layers`

**§6 and brief §7 say both E7 arms are exported `--fold none`.** That is true of the fp32 arm and
**false of the packed one**. The sidecars are the authority and they disagree with the prose:

| artifact | `load_dtype` | `fold` | `n_gains_folded` |
|---|---|---|---|
| `qwen25-coder7b_f32.bin` | `bfloat16` | none | — |
| `qwen25-coder7b_p.bin` | *(absent)* | **`layers`** | **56** |

`export_packed.log` agrees: `folded 56 RMSNorm gains (--fold layers)`, and its
`sha256 d31c5047cb331f15…` matches the packed sidecar. The reasoning in §6 is sound — the bf16
low-memory loader does refuse `--fold` — but it applies only to the **fp32** arm, which is the one
that used it. The packed arm was exported before that path existed, under the default float32
load, and took the exporter's default fold.

**What this changes, and what it does not.**

- **No speed number moves.** E7 brief §7 already argues the fold "changes neither the file layout
  nor the weight count, and therefore cannot move a speed number", and that argument holds
  regardless of which way the fold went. §8's `REAL-WEIGHTS-CONFIRMED` and every rate quoted from
  this probe stand.
- **No parity or greedy gate moves.** G-L, G-P, G-G and G-C all compare the engine against a
  PyTorch reference **built from the same export**, so they are indifferent to the fold.
- **It does change what the packed arm is.** §5's "§5's arm is R0/fold-none" is half right: R0 yes,
  fold-none no. The packed 7 B is **R0 + `--fold layers` + `--head-ternary`**, a configuration no
  other standing artifact matches on either axis.
- **It matters because the fold is not neutral under ternarization.** It multiplies a gain into
  every row before `sign(w)` and `mean|w|` are taken, so it changes the codes. E2 §3.1 measured
  that at 0.5 B: `TQH` 4.531234 -> `NLH` 4.001988, **−0.529 BPB**, and `NLH` is the only 0.5 B
  ternary artifact on record that lands *below* the chance line.

**Nothing above §12 has been edited.** §5's honest reading — "R0 at 7 B collapses completely, and
how much of that is the rule versus the scale is a separate experiment nobody has run" — is what
E15 was pre-registered to answer (`briefs/BRIEF_E15_DOES_THE_7B_PREDICT.md`, `7c4243f`).

---

## 14. §5's question is answered — E15, and it is worse than "collapses"

§5 wrote that "R0 at 7 B collapses completely, and how much of that is the rule versus the scale is
a separate experiment nobody has run", and deliberately reported no BPB. **E15 ran it.**
`qwen25-coder7b_p.bin`, full 24×512 heldout slice, `--seqlen 512`:

| arm | BPB | chance 4.070106 |
|---|---|---|
| fp32 `qwen25-coder7b_f32.bin` | **0.674026555** | **−3.396080** |
| packed `qwen25-coder7b_p.bin` | **5.299200075** | **+1.229094** |

**The packed artifact is 1.229 BPB WORSE than guessing uniformly.** §5's refusal to quote a number
was right, and its `0/160` greedy — E7's planted control, required to fail — was reading a model
that genuinely carries no information about the next token.

**This does not move a single number in §§3, 8, 9, 10 or 11.** Those measured the engine on a
ternary weight stream, and a ternary weight costs the same bandwidth whatever scale multiplies it.
**What it removes is the right to call 6.79 tok/s a rate for a working 7 B model.** Both halves of
E7's headline stay true and they are about different artifacts: the fp32 arm at `160/160` predicts
(0.674027) and the packed arm at `4.853–4.888 tok/s` does not.

**§12 owed item 3 is promoted, not closed.** E15 measured `R0` at 7 B; `R3` at 7 B has still never
been built, and E15 §7 makes it the programme's most consequential open measurement — R3 reads
3.475707 at 1.5 B, **0.594 below** the line, so a ternary 7 B that predicts is not ruled out.
