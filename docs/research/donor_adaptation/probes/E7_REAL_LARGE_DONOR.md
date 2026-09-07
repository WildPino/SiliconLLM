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

## 10. Owed

1. **A real 10 B.** Still the open item. E7 reached 67% of the size; the rest needs a download.
2. ~~**`f` beyond 800 tokens** — E4's owed item 3.~~ **CLOSED by §8.2: linear, 0.0319 ms per token of context, no knee to 1600.**
3. **R3 at 7 B**, if anyone wants a clean scale statement about the ternary damage.
4. The 11.2× gap to 50 tok/s is **entirely the weight path**, now on trained weights.
