# E13 — the LUT collapse was the layout, and fixing it makes the LUT kernel the fastest one here

**Verdict: `LAYOUT-CONFIRMED`.** Pre-registered and pushed as
`briefs/BRIEF_E13_BLOCKED_TILE_MAJOR.md` (`89fd63e`) before the arm was written; the arm itself
landed in `076381a`. Closes E11 §7 item 1 and E10 §6 item 1's successor.

Donor: `Qwen2.5-Coder-7B` packed, 7.072 B active weights/token, and `Qwen2.5-0.5B` packed.
Machine: Ryzen 5 3600X, 6 threads, **idle** — see G-M2.

---

## 1. The question, and why E10 forced it

E10 returned `CORE-BOUND` and named the side without a mechanism. It also left one row of its own
table unread:

| E10's sweep, 2048 MB, DRAM-resident, 6 threads | moved GB/s |
|---|---|
| fp32 arm | **38.58** |
| packed arm | **26.48** |

Same instrument, same threads, same **moved**-byte convention on both sides — so the comparison is
legal, and it says the packed kernel was running at **69% of a streaming rate the same sweep had
just demonstrated**. `CORE-BOUND` therefore left **1.46× measured**, not zero.

E11 then found a kernel 1.8× faster inside L3 that fell 4.2× across 16 MB, and blamed **layout**:
`codes + t*Mpad + base` puts consecutive reads `Mpad` apart. **That was asserted, never tested.**

## 2. The defect, read in the source before anything was measured

`build_tm` stored `tm[t*Mpad + o]`; `matvec_lut` holds `base` fixed and walks `t`, so consecutive
32-byte reads sit `Mpad` apart:

| organ, Coder-7B | stride | line utilisation |
|---|---|---|
| `gate_proj` / `up_proj` (out 18944) | **18.5 KB** | 32 of 64 bytes |
| `down_proj` (out 3584) | **3.5 KB** | 32 of 64 bytes |
| kbench 512 MB cell | **299 KB** | 32 of 64 bytes |

Every read leaves the page and uses half a line. Total bytes read are **identical** to the packed
arm's — the same matrix, once — so this is purely an ordering defect.

**The change (`--lutblk`)**: store each 32-row tile's bytes contiguously,
`tm[(base/32)*T*32 + t*32 + r]`. Same bytes, permuted. The `t` order and the accumulate tree are
untouched, so it is **bit-identical** and the gate is sha256, not parity.

## 3. Gates, in the order they were read

| gate | reading | |
|---|---|---|
| **G-M2** — the known-positive, read first | packed arm **48.34 – 59.02** G-w/s across all seven cells; band 47–62; max/min **1.22** against ≤1.30; no slope | **PASS — the machine was quiet** |
| **G-M0** — engine, end-to-end | `--logits` 512 tokens at 0.5 B, `--lut` vs `--lutblk`: `7a9b3e04704aa416ba51ae88b26ccd499449dd687141e77217adeec0e2f48e44` **both**, 311 MB byte-identical | **PASS** |
| **G-M0b** — microbench mirror | blocked vs tile-major, **bitwise**: 0/2340 floats differ | **PASS** |
| `selftest-lut`, both layouts | A PASS · **B planted control FIRES 100/100** · C 7.89e-08 | **PASS** |
| **G-M1** — the verdict, 512 MB cell named in E11 in advance | **67.51 G-w/s** against **≥65 `LAYOUT-CONFIRMED`** | **`LAYOUT-CONFIRMED`** |

E10's own planted control also fired unprompted inside this run: fp32 4 MB → 512 MB = **4.91×**.

**The prediction was 65–77 and the measurement was 67.51 — inside its own band.** That is the
second prediction in this programme to land inside (after E9's 1.055–1.066), and it was derived the
same way: a **measured** quantity (the fp32 arm's demonstrated stream rate) over a **structural**
factor (0.5 B/weight), per E8 §3's corollary.

## 4. The sweep

`n_in` 3584 fixed, 5 reps, medians, `--mvacc 4`, 6 threads.

| footprint | packed | `lut` | **`lutblk`** | blk ÷ lut | blk ÷ packed |
|---|---|---|---|---|---|
| 4 MB | 51.40 | 83.80 | **87.84** | 1.05 | 1.71 |
| 8 MB | 50.22 | 75.50 | **97.87** | 1.30 | 1.95 |
| 12 MB | 51.59 | 86.55 | **103.05** | 1.19 | **2.00** |
| 24 MB | 55.32 | 63.77 | **101.22** | 1.59 | 1.83 |
| 48 MB | 48.34 | 34.90 | **73.11** | 2.09 | 1.51 |
| **512 MB (verdict cell)** | 56.72 | 22.26 | **67.51** | **3.03** | **1.19** |
| 2048 MB | 59.02 | 20.84 | **71.82** | **3.45** | 1.22 |

G-w/s, medians of 5. **The `lut` 8 MB cell carried a 100.8% spread and its median is not to be
quoted**; it does not touch the verdict, which is at 512 MB, but it is recorded rather than dropped.

**Three things this settles.**

1. **E11 §3's mechanism paragraph was right, and is now tested rather than asserted.** Blocking
   recovers **3.0–3.5×** at the two largest cells. The 4.2× cliff is gone: `lutblk` falls 103.05 →
   67.51 across L3, **1.53×**, not 4.2×.
2. **The LUT kernel now beats packed at every footprint measured**, 1.19× to 2.00×. **103.05 G-w/s
   at 12 MB is the fastest weight kernel this programme has measured** (the previous best was
   E11's 91.07, same kernel, worse layout).
3. **At donor-scale footprints `lutblk` is now bandwidth-bound.** It moves **33.76 GB/s** at the
   512 MB cell against the fp32 arm's **34.75 GB/s** in the same sweep — **97%**. The 1.46× that
   E10 left on the table has been taken, and what remains at that footprint is ~3%.

**What E11's `NO-LIFT` was and was not.** It was a correct verdict on **the layout as shipped** —
`--lut` really was 2.5× slower per weight at donor scale, and remains so. It was **not** a verdict
on the LUT arithmetic, and this probe does not overturn its numbers; it explains them. E11 §3 said
so at the time and labelled itself exploratory. The `SPEED_LEDGER` §12 and INDEX §2.1 corrections
E11 made still stand.

## 5. Does it compose? (Phase 61's law)

A compute-bound microbench does not compose to a memory-bound engine, so the kernel result buys
nothing until it is measured through `forward()`.

**0.5 B, `--bench 300`, 7 interleaved reps per arm, idle machine:**

| arm | rates, sorted | median | spread |
|---|---|---|---|
| **`--lutblk`** | 86.09 87.04 88.49 91.31 92.65 93.41 94.14 | **91.31** | 8.8% |
| `--lut` | 58.00 58.72 59.14 59.72 60.23 61.00 62.24 | 59.72 | 7.1% |
| packed (default) | 65.33 74.19 74.84 75.03 76.24 76.36 76.91 | 75.03 | 15.4% |

**`--lutblk` = 1.217× the packed default, and 1.529× the shipped `--lut`.** `ffn~` witness moves the
same way: 6.0–6.7 ms against packed's 7.6–9.2 and `--lut`'s 9.8–10.5.

**A first 3-rep pass gave `lutblk` a 180% spread (31.02 … 86.86) and is discarded here as
warm-up**, not quoted as a result. It is recorded because it is exactly the shape of reading the
±5%/≥3-rep rule exists to catch, and three reps were **not** enough at this shape.

**The engine gets less than the kernel does, and that is expected but not yet decomposed.** The
microbench gives 1.7–2.0× at resident footprints; the engine gives 1.217×. Amdahl on a ~70% weight
path at 1.8× would predict ~1.45×, so roughly a quarter of the kernel win is being spent
elsewhere — most plausibly the LUT path's per-call `quant_i8` (O(`n_in`)) and `build_lut`, which
writes 16 bytes per input pair on **every matvec call** and which the packed path does not pay.
**That is arithmetic and a hypothesis, not a measurement — a `--profile` decomposition is owed.**

**Coder-7B, `--bench 100`, 5 interleaved reps per arm, idle machine:**

| arm | rates, sorted | median | spread | `ffn~` |
|---|---|---|---|---|
| **`--lutblk`** | 8.99 9.03 **9.21** 9.27 9.33 | **9.21** | 3.7% | 85.7 – 90.1 ms |
| `--lut` | 3.73 3.74 **3.78** 3.99 4.07 | 3.78 | 9.0% | 175.5 – 184.8 ms |
| packed (default) | 6.22 6.38 **6.78** 6.82 6.84 | 6.78 | 9.1% | 115.8 – 128.3 ms |

**`--lutblk` = 1.358× the packed default at donor scale.** And `--lut` as shipped is **0.557×** —
**1.8× slower through the engine**, which confirms E11's `NO-LIFT` end-to-end rather than softening
it.

**The engine baseline reproduces to 0.15%.** Packed median **6.78** against E9's published
**6.79**, and its `ffn~` plateau on reps 3–5 reads 115.8 / 117.3 / 117.4 against E9's published
114.5–117.0. Reps 1–2 sat at 128.3 / 125.9 — above the plateau, the same warm-up the 0.5 B pass
showed. **The witness is doing its job from outside the run**, which is the property E11 run 1
depended on.

**Here the microbench composes, once the right cell is used.** Coder-7B's fused organs are
`gate|up` at **67.9 MB** and `down` at **33.9 MB** — the 24–48 MB region, not 512 MB. The sweep
gives `lutblk ÷ packed` of **1.83** and **1.51** there, and Amdahl on a ~70–80% weight path
predicts **1.3–1.4×**. Measured **1.358×**. **The 1.19× at the 512 MB cell was never the engine's
number**; reading it as one would have been the Phase 61 error in the other direction.

## 6. What this does NOT license

- **`--lutblk` is not shippable on the strength of this.** The LUT path quantizes activations to
  int8, and E11 measured that cost as **1.40e-01** relative L2 whole-vector, **3.10e-02** at G=32.
  That is a **quality** cost on the donor and it has never been carried through to BPB or greedy
  parity end-to-end. **This probe makes the LUT path fast enough to be worth that question; it does
  not answer it.** Until it is answered, the packed default stays the default.
- G-M0's sha256 identity is `--lut` vs `--lutblk` — **the two LUT layouts agree bit-for-bit with
  each other**. It says nothing about either agreeing with the packed path, which they do not, by
  construction (int8 activations).
- Every absolute tok/s here carries ±5% between sweeps; the ratios do not. Today's packed 0.5 B
  median 75.03 against E9's published 79.12 is 5.4% — at the edge of that law, and the reason the
  arms were interleaved within one sweep rather than compared across sessions.

## 7. Where the goal stands

| Coder-7B, 7.072 B active/token | tok/s | short of 50 | path |
|---|---|---|---|
| E7, this morning | 4.460 | 11.2× | exact |
| after E8 (dependency chain) | 6.37 | 7.8× | parity-gated |
| after E9 (glue) | 6.79 | 7.4× | bit-exact |
| **after E13 (`--lutblk`)** | **9.21** | **5.4×** | **LOSSY — int8 activations, quality unmeasured** |

**2.065× in one day.** But the last row is not the same kind of number as the three above it, and
it must never be quoted as if it were:

- 4.460 → 6.79 is on a path whose parity to PyTorch is **1.070e-05** and whose greedy agreement is
  **160/160**.
- 6.79 → 9.21 is on a path that quantizes activations to int8 with a measured **1.40e-01**
  relative L2 error and **has never been carried to BPB or greedy parity on the donor**.

**So the honest statement of today's speed is 6.79 tok/s exact, with a 1.358× lever available on a
lossy path whose cost is the next thing that has to be measured.** If that cost turns out to be
acceptable, the goal is 5.4× away; if it does not, it is 7.4× away and E13 is a kernel result
without a product.

**What E13 does close** is E10's open 1.46×. The weight path at donor-scale footprints is now at
**97% of the streaming rate this machine demonstrates**, so there is no third kernel to write.
**§19.3 stands and is now better supported than ever: the remaining 5.4× is a property of the
model.** The only levers left are fewer active weights per token and residency — which is what E12
was pre-registered to test, and why it is still the decisive experiment.

## 8. Owed

1. **The `--profile` decomposition of §5's missing quarter** — is it `build_lut`/`quant_i8`, and is
   it removable by hoisting the table across the organs that share an input (`q|k|v`, `gate|up`)?
2. **The quality question `--lut` has always owed**: BPB and greedy parity on the donor with int8
   activations, at `--lut-group 32` as well as whole-vector.
3. **The same kbench sweep at `n_in` = 18944** — E10 §6 item 2, still open, and now more
   interesting because `lutblk` changes the stride story.
4. The non-packed ternary path still has a single accumulator (E8 §9 item 3).
5. A vectorised `expf` (E9 §5).
