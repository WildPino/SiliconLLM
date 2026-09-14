# E62 — does the one-byte rung stay faithful as the donor grows?

**Verdict: `THE-COST-DOES-NOT-RANK`.**

Brief: `briefs/BRIEF_E62_DOES_THE_ONE_BYTE_RUNG_SURVIVE_SCALE.md`, pre-registered and pushed
`4c91b1c` before the runner existed; apparatus `695371b`, pushed before the run.
Runner: `benchmarks/donor_adaptation/engine/e62_third_scale.py` (33 self-tests).
Results: `benchmarks/donor_adaptation/engine/results/e62_third_scale.json`.
Run: 2026-09-14, **147.8 min**, one binary (`donor_engine_e61.exe`), `--threads 6`.

---

## 1. What was asked, and the one-sentence answer

`SPEED_LEDGER` §60.4's 10 B headline — 36.6–37.6 tok/s, 73–75% of the bar — is a **speed** claim
resting on a **quality** assumption: that one byte per weight is still faithful at 10 B. Two
points existed and they fitted either `N^2.578` (≈0.15 BPB at 10 B, headline void) or linear in N
(≈0.008, rung fine): a ~19× spread at the target. E62 measured a third scale to separate them.

**It separated them, and then took the question away.** The damage at 3 B is **0.000579 BPB**, in
the `AT-MOST-LINEAR` band — the explosion is not happening. But the damage is **not monotone in
N**: the 1.5 B donor is the worst of the three, 2.16× the 3 B. `G-E62d` clause 1 breaks, and by
the rule registered before the run, **the score is reported without a trend claim.** The two
anchor points were never a segment. A law was fitted through two adjacent points and the third
point refuted the law's *existence*, not merely its exponent.

## 2. The three points

| donor | N (total) | BPB fp32 (engine) | BPB int8 | **dBPB** | σ_seed |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.4941 B | 0.871810465558026 | 0.8718768872984035 | **6.64185e-05** | 0.013 |
| Qwen2.5-1.5B | 1.5437 B | 0.767606372511151 | 0.7688588381536873 | **1.2524656e-03** | 0.250 |
| **Qwen2.5-3B** | **3.0859 B** | **0.724460853881077** | **0.725040070561116** | **5.792167e-04** | **0.116** |

`dBPB(3 B)` is **0.0008 of the baseline** (relative 7.995e-04) and **0.116 σ_seed** — an eighth of
the constant this programme judges every delta with.

**The rank break, stated plainly:** 6.64e-05 → 1.25e-03 → **5.79e-04**. Up, then down. The 1.5 B
cell is the extreme, not the 3 B one.

| fit | exponent | implied at 10 B | in σ_seed |
|---|---|---|---|
| the two points that existed before today | `N^2.578` | 0.1548 | 31.0 |
| all three points | `N^1.321` | 0.0052 | 1.0 |
| linear in N from the 3 B point | — | 0.0019 | 0.38 |

**The three-point fit is printed to show that it is meaningless, not to be used.** A power law
through non-monotone points is an artefact of least squares, and §7 forbids reading it.

## 3. `G-E62b` — the cross-path control: ADMISSIBLE

Engine fp32 against E12's PyTorch fp32 on the same weights (**revision `3aab1f19…`, the same
snapshot E12 measured, checked on disk**) and the same slice.

| donor | engine | E12 PyTorch | \|d\| |
|---|---|---|---|
| 0.5 B | 0.871810465558026 | 0.8717951206093644 | 1.53e-05 |
| 1.5 B | 0.767606372511151 | 0.7675949584171732 | 1.14e-05 |
| **3 B** | **0.724460853881077** | **0.7244497971214012** | **1.11e-05** |

Tolerance 5.0e-05. The value of this control is not that it passed but that it passed **in
family**: two independent implementations disagree by the same ~1e-05 at all three scales. A
wrong `--fold` — the failure this control exists to catch — would have appeared here as ~0.22 BPB
(T3's measured fold), not 1.1e-05.

## 4. `G-E62a` — the planted control, and what it caught

| cell | measured | reference (E60) | d |
|---|---|---|---|
| `05b F32` | 0.871810465558026 | 0.871810465558026 | **+0.00e+00** |
| `05b I8` | 0.8718768872984035 | 0.8718768840274047 | **+3.27e-09** |

`FIRES` at a 1e-08 tolerance. The nonzero cell is **fully attributed and was already measured**:
E60's int8 reference was taken before E61 wired `--mvacc` into `matvec_sel`'s third branch, and
`donor_engine_e61.exe` defaults to four accumulators, so the dot sums in a different order.
E61's `results/e61_one_chain.json` records `05b` `m1_bpb 0.8718768840274047` and
`m4_bpb 0.8718768872984035` — **the two numbers in this table, to the last digit.** `05b F32`
reproduces at exactly zero because the fp32 branch has had four chains since E8.

So the two planted cells differ precisely as E61 said they would, on a different day and a
different run. That is a stronger instrument check than a pass would have been — and it still
**falsifies prediction 1**, which said both would read 0.00e+00.

The anchors are `m1` cells and the 3 B cell is `m4`; the difference is ~3e-09, six orders of
magnitude below `G-E62c`'s 0.0025 band edge, and cannot move any verdict here.

## 5. `G-E62e` — E12's owed 3 B packed cell, paid: ABOVE-CHANCE

The R3 packed arm, at the rule the programme actually ships:

| donor | BPB packed R3 | vs chance 4.069819 | damage vs own fp32 |
|---|---|---|---|
| 0.5 B | 4.531234 | above | 3.659423 |
| 1.5 B | 3.475707 | **below** | 2.708100 |
| **3 B** | **4.234751** | **above again** | **3.510290** |

This closes `probes/E12_TERNARY_COST_VS_SCALE.md`'s own open item. **It is also non-monotone, in
the opposite direction**: at half a byte the 1.5 B donor is the *easiest* of the three, while at
one byte it is the *hardest*. Both formats make 1.5 B the extreme; they disagree about the sign.
No mechanism is offered for that here — it is recorded as an observation, and §8 names it as
owed.

Post-hoc ternary at 3 B is **above chance**, i.e. worse than emitting no model at all. The rung
that works is still one byte, and the cliff between 1 B and half a byte (E60) is now confirmed at
a third scale.

## 6. `G-E62d` — the RANK partner: CLAUSE1-BREAKS

- **Clause 1** — dBPB strictly increasing in N: **breaks** (§2).
- **Clause 2** — within 3 B, `BPB(F32) < BPB(I8) << BPB(PACKED)`: **holds**,
  0.724461 < 0.725040 < 4.234751, the two gaps being 5.79e-04 and 3.51, a ratio of **6062×**.

E14 §3 exists for exactly this. The SCORE (`AT-MOST-LINEAR`) survives; the trend claim does not.
Registering the partner *before* the run is what makes that separation binding instead of a
choice made after seeing the number.

## 7. What E62 cannot claim

- **It does not establish the 10 B value, and now it cannot even bound it by extrapolation.**
  Before today the worry was that the fit pointed the wrong way; the finding is that **there is
  no fit**. `AT-MOST-LINEAR` removes the strongest reason to think the rung dies at 10 B and
  replaces it with a weaker epistemic position than three ranked points would have given:
  damage at one byte per weight is not a function of scale in any simple sense.
- **§60.4's 10 B headline is not void, and is not confirmed either.** Its fidelity assumption was
  tested at the only scale available and was not contradicted. The 10 B cell remains owed and
  **measured is the only thing that will settle it**.
- **Nothing about speed.** No `--bench`, no tok/s, no bandwidth was run.
- **Nothing about other families** (Coder-7B excluded on purpose: different pretraining mix).
- **Nothing about healing.** These are post-hoc conversions; H2T's question is untouched and
  still T4-gated behind H1.
- **No T4 time is asked for.**

## 8. Predictions: 2.5 of 6 — the worst record in this series

| # | registered | read | |
|---|---|---|---|
| 1 | both planted cells at \|d\| = 0.00e+00 | F32 yes, I8 +3.27e-09 | **half** |
| 2 | `G-E62b` in 1e-05 … 3e-05 | **1.11e-05** | ✓ |
| 3 | `G-E62c` 0.0030 – 0.0060 → SUPERLINEAR-BUT-BOUNDED | **0.000579** | ✗ |
| 4 | clause 1 holds; PACKED below chance | clause 1 breaks; PACKED above chance | ✗✗ |
| 5 | packed R3 at 3 B reads 3.0 – 3.5, still falling | **4.234751**, rising | ✗ |
| 6 | 3 B fp32 reads 0.7244 ± 0.0001, best fp32 in the family | **0.724461** | ✓ |

The two that hit are both *instrument* predictions. **Every prediction about the physics was
wrong**, and prediction 3 was wrong by an order of magnitude on the side I had named in the brief
as the embarrassing one (*"the power-law fear was groundless and I over-alarmed"*). E61 scored
4.5/7 and E62 scores 2.5/6: on this axis my model is poor, and the brief's placement of
prediction 3 between the two registered extremes did not save it.

## 9. Conduct

Foreign occupancy **8.39% – 14.06%** across five cells, every one over `OCC_BAR = 4.39`.

| cell | foreign | seconds |
|---|---|---|
| `05b F32` (planted) | 13.50% | 714 |
| `05b I8` (planted) | 10.20% | 237 |
| `3b F32` | 14.06% | 4129 |
| `3b I8` | 11.25% | 1145 |
| `3b PACKED` | 8.39% | 918 |

**This changes nothing here and that is the point of the design.** E62 was scoped to the one axis
a dirty box cannot corrupt: the planted `05b F32` cell reproduced E60's value to `+0.00e+00` while
the machine was 13.5% foreign-occupied. Occupancy is recorded because E61's conduct section had to
be written from it, and it gates nothing.

## 10. Apparatus defects found, and what happened to them

1. **The runner printed foreign occupancy 100× high** (`foreign 1350.45%`). `e44_interval.Split.close()`
   returns percent already; E62 multiplied again in three print statements. **The stored JSON was
   never affected** — it holds the raw return — so no measurement is touched and every number in
   this probe is read from the JSON. Fixed in the runner mid-run; the running process kept the old
   source, so the run's own log still reads 100× high. Same family as the `CONFIG mvacc=4` defect
   the runner was built to guard against: *a printed number that looks plausible until you check
   its units.*
2. **The brief's `~size` column was wrong on two of three arms.** It costed the files at one byte
   per weight throughout and forgot the embedding table stays fp32.

   | arm | brief | measured | reconstruction |
   |---|---|---|---|
   | `F32` | 12.3 GB | 12.344 GB | — |
   | `I8` | 3.1 GB | **4.336 GB** | 1.245 embed + 2.775 body + 0.311 head |
   | `PACKED` | 1.6 GB | **2.793 GB** | 1.245 embed + 1.387 body + 0.156 head |

   The same arithmetic reproduces E60's 0.5 B int8 file to 2 MB, so the accounting is right and
   only the estimate was sloppy. **It moves no speed number**: an embedding is a lookup, not a
   stream, so every projection divides `active_weights_per_token` (**3,085,697,024** at 3 B) by a
   bandwidth, never the file size. This is the charged-vs-moved-bytes law applied to a file
   listing rather than a rate.
3. **Every export's sidecar and every engine run's `CONFIG quant=` was checked against what was
   asked**, and all eight checks passed silently. Nothing was caught — which is what a working
   guard looks like when the protocol is right.

## 11. Owed after E62

1. **Why is 1.5 B the extreme at both formats, with opposite signs?** New, and the most
   interesting thing E62 produced. It is not head-dim (0.5 B is HD=64, the other two HD=128) and
   not NKV (2 everywhere). A fourth scale would say whether 1.5 B is peculiar or the curve
   genuinely oscillates. Qwen2.5-7B/14B are not in the local cache; Coder-7B is and is excluded
   for corpus reasons.
2. **The 10 B cell at one byte per weight, measured.** Now the only thing that can settle the
   headline's fidelity leg, since extrapolation is off the table. At the carve this needs
   row-selected `matvec_sel` and `matvec_colacc` to accept an int8 kind.
3. `G-E61c` repeated on an idle box, with the **3 B speed cell** in the same clean sweep — the 3 B
   artefacts now exist on disk, so that sweep costs only its own time.
4. `C/T` on the donor engine; channel-granularity mixed precision; the `I8`−`T1` value-dependence
   (E61 §6, reopened); `G-E55a2`'s replacement interval gate; E56's `OCC_BAR` zero point; E42's
   void control.

## 12. Artefacts

| what | where | bytes | sha256[:16] |
|---|---|---|---|
| 3 B fp32 | `D:/_ktmp/e62/qwen25-3b_f32.bin` | 12,343,754,804 | `c479e1a62724c3e2` |
| 3 B int8 R8 | `D:/_ktmp/e62/qwen25-3b_i8.bin` | 4,336,059,956 | `9c5b5c11e20f1771` |
| 3 B packed R3 | `D:/_ktmp/e62/qwen25-3b_packed.bin` | 2,793,211,444 | `8a26bda651dc86d3` |

All three: `--model Qwen/Qwen2.5-3B --revision 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
--fold none --calib-seqs 32 --threads 6`, with `--rule R8 --head-ternary` (int8),
`--rule R3 --head-ternary` (packed), and nothing further (fp32). `mean_ternary_zero_fraction`
0.0186 (int8) and 0.4825 (packed R3).
