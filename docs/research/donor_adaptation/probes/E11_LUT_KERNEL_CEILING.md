# E11 — the LUT kernel is refuted as a speed lever, and the two kernels swap places at 16 MB

**Verdict: `NO-LIFT`.** Pre-registered and pushed as `briefs/BRIEF_E11_LUT_KERNEL_CEILING.md`
(`00d4446`) before the arm was written. E10 §6 owed item 1, closed.

---

## 1. The question

E10 left the engine on the packed kernel's ceiling, so the only remaining engine lever on the weight
path was **a different kernel**. One was already in the engine and had never been measured this way:
`--lut` (`matvec_lut`), probe-1's `pshufb` int8-accumulate path, which produces **64 weights per
`vpshufb`** where the packed path spends **four** to produce 32.

The number in `SPEED_LEDGER` §12 — `--lut` at **1.05×** — could not answer it: that packed arm still
had E8's single accumulator, and `matvec_lut` already used four, so E8 never touched it.

## 2. The answer, at the footprint the donor actually lives at

| E10's instrument, `n_in` 3584 fixed, 5 reps | packed G-w/s | **lut G-w/s** | lut ÷ packed |
|---|---|---|---|
| 4 MB | 48.14 | **85.04** | **1.77** |
| 8 MB | 50.13 | **91.07** | **1.82** |
| 12 MB | 53.41 | **90.48** | **1.69** |
| 24 MB | 51.41 | 44.66 | 0.87 |
| 48 MB | 48.59 | 31.09 | 0.64 |
| **512 MB (verdict cell, named in advance)** | 55.95 | **21.90** | **0.391** |
| 2048 MB | 60.25 | 21.42 | 0.36 |

**G-L1 = 0.391 against a ≤1.15 NO-LIFT boundary.** The LUT path is **~2.5× slower per weight** than
packed at donor-scale footprints. As a speed lever for the dense streamed donor it is **refuted**,
and the verdict cell was fixed in the brief before the arm existed precisely so this could not be
chosen afterwards.

## 3. The result nobody asked for: the ordering inverts at the L3 boundary

**The LUT kernel is the fastest weight kernel this programme has ever measured — 91 G-weights/s —
and only while its weights are L3-resident.** It falls **4.2×** across 16 MB. The packed kernel,
per E10, does not move across that boundary at all.

**The mechanism is the layout, read off the code rather than modelled.** `matvec_lut` reads
`codes + t*Mpad + base`, so consecutive `t` are **`Mpad` bytes apart**:

| cell | `Mpad` | stride between consecutive reads |
|---|---|---|
| 4 MB | ~2,368 | 2.3 KB, over a resident array |
| 512 MB | ~299,600 | **299 KB — 1,792 reads of 32 bytes across as many pages** |

**Tile-major is what lets one `vpshufb` serve 32 rows, and it is the same thing that destroys the
stream.** The property that makes the kernel fast and the property that makes it slow are the same
property, which is why no amount of tuning inside the loop would have found this.

**Labelled exploratory**: this was not in the pre-registered bands. It is reported because it is
large, reproduced across both runs, and points somewhere specific.

## 4. Gates, including one I got wrong

| gate | result |
|---|---|
| **G-L0** correctness | **PASS** — rel l2 **8.31e-03** between the LUT and packed arms on the same matrix, non-zero so the branch really ran |
| **G-L1** discriminator | **0.391 → NO-LIFT** |
| **G-L2** harness known-positive | **FAILS AS WRITTEN at 2 of 7 cells** |
| **G-L3** dispersion | 5 reps; the LUT arm is not flat, and that is the finding rather than a defect |

**G-L2 failed by my mis-specification, not by contamination — the fourth pre-registered rule in this
programme aimed at the wrong number.** I drew the 49–53 G-w/s band from E10's 7-rep sweep and
ignored E10's own 3-rep sweep, which read **57.11 / 60.78** at the two largest cells. Run 2 reads
**55.95 / 60.25** — inside E10's own between-run range, outside a band written from half of it.
**A known-positive band must come from every reading of the known-positive.** The five smaller cells
are in band, and the verdict is a 2.5× effect that the void run 1 also produced (0.30).

**G-L0 is not a quality number and must not be quoted as one.** The bench's synthetic `x` has a
crest factor near 1.7; the donor's activations measure **8.3 average, 69.6 maximum** (INDEX §2.1).
The published activation-quantization costs stand: **1.40e-01** whole-vector, **3.10e-02** at
`--lut-group 32`.

## 5. Run 1 was void, and the thing that caught it

The first sweep was declared **VOID** by G-L2 while a concurrent worker — a different checkout,
`SiliconLLM_private`, carrying a `--vecexp` flag that does not exist in this tree — streamed the
30.46 GB fp32 donor across six threads. **The packed arm acquired a slope it does not have**
(35.6 → 61.4 where E10 measured it flat).

**A within-run rule could not have caught it: every cell was contaminated together**, which is
exactly how E8's discard rule discarded zero pairs in a 12%-contaminated sweep. What caught it was
**a number carried in from a previous session and compared from outside the run** — E8 §9 item 4's
absolute plateau, in its first real use. Details in the brief §7.

## 6. What this changes

**Nothing about the goal.** Coder-7B is **7.4× short of 50 tok/s**, E10's §23.3 stands, and the
packed kernel remains the engine's weight path with ~1.03× of headroom.

**Something about the architecture already written down.** `SCALEUP_ARCHITECTURE` specifies a
**cache-resident keystone ≤16 MB L3** with experts streamed from DRAM, and probe-3 put that cliff at
exactly 16 MB. **A kernel that is 1.8× faster inside L3 and 2.5× slower outside it is shaped for
that split** — LUT resident, packed streamed. That is a hypothesis with a measurement under it, not
a plan, and before it becomes one it owes an end-to-end `--lut` rate through the whole engine
(Phase 60: a kernel result never composes to a system claim) and the activation-quantization cost on
**real** activations.

## 7. Owed

1. **A `--lut` rate through the engine at a resident shape**, with the parity gate — the claim above
   is a microbench and Phase 61's law says microbenches do not compose.
2. **The same sweep at `n_in` = 18944** — E10 §6 item 2, still open for both kernels.
3. **The non-packed ternary path still has a single accumulator** — E8 §9 item 3.
4. **A vectorised `expf`** — E9 §5. Note that a concurrent worker appears to be building exactly
   this (`--vecexp`), in another checkout.
