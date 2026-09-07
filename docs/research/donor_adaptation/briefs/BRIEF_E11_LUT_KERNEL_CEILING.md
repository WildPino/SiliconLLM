# E11 — does the LUT kernel have a higher per-weight ceiling than the packed one?

**Pre-registered. Pushed before the arm is written.** E10 §6 owed item 1.

---

## 1. Why this is the next question and not a re-run

E10 measured the packed `matvec` at **49–53 G-weights/s at every footprint from 4 MB to 2 GB**, and
the engine's own FFN organs at **51.66 / 52.22**. The engine is on that kernel's ceiling, so the
only remaining engine lever on the weight path is **a different kernel**.

There is one already in this engine, and it has never been measured this way: **`--lut`**
(`matvec_lut`), probe-1's `pshufb` int8-accumulate path. Per `vpshufb` it produces **64 weights**
(32 rows × 2 trits); the packed path spends **four** `vpshufb` plus four `vpmovsxbd` and four
`vcvtdq2ps` to produce **32**.

**Why the number already in the ledger cannot answer this.** `SPEED_LEDGER` §12 records `--lut` at
**1.05×** over packed. **That packed arm still had E8's single accumulator.** Packed has since
gained **1.349×**, and `matvec_lut` already used four accumulators (`acc[4]`), so E8 never touched
it. Taken at face value the old ratio now implies the LUT path is ~1.29× *behind* — but §12's
figures are whole-token rates at the 0.5 B shape, not kernel ceilings, so they cannot settle it
either way. **The published "`--lut --fuse` adds ~3.9%" line in `INDEX` §2.1 is, on the same
grounds, a number taken against a crippled arm and is flagged here as suspect pending this run.**

## 2. What this brief is NOT about

**`--lut` is lossy and stays lossy.** It quantizes the activation vector to int8: `rel l2`
**1.40e-01** whole-vector, **3.10e-02** at `--lut-group 32` (T2b §, ledger §12). E11 measures a
**ceiling**, nothing else. **No adoption, no tok/s claim, and no comparison of engine rates follows
from it** — that would need the end-to-end parity gate this programme requires, and Phase 60's law
says a kernel result never composes to a system claim on its own.

## 3. The instrument

`kbench.c` from E10, unchanged in its packed and fp32 arms, plus a third arm that sets `m->tm` and
lets the engine's own `matvec` take the `--lut` branch — same source, same harness, same fixed
`n_in` = 3584, same footprints, same reps.

**The LUT arm carries per-call work the packed arm does not:** `quant_i8` over `n_in` and
`build_lut` over `H`, writing 16 bytes per input pair (~28.7 KB per call at this `n_in`). That cost
is amortised over the row count, so it **penalises the small cells**. **The verdict is therefore
read at the 512 MB cell**, which is the size range the donor's real organs live in, and the small
cells are reported as context only. Fixed here so it cannot be chosen after the fact.

## 4. Gates and bands, fixed before the arm exists

| gate | test |
|---|---|
| **G-L0** | **correctness**: the LUT arm's `y` against the packed arm's `y` on the same matrix. Must be `rel l2` **≤ 0.20** — the published costs are 1.40e-01 whole-vector and 3.10e-02 at G=32, and garbage is O(1) — **and must not be ≈0**, which would mean the LUT branch never ran. FAIL either way ⇒ every LUT cell is void |
| **G-L1** | **the discriminator**: LUT G-w/s ÷ packed G-w/s at the **512 MB** cell |
| **G-L2** | **harness known-positive**: the packed arm in *this* run must land in E10's **49–53 G-w/s**. If the harness no longer reproduces the result it was built to produce, nothing in the run counts |
| **G-L3** | ≥3 reps/cell, idle, dispersion reported — **and the verdict is read off the arm's flatness across cells, not off one cell's ratio.** This is E10's G-K3 correction applied immediately rather than discovered again |

**Verdict bands:**

| LUT ÷ packed at 512 MB | verdict |
|---|---|
| **≥ 1.50** | **LUT-LIFTS** — a real kernel replacement exists, at a known quality cost, and E10's ceiling is not the engine's ceiling |
| 1.15 – 1.50 | **PARTIAL** |
| **≤ 1.15** | **NO-LIFT** — the binder is common to both kernels, and it is not the packed unpack |

**No magnitude is predicted, and the reason is E10 itself:** that probe measured this kernel family
at **0.29 FMA per cycle per core out of 2** and could not name the mechanism. A prediction built on
uop counts would be exactly the port model E10 refused to build. **Three verdicts, no favourite.**

## 5. What each verdict buys

**NO-LIFT** is the informative one: two kernels with completely different unpack work and the same
0.5 B/weight stream, landing at the same ceiling, would say **the binder is per-byte and common** —
which retires the half-wasted-`vpshufb` candidate too, and leaves the engine's weight path finished
at ~52 G-w/s until the *format* changes. It would also confirm the §12 correction in §1.

**LUT-LIFTS** re-opens the weight path — and immediately owes a quality experiment, because
1.40e-01 is not a rounding error.

## 6. Honest ceiling

Even LUT-LIFTS at 1.5× moves Coder-7B from 6.79 to roughly 8.5 tok/s: **still ~6× short of 50**,
and only if the quality cost were acceptable, which is not established. **This does not change the
strategic picture** — §19.3 and §23.3 stand. It decides whether the engine's weight path is
finished or has one more move in it.
