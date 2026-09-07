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

---

## 7. RUN 1 — **VOID**, and the gate that caught it was the one added for this reason

The LUT arm was built and the sweep ran. **G-L2, the harness known-positive, FAILED, so no cell in
the run counts and none is quoted as a result.**

| | E10, clean | run 1 |
|---|---|---|
| packed 4 MB | 52.07 G-w/s | **35.64** |
| packed 12 MB | 50.40 | **39.54** |
| packed 512 MB | 50.97 | **58.35** |
| packed 2048 MB | 52.95 | **61.36** |
| packed spread, worst cell | 30.6% | **37.8%** |
| fp32 spread, worst cell | 37.0% | **77.7%** |

The packed arm did not merely drift — it **acquired a slope it does not have**, rising 35.6 → 61.4
across the sweep where E10 measured it flat to ±4%. The pre-registered rule said: *if the harness no
longer reproduces the result it was built to produce, nothing in the run counts.* It doesn't, so it
doesn't.

**The cause, found by looking rather than assumed:**

```
D:\_THINGS\Progetti\SiliconLLM_private\...\donor_engine.exe
  --weights D:\_ktmp\e7\qwen25-coder7b_f32.bin --threads 6 --vecexp 1
  --generate D:\_ktmp\e7\p0.bin 32 D:\_ktmp\test_gen_p0
```

PID 7484, started 17:26:17, **six threads and a 30.46 GB fp32 model being streamed off disk**, from
a **different checkout** (`SiliconLLM_private`) and carrying a **`--vecexp` flag that does not exist
in this tree** — E9 §5's owed vectorised `expf`. **A concurrent worker, not an orphan of mine.**
It was left running. E8's kill was of my own surviving child; this is somebody else's run.

**This is the first time the absolute-plateau discipline has actually earned its keep.** E8 §9 item
4 asked for a per-shape witness plateau *published as an absolute*, precisely because E8's
within-sweep discard rule discarded zero pairs in a sweep that was 12% contaminated end to end. A
within-run rule could not have flagged run 1 either — every cell was contaminated together. **What
flagged it was a number carried in from a previous session** (E10's 49–53 G-w/s), compared from
outside the run. G-L2 exists because E10 wrote that lesson down two hours earlier.

**E10 is unaffected.** Its sweeps predate 17:26:17, its own dispersion was 0.8–30.6%, and its G-K2
tied it to the engine's independently measured 26.1 GB/s — a cross-session anchor of exactly the
kind that caught this.

**One thing run 1 does establish, because it is deterministic and immune to load: G-L0 PASSES.**
The LUT arm computes the same matvec as the packed arm — `rel l2` **8.31e-03** against the packed
arm's output on the same matrix, non-zero (so the LUT branch really ran) and far under the 0.20
gate. **But that number is NOT a quality measurement and must not be quoted as one:** the bench's
synthetic `x` has a crest factor near 1.7, where the donor's activations measure **8.3 average and
69.6 maximum** (INDEX §2.1). The published costs on real activations remain **1.40e-01**
whole-vector and **3.10e-02** at `--lut-group 32`. G-L0 checks the arm, not the format.

**Status: E11 is open. The gates stand as written; run 2 waits for an idle machine.** Quality and
parity work is unaffected by the contention and can proceed meanwhile — that is the standing rule
and it applies here.
