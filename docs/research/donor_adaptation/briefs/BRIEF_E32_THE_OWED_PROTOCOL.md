# BRIEF E32 — does `--lutblk` survive its own protocol? The `--seqlen 512` re-run E14 owes

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. Why this probe, and why it is the item that can still overturn E28 and E30

E30 closed the numerator: the engine sits at **0.909** of this box's measured read bandwidth, a
perfect kernel is worth **6.83 tok/s** at the goal's shape, and the remaining gap is not available
to kernel work. **Every one of those numbers is read off `--lutblk`.** E28's 61.64 G-w/s, E30's
33.01 GB/s, the 1.424× shape lever — all of it is the int8-activation LUT path.

What licenses spending that lever is E14's verdict `ACTIVATION-CHEAP`. **And E14 disqualified its
own verdict in its own §1**, before publishing it:

> `donor_engine.c:1249` reads `int SL = seqlen>0 ? seqlen : (int)n;` — with no `--seqlen`, the
> whole ids file is one sequence. **E14 scored one 12,288-token sequence, not 24 documents of
> 512.** Every arm ran with RoPE positions up to 12,287 and attention reaching across 23 unrelated
> document boundaries. G-N2's bands (`0.010` / `0.020`) were drawn from **512-context** regimes.
> **A `--seqlen 512` re-run of both cells is owed**, and until it lands the verdict is a verdict
> at 12 k context.

E14 §5 has carried that item since. E28 §8 item 3 and E30 §8 item 1 both re-state it. **It has
never been run**, and in the meantime the whole engine-side story was built on top of it.

## 1. The question

> **At the protocol the bands were drawn from — 24 documents of 512, `--seqlen 512`, E1's byte
> convention — what does int8 activation quantization cost in BPB, and does the greedy trajectory
> agree?**

## 2. Cells, arms and protocol

Real trained donors, the same artifacts E1 and E14 used, unchanged on disk since 2026-09-05:

| cell | weights | ids |
|---|---|---|
| 0.5 B | `qwen25-05b_tqh.bin` | `ids_qwen25-05b_tqh.bin` |
| 1.5 B | `qwen25-15b_tqh.bin` | `ids_qwen25-15b_tqh.bin` |

| arm | activations |
|---|---|
| `A0` | fp32 — the baseline, the shipped packed path |
| `A1` | int8, one scale per vector (`--lut`) |
| `A2` | int8, one scale per 32 channels (`--lut --lut-group 32`) |
| **`A3`** | **E13 blocked layout (`--lutblk`) — the arm E28 and E30 are built on** |

**Protocol, and this is the entire point of the probe**: `--seqlen 512`, and **E1's byte
convention — 51,870 scored bytes over 12,264 predictions**, not E14's 51,941 over 12,287. The
denominator is where the unit hides and in E14 it moved twice at once. The chance line is
recomputed from *this* denominator and reported as such.

This is a **quality** measurement: deterministic, and by the standing law it may be taken on a
loaded box. **No timing is taken here and none may be quoted from this run.**

## 3. The gates

**`G-E32A` — THE ANCHOR, and it is a planted control on the protocol itself.** E1 published
`4.531234` for the 0.5 B TQH artifact on this exact slice at 512 context (`e1_05b.log:30`).
**`A0` at 0.5 B must reproduce it within `0.001` BPB.** If it does not, this run is not at E1's
protocol, the whole reason for the re-run is gone, and the probe is VOID.

**`G-E32B` — THE KNOWN POSITIVE, taken from E14's own strongest result.** E14 measured
`A3 − A1 = +0.000e+00` at both cells, identical to the last digit, verified three independent
ways. **`A3` must equal `A1` exactly here too**, at both cells. An instrument that cannot
reproduce an exact identity is not reading the arms it believes it is reading, and the probe is
VOID.

**`G-E32C` — the regime condition.** E14's §2 demoted its own 0.5 B cell to instrument-only
because `A0` sat **above** the chance line, where a `dBPB` toward chance is not a quality signal.
The verdict cell is read at 1.5 B and **only if `A0(1.5 B)` sits below the chance line at this
protocol.** If it does not, there is no verdict cell and the probe reports instrument only.

## 4. The verdict cell, named before the run

**`dBPB(A3) = BPB(A3) − BPB(A0)` at the 1.5 B cell, at `--seqlen 512`**, read against E14's
registered bands — which were drawn from 512-context regimes and, for the first time, share a
provenance with the measurement:

| band | name |
|---|---|
| `abs(dBPB) ≤ 0.010` | `ACTIVATION-CHEAP` |
| `0.010 – 0.020` | `MARGINAL` |
| `≥ 0.020` | `ACTIVATION-COSTLY` |

**And the RANK partner is mandatory, not optional** (E14 §3's law: every SCORE metric needs a
RANK partner). Greedy top-1 agreement between `A3` and `A0` over the E6 prompt set, at this
protocol. E14 read **45.6% (73/160)** at 12 k while BPB called the same arm cheap. **A cheap BPB
with a broken trajectory is reported as exactly that and is not allowed to inherit the word
"cheap" on its own.**

## 5. Predictions — fixed here, before the run

1. **`G-E32A` fires**: `A0(0.5 B)` within `0.001` of E1's `4.531234`.
2. **`G-E32B` fires**: `A3 − A1 = 0` exactly, at both cells.
3. **The verdict does NOT flip to `ACTIVATION-COSTLY`**: `abs(dBPB(A3))` at 1.5 B stays below
   `0.020`. E14's arm-vs-arm comparisons were protocol-invariant by construction, and only the
   band reading was contaminated.
4. **`dBPB(A3)` at 1.5 B moves TOWARD zero relative to the 12 k reading of `−0.016961`** — I
   predict the interval `[−0.015, +0.005]`. At 12 k the baseline is attending across 23 unrelated
   document boundaries, and an over-confident logit vector is exactly what int8 noise "improves";
   at 512 there is less of that pathology for the noise to fix.
5. **The RANK partner stays bad**: greedy top-1 agreement between `A3` and `A0` at 1.5 B stays
   **below 70%**. The score/rank divergence is E14's real finding and it should survive a protocol
   change that was never about ranking.
6. **Registered in advance as independent of every band above.** If `--lutblk` turns out costly,
   the operative kernel becomes the packed default, which E30 measured at **0.639** of the
   ceiling — a perfect-but-packed engine reads about **4.8 tok/s** at T10 instead of 6.83.
   **E30's architectural conclusion is identical either way**, and this probe cannot rescue the
   goal; it can only decide which kernel the record is allowed to quote.

## 6. What this decides

* **`ACTIVATION-CHEAP` with an acceptable rank partner** ⇒ E28's and E30's numbers stand as
  published, and E14's owed item closes.
* **`ACTIVATION-CHEAP` in BPB with a bad rank partner** (what I expect) ⇒ the owed item closes but
  **`--lutblk` is a benchmark kernel, not a shippable one**, and every place the record quotes
  61.64 G-w/s or 33.01 GB/s must say so. E30's ceiling arithmetic is unaffected; E30's *engine*
  reading becomes a reading of a path that changes the model's output.
* **`MARGINAL` or `ACTIVATION-COSTLY`** ⇒ E28's headline and E30's engine row are both read off a
  kernel that is not usable, and the operative numbers fall back to the packed default. A
  correction goes at the TOP of both probe documents, not in a footnote.

## 7. What E32 will NOT be able to claim

- **Nothing at the goal's shape.** Both cells are real donors at 0.5 B and 1.5 B. T10 has
  synthetic noise weights and no BPB exists there; **the int8 cost at a 10 B shape remains
  unmeasured and unmeasurable with what is on this disk**, because activation outliers — the thing
  that makes int8 hard — are a property of *trained* weights and a noise export has none. A
  synthetic T10 reading would systematically understate the cost and is deliberately not taken.
- **Nothing about speed.** No timing, and E30's ±5% absolute rule is irrelevant because no
  absolute is produced.
- **Nothing about `--lut-group 32`'s speed.** E14 §0 already recorded that A2, the quality-cheaper
  arm, has no speed number at all. Still true, still owed.
- **Nothing about the head.** `--lut-no-head` is a separate lever with its own measurement.
