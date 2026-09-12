# BRIEF E34 — what does T10 read with the FFN GONE? The floor under the floor

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. Why this probe, and why it is the most goal-relevant one available

The goal is **10 B at 50 tok/s**. Every probe from E18 to E33 has attacked the FFN, because at T10
the FFN is nine tenths of the weight. **Nobody has ever measured what is left when the FFN is
gone.**

That number is decidable today, from two things already measured on this box:

* **E30's ceiling**: `BW-CEIL = 36.30 GB/s`, planted-control fired at 12.5×, cliff at 16 MB.
* **E32's operative numerator**: the packed kernel, `46.16 G-w/s` charged (E28's session), after
  `--lutblk` lost its quality licence.

T10's attention + head floor is **not optional**: it is what the model is when every FFN neuron is
deleted. If that floor alone already exceeds the 50 tok/s budget, then **no amount of FFN work —
carve, MoE, rank, sparsity, coarser groups, a better kernel — can reach the goal at T10's literal
shape**, and the programme's remaining lever is the shape of attention itself. That is a
different instruction to the architecture than "shrink the FFN", and it is the instruction
`SCALEUP_ARCHITECTURE.md` is waiting on.

**This probe is cheap** (one export, one bench sweep) and it converts a desk claim into a measured
one.

## 1. The question

> **With the FFN reduced to nothing at T10's shape, what does the engine actually read, and how
> does it compare to the 50 tok/s budget?**

## 2. The desk arithmetic this probe exists to check — stated before the measurement

Hand-computed here, in the charged convention, from T10 = `D=4096, QD=4096, KD=1024, F=14336,
L=48, V=32768`:

| term | weights |
|---|---|
| `q` + `o` per layer | 33,554,432 |
| `k` + `v` per layer | 8,388,608 |
| × 48 layers | 2,013,265,920 |
| head `V·D` | 134,217,728 |
| **attention + head floor** | **2,147,483,648** (`2^31`, charged) |

| quantity | value | source |
|---|---|---|
| operative numerator (packed) | 46.16 G-w/s | E28 session, E32 §6 |
| **floor-only rate, predicted** | **21.5 tok/s** | 46.16 ÷ 2.1475 |
| charged budget for 50 tok/s | 0.9232 G | 46.16 ÷ 50 |
| **floor ÷ budget** | **2.33×** | over budget **with zero FFN** |

And in the **moved-byte** convention against E30's ceiling (kept strictly separate — the two
conventions never meet inside a fraction):

| quantity | value |
|---|---|
| attention+head floor, moved | **≈ 1.0786 GB/token** (hand-computed; the runner recomputes it) |
| perfect-kernel floor-only rate | **≈ 33.7 tok/s** (36.30 ÷ 1.0786) |
| moved budget for 50 tok/s | 0.726 GB/token |
| **floor ÷ budget** | **≈ 1.49×** — over budget with a PERFECT kernel and zero FFN |

**Both hand tables are suspect until the runner reproduces them from the file**, because E30 §1.1
caught exactly this kind of hand table short by the `q/k/v` biases. `G-E34B` exists for that.

### 2.1 The residency lever cannot fix this, and the number says why

E30 §5 measured that a 16 MB resident core costs 0.18% of a 20 ms token — free. 16 MB is **32 M
ternary weights**, i.e. **0.016 GB** of the floor's 1.0786 GB. To bring the floor under the
0.726 GB budget you must remove **0.353 GB per token** from DRAM — **22× more than this CPU's
entire L3.** So residency is a real lever and it is *not* a lever on attention at T10's shape.
Registered here so that no later document can quietly use it as one.

## 3. Arms

One export, `--shape T10`, plus arms the engine can make from it:

| arm | what it is | why |
|---|---|---|
| `T10-FLOOR` | `--carve 256 --carve-k 1`: one FFN group of 56 neurons survives | the closest thing to "FFN gone" the container can express; charges the floor + 0.18% |
| `T10-K4` | same file, engine's `--carve-k 4` | E26 measured **17.887 tok/s** in its own session — the planted control |
| `T10-K16` | same file, `--carve-k 16` | E26 measured **15.190** — a second known point |
| `T10-DENSE` | plain packed T10 | the same-session anchor, and the only absolute allowed out |

Every carved arm is **one file through the engine's `--carve-k`**, E26's own design, so no cell can
be confounded by a different file or a different permutation. Five interleaved reps, reps
outermost. Idle box, or the run does not happen — and **the operator (me) runs nothing else while
it runs**, which is the defect that voided E33 run 1.

## 4. The gates

**`G-E34A` — the planted control, from E26's ratios, not its absolutes.** `T10-K16 ÷ T10-DENSE`
must reproduce E26's `ratio_vs_dense = 3.735` within ±10%. E26's absolutes are disowned (E33
addendum §2); its ratios are not.

**`G-E34B` — the hand table must not survive unchecked.** The runner recomputes the charged floor
and the moved floor from the exporter's own layout, independently of §2, and **must reproduce the
artifact's byte count exactly (tolerance 0 bytes)**. If §2's hand numbers disagree with the
runner's, **the runner wins and §2 is reported as wrong, in the result document, with the delta.**

**`G-E34C` — the numerator model must predict a point it did not fit.** Using the operative
numerator, the predicted rate for `T10-K4` (`46.16 ÷ 2.3299 = 19.8 tok/s`) must land within ±15%
of this session's measured `T10-K4`. This is the model that produces every prediction in §2; if
it cannot hit a point E26 already measured, §2's predictions are not quotable.

## 5. The verdict cell, named before the run

**`T10-FLOOR` tok/s**, read against the goal.

| band | name | what it would mean |
|---|---|---|
| **≥ 50** | `FLOOR-CLEARS` | the goal is an FFN problem after all, and every probe since E18 was aimed correctly |
| **25 – 50** | `FLOOR-BINDS` | the FFN is not the whole problem; attention must shrink too, but the goal stays on this box |
| **< 25** | `FLOOR-IS-THE-WALL` | **at T10's literal shape the goal is unreachable on this box even with zero FFN**, and the remaining lever is the shape of attention itself |

## 6. Predictions — fixed here, before the run

1. **`G-E34A`, `G-E34B`, `G-E34C` all fire.**
2. **The verdict lands `FLOOR-IS-THE-WALL`, and `T10-FLOOR` reads `19–23 tok/s`.** The desk model
   says 21.5.
3. **`T10-FLOOR` is only slightly faster than E26's `T10-K4` once the session is normalised.**
   K4 charges 2.3299 G against the floor's 2.1475 — 8.5% apart — so I predict the measured gap is
   **under 12%**. If `T10-FLOOR` comes back dramatically faster than `T10-K4`, the carve machinery
   is costing more than its bytes and E26 §9's owed item reopens.
4. **The moved-byte reading lands `≈ 33–35 tok/s` for a perfect kernel**, i.e. even the ceiling
   arithmetic does not clear 50 with the FFN gone.
5. **Registered as the honest negative**: this probe **cannot** improve any rate. It can only
   establish where the shape breaks. **If it lands `FLOOR-IS-THE-WALL`, that is not a failure of
   the engine branch — it is the engine branch delivering the constraint the architecture needs**,
   and it is the answer to "is 50 tok/s reachable at 10 B on this box" for the *literal* T10
   shape: no, and here is the smallest shape that could be.
6. **The smallest attention that WOULD fit is computed and reported** — the number of layers, or
   the `q/o` rank, or the KV-head count that brings the floor under 0.726 GB moved. Reported as
   desk arithmetic on a measured ceiling, labelled as such, and **not** as a recommendation.

## 7. What E34 will NOT be able to claim

- **Nothing about quality.** Synthetic noise weights; no BPB exists at T10 and none is computed.
  A model with one FFN group is not a model.
- **Nothing about `--lutblk`.** E32 revoked its licence; `donor_engine.c:1437` refuses the LUT
  path on `quant==4` anyway, so every arm here runs packed at E30's measured **0.639** of the
  ceiling — which is the point, since packed is now the operative path.
- **Nothing about a real router.** `k` is oracle-free and fixed; E23 priced a real router and
  that cost is additive.
- **Nothing about other shapes.** T10 only. The floor is a property of a shape, and a different
  10 B (fewer layers, more width, GQA with fewer KV heads) has a different floor — which is
  exactly what prediction 6 is for.
