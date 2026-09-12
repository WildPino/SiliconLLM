# E34 — what does T10 read with the FFN GONE? The floor under the floor

**Verdict: `FLOOR-IS-THE-WALL`.** `T10-FLOOR` reads **20.03 tok/s** — **2.50× short of the goal
with the FFN deleted.** The attention + head floor alone is **2,147,483,648 charged weights** and
**1,078,624,256 moved bytes** per token; against this session's numerator that is **2.297× the
charged budget for 50 tok/s**, and against E30's measured ceiling **a perfect kernel on that floor
reads 33.65 tok/s** — still **1.486×** short with no FFN at all.

**So at T10's literal shape, 50 tok/s is not reachable on this box by any amount of FFN work.**
Every probe from E18 to E33 attacked the FFN because it is nine tenths of the weight. The
remaining tenth does not fit the budget either. **The lever that is left is the shape of
attention** — depth, `q/o` rank, KV heads — which is a different instruction to the architecture
than "shrink the FFN".

**Brief**: `briefs/BRIEF_E34_THE_FLOOR_UNDER_THE_FLOOR.md`, pushed at `684e677` before the runner
existed. **Runner**: `engine/e34_floor_under_floor.py`. **Result**:
`engine/results/e34_floor_under_floor.json`, 205 s, four arms, five interleaved reps.
**Run 1 VOID and kept** (`results/e34_floor_under_floor_void_run1.json`) — see §5.

---

## 1. The gates

| gate | what it demanded | reading | |
|---|---|---|---|
| **`G-E34A`** the planted control | `T10-K16 ÷ T10-DENSE` within ±10% of E26's `ratio_vs_dense = 3.735` — E26's **ratio**, never its absolute (E33 addendum §2) | **3.601**, `−3.6%` | **FIRES** |
| **`G-E34B`** the hand table must not survive unchecked | the runner recomputes both floors independently and the exporter must agree exactly | charged **2,147,483,648** — the brief's §2 hand table **agrees to the byte**; moved **1,078,624,256** vs the hand `1.0786 GB`, **+0.002%** (rounding); exporter and runner agree on `2,230,845,440` at `k=1` | **FIRES** |
| **`G-E34C`** the model must predict a point it did not fit | this session's numerator must predict E26's already-measured `T10-K4` within ±15% | numerator **46.74 G-w/s** → predicts **20.06**, measured **19.03**, **+5.4%** | **FIRES** |

`G-E34C` is the load-bearing one and it is worth stating plainly: **the numerator measured here
(46.74 G-w/s) lands within 1.3% of E28's independently measured packed numerator (46.16 G-w/s)**,
in a different session, from a different file, after E32 made packed the operative path. The
model that produces every prediction below is not fitted to the points it predicts.

**Contention**: pre-run witness **10.5% mean / 22% peak** against a 25% bar; mid-run witnesses
**2.0 / 10.5 / 2.0 / 3.0 / 13.0%**, all under it. The operator ran nothing else, which is E33 §6's
lesson applied.

## 2. The measurement

One carved artifact (`E=256`) through the engine's `--carve-k`, plus E26's own dense control as a
**separate file** — E26's design exactly, and the thing run 1 got wrong.

| arm | `k` | charged weights/token | mean tok/s | median | spread | charged G-w/s |
|---|---|---|---|---|---|---|
| `T10-DENSE` | — | 10.6032 G | **4.408** | 4.400 | 5.0% | 46.74 |
| `T10-K16` | 16 | 2.7263 G | 15.872 | 15.940 | 4.7% | 43.27 |
| `T10-K4` | 4 | 2.3299 G | 19.026 | 18.850 | 7.3% | 44.33 |
| **`T10-FLOOR`** | **1** | **2.2308 G** | **20.030** | 19.660 | 11.4% | 44.68 |

`T10-DENSE` at 4.408 agrees with E28's 4.353, E30's 4.363 and E26's 4.067 — the same box reading
the same shape across four sessions.

`T10-FLOOR` charges 2.2308 G against the pure floor's 2.1475 G: a **3.882% residue** (the
50.3 M/token router, which is charged, plus the one surviving FFN group of 56 neurons). **The
measured 20.03 tok/s is therefore a slight UNDER-statement of a true zero-FFN floor**, by about
that much — which makes the verdict conservative, not generous.

## 3. What the floor costs, in both conventions kept apart

| charged convention | |
|---|---|
| numerator, this session | **46.74 G-w/s** |
| charged budget for 50 tok/s | **0.9348 G/token** |
| attention + head floor | **2.1475 G/token** |
| **floor ÷ budget** | **2.297× — with zero FFN** |

| moved convention | |
|---|---|
| ceiling (E30, measured on this box) | **36.30 GB/s** |
| moved budget for 50 tok/s | **0.7260 GB/token** |
| attention + head floor, moved | **1.0786 GB/token** |
| **perfect-kernel rate on the floor alone** | **33.65 tok/s** |
| **floor ÷ budget** | **1.486× — with zero FFN and a perfect kernel** |

The two conventions never meet inside a fraction. **The charged reading says the current engine
is 2.50× short on the floor; the moved reading says even a flawless engine is 1.49× short.** The
second is the one that cannot be engineered away on this machine.

### 3.1 Residency cannot close it, and the number says why

E30 §5 measured a 16 MB resident core at 0.18% of a 20 ms token — free. 16 MB is **0.016 GB** of
the floor's **1.0786 GB**, while the gap to be closed is **0.353 GB**: **22× the entire L3.**
Registered in the brief before the run so that no later document can quietly use residency as a
lever on attention at this shape. It remains a real lever — on a *small* resident core, which is
what `SCALEUP_ARCHITECTURE.md` describes — and it is not one here.

## 4. Predictions — 5 HIT / 0 MISS, the first clean sheet in this branch

| # | registered | outcome | |
|---|---|---|---|
| 1 | all three gates fire | `A` `−3.6%`, `B` exact, `C` `+5.4%` | **HIT** (run 2; run 1 VOID, §5) |
| 2 | `FLOOR-IS-THE-WALL`, `T10-FLOOR` reads **19–23 tok/s** (desk: 21.5) | **20.03** | **HIT** |
| 3 | `T10-FLOOR` within **12%** of `T10-K4` | **+5.3%** | **HIT** |
| 4 | perfect-kernel reading lands **33–35 tok/s** | **33.65** | **HIT** |
| 5 | this probe cannot improve any rate; it can only locate where the shape breaks | holds | **HIT** |
| 6 | the smallest attention that would fit is computed and reported as desk arithmetic | §4.1 | reported |

A clean sheet is not a virtue here — it means this probe was mostly confirmatory arithmetic, and
its value is in the gates, not in my prediction record. The two probes today where I was **wrong**
(E32, E33) moved the programme further than this one did.

### 4.1 Prediction 6 — the smallest attention that fits, as desk arithmetic on a measured ceiling

**Not a recommendation.** At T10's width, with the FFN at **zero** and a **perfect** kernel:

* attention, per layer, moved: **21,069,824 B**
* head, moved: **67,239,936 B**
* moved budget at 50 tok/s: **726,000,000 B**
* → **31 attention layers of the 48 fit. 65% of the depth.**

And that is the *ceiling* case. At the operative numerator the same arithmetic is far harsher, and
a real model needs its FFN back. **The honest shape of the instruction is: at this width, a 10 B
that runs at 50 tok/s on this box has roughly two thirds of T10's depth at most, before the FFN
gets any budget at all** — or it keeps the depth and pays for it with width, rank, or KV heads.
E27 already measured that depth can be cut cheaply *if the right layers go* (`L21-MINRES` 113/160
against `L21-LAST` 57/160 at identical cost), and E29 confirmed the ordering with a proper
control. **This is the first time the engine side has said how much depth the budget actually
buys.**

## 5. Run 1 was VOID, and the controls caught an error I would not have caught

`ARMS` listed `T10-DENSE` with no `--carve-k`, on the carved file. **An un-flagged run of a carved
file uses the `k` stored in the file — 1 here — so the "dense" arm was the FLOOR arm wearing the
dense arm's label and the dense arm's 10.6 G charge.** Both gates fired on it:

* `G-E34A` read **0.783** against E26's `3.735` — off by 79%.
* `G-E34C`'s numerator came out at **217.62 G-w/s**, 4.7× the machine's entire measured
  throughput, predicting `T10-K4` at 93.40 against 19.86 measured.

Either number should have been obvious to me on sight; the point of a planted control is that it
does not depend on my noticing. **The void run's `T10-FLOOR` arm landed at 21.27 tok/s — close to
the desk prediction — and is not quoted as evidence**, because the anchors did not fire, and a
number that agrees with my prediction is the one I should be least willing to keep from a void
run. Kept at `results/e34_floor_under_floor_void_run1.json`; the cause is written into the code at
the line that caused it.

### 5.1 A unit slip in my own reporting, corrected on recorded values

The runner printed `floor is 2297319028.69x the charged budget`. `floor_charged` is in **weights**
and `charged_budget_for_50` is in **G-weights**, so the ratio was `1e9` too large — **the unit
hiding in the denominator, which is exactly what this programme's byte-convention law is about.**
The printed rate numbers were unaffected. Fixed in the runner and recomputed from the recorded
values (no re-run); the JSON carries the erratum and the original value.

## 6. What E34 cannot claim

- **Nothing about quality.** Synthetic noise weights; a model with one FFN group is not a model,
  no BPB exists at T10 and none was computed.
- **Nothing about other shapes.** T10 only. A different 10 B — fewer layers, more width, GQA with
  fewer KV heads, MLA — has a different floor, which is precisely what §4.1 is for and is
  **not** measured here.
- **Nothing about `--lutblk`.** E32 revoked its licence and `donor_engine.c:1437` refuses the LUT
  path on `quant==4` anyway; every arm ran packed, the operative path.
- **Nothing about the KV cache.** These are 40-token benches; the floor measured here is the
  WEIGHT floor. KV traffic at long context is additive and still owed (E30 §8 item 2).
- **Nothing about a real router.** `k` is fixed and oracle-free; E23's router cost is additive.

## 7. Owed after E34

1. **The attention-shape sweep is now the main line.** Depth, `q/o` rank, KV-head count, at T10's
   width, measured the way E34 measured this — with the FFN held at `k=1` so the floor is what
   moves. §4.1's "31 of 48" is desk arithmetic and needs the same treatment this probe just gave
   the brief's own hand table.
2. **Time `--lut --lut-group 32`** (E32 §8 item 1). Unchanged, and still the cheapest way to buy
   back numerator.
3. **KV traffic at long context** (E30 §8 item 2) — the floor here is weights only, and the goal
   is not a 40-token benchmark.
4. **`PT_BLK`** (E33 §7 item 2) and the T10 non-result (E33 §7 item 3) are unaffected by this
   probe and still open.
