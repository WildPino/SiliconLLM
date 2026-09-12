# BRIEF E40 — after every attention lever is pulled, how much FFN can 50 tok/s buy at ten billion?

**Pre-registered. Pushed before the runner exists and before the exporter can build the objects.**
Nothing here may be edited after the push; the result document scores it as written.

**SPEED ONLY, and the weights are noise.** Same limit as E36 and E39, in the same place.

---

## 0. Why this probe, and why now

E39 moved the attention shape and the floor fell from 17.2 ms to 10.0 ms — **99.7 tok/s with the
FFN at zero, at a genuine 9,999,220,736 parameters.** That is the speed half of the goal standing
on the excellent target.

**But the goal is not a shape, it is a working model**, and the two halves now have to be held
in one sentence:

| what it costs | what it is worth |
|---|---|
| **E39**: at 10 B, 50 tok/s buys **4.25–4.78%** FFN activation (measured) | **E38**: at 6.25% activation the best attainable selector reads **3.597 BPB** against dense **0.768** and chance **4.070** — 86% of the way to chance |
| **E39**: 100 tok/s buys ≈ **0%** | **E38**: the value of selection **peaks at 25%** and the goal's rate is far below it |
| | **E37**: post-hoc conversion at 1.17% reads **4.029398** — below the chance line |

So the question that decides where every remaining GPU-hour goes is not "can the engine go fast"
— E39 answered that — but:

> **Is there ANY attention shape at ten billion parameters on this box for which 50 tok/s buys an
> activation fraction that E38 says is worth selecting at?**

If yes, post-hoc conversion is still alive at some shape. **If no, then the only route left to
the goal is TRAINING INTO the format at whatever fraction the box affords**, and the T4 clause in
the standing goal should be spent on that and nothing else.

## 1. Why a desk model will not settle it

I can compute the answer in four lines, and the computation currently says **~6.5%**. I am not
allowed to stop there, and E39 is the reason: its predictions scored **1 HIT / 4 MISS**, every
miss from a desk model that did not survive a change of matvec *kind*. This probe changes
matvec kinds again — `NKV` 8 → 2 reshapes `k/v`, rank 512 → 256 halves the factored inner
dimension. **The arithmetic is registered here as a prediction so the measurement can contradict
it.**

## 2. The objects — all three exactly 9,999,220,736 parameters

Solved for `F` so each shape hits E36's integer exactly, with `F % 256 == 0` so the carve is
expressible. `D = 4096, L = 16, NH = 32, HD = 128, V = 32768, --carve 256` throughout.

| arm | `F` | `NKV` | head | rank on `q/o` | role |
|---|---|---|---|---|---|
| **`R512`** | 48,128 | 8 | untied | 512 | **E39's own artifact, re-timed: the control** |
| **`NKV2`** | 48,640 | **2** | untied | 512 | **one lever**: `k/v` heads 8 → 2, nothing else moves |
| **`ALL`** | 49,664 | **2** | **tied** | **256** | **every lever at once** |

`ALL` is the cheapest attention this programme can express at ten billion parameters without
touching `D`, `L` or `HD`.

## 3. The verdict cell and the bands, named before the run

**The cell is `ALL`'s measured FFN activation fraction at 50 tok/s** — `k*₅₀` converted to a
fraction of `F` — taken from the two-term fit exactly as E36 and E39 took theirs.

| band | name | what it would mean |
|---|---|---|
| **≥ 25%** | `SPEED-BUYS-THE-PEAK` | E38's peak is affordable; post-hoc conversion is alive again |
| **10 – 25%** | `HALFWAY-TO-THE-PEAK` | worth one more shape probe before spending GPU |
| **6 – 10%** | `LEVERS-NEARLY-EXHAUSTED` | the attention axis is nearly spent |
| **< 6%** | **`ATTENTION-LEVERS-EXHAUSTED`** | **no attention shape at 10 B makes selection affordable on this box; training into the format is the only route left to the goal** |

Reported alongside and worth more than the cell, as in E39: **the floor** (the base term of the
fit), which is the quantity the two arm orders agreed on to 0.62% last time.

## 4. Arms and protocol

Each arm at `k ∈ {1, 2, 3, 4, 6}`, **5 interleaved reps, reps outermost**, one warm token before
`--bench`, `--threads 6`, `donor_engine_e26.exe`. **Second run with the arm order reversed** —
E36's standing rule, and **E36's run-2 rule governs the cell**: run 1 is the registered
measurement, run 2 may not promote it, and if the slopes disagree beyond the reps' own dispersion
the cell is declared unresolvable. E39 §2 is the worked precedent, including that it is applied
in the direction that goes against me.

**A session whose `R512` control drifts more than ±5% from E39's own 78.46 is reported as not
comparable and yields no cross-session ratio** — E39 prediction 5 caught exactly this and the
rule is inherited verbatim.

## 5. The gates

**`G-E40A` — it is still ten billion, three times over.** Each arm's parameter count, recomputed
**from its own exported header**, must equal **9,999,220,736** exactly, and bytes on disk must
match E1's independently written v4 layout. Zero tolerance.

**`G-E40B` — charged accounting, zero tolerance.** For each arm, `synth_export.active_weights()`
on the file's header must equal a closed form written independently in the runner from the
shape's own dimensions.

**`G-E40C` — the planted control, and it must FIRE before any speed number counts.** `NKV2`
charges **strictly less per token than `R512` at every `k`** (its `k/v` are a quarter the size and
nothing else differs). **It must therefore be FASTER at every `k`.** If a shape that moves less
weight does not go faster, the stopwatch is not on the shape and nothing below it counts.

## 6. Predictions — fixed here, before the run

1. **All three gates fire.**
2. **`ATTENTION-LEVERS-EXHAUSTED`, and specifically `6.50%`** for `ALL` at 50 tok/s. From E39's
   two-term constants with no new parameters (base 42.0 G-w/s, group 45.2 G-w/s).
3. **The floors: `NKV2` 131.8 tok/s, `ALL` 166.9 tok/s**, against `R512`'s measured ~100. I am
   registering the numbers, not just the ordering.
4. **The factored penalty GROWS at rank 256.** E39 measured 4–6% fewer charged weights per second
   on the factored path and attributed it to matvec calls going 113 → 145. Rank 256 keeps the
   call count and halves the work per call, so **`ALL`'s base rate should come in BELOW `R512`'s
   41.96 G-w/s** — I predict **39–41**. If it does not, the penalty is not call overhead and
   E39 §8 item 2 is answered in the other direction.
5. **100 tok/s stays out of reach with any FFN at all**: `ALL`'s `k*` for 100 tok/s lands in
   `(0, 2]` groups, i.e. under 1% activation — fast enough to clear 100 only by giving up the FFN
   almost entirely.

## 7. What E40 will NOT be able to claim

- **Nothing about quality.** Synthetic weights, again and by construction. This probe's *entire*
  purpose is to price the shape so the quality question can be asked at the right activation
  fraction.
- **Nothing about `D`, `L` or `HD`.** Three levers stay untouched. A `<6%` result closes the
  levers *this probe pulled*, not every conceivable shape.
- **Nothing about whether ~6% can be TRAINED into.** That is the question this hands to the T4
  clause, and E40 cannot answer it on a CPU with noise for weights.
- **One box, one thread count, one engine build.**
