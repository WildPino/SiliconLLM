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

---

# ADDENDUM A — `G-E40A` went VOID on the `ALL` arm, and the arm is replaced

**Pushed before the replacement object exists and before any timing.** The original brief above
is unedited; this addendum is additive and is scored alongside it.

## What fired

`G-E40A` requires every arm's parameter count, recomputed **from its own exported header**, to
equal 9,999,220,736 exactly. As built:

```
G-E40A  ALL 10133438464 == NKV2 9999220736 == R512 9999220736 == 9999220736 ?  -> VOID
```

`ALL` is over by **134,217,728 = V·D exactly**, i.e. one whole head.

## Why — and it is a rule I should have read first, not a bug

`synth_export.py:207` carries a comment that predates this probe by many experiments:

> **RUN 1 WROTE THE WRONG ARM.** SHAPES carries the donor's own `tied` flag, and a tied model runs
> its head as the fp32 **EMBEDDING**: 544 MB/token at S05, 13.8 ms, **52% of the token**. […] the
> runnable configuration is UNTIED and packed — 68.1 MB/token, the configuration
> `SPEED_LEDGER §12.2` actually measured its 56.1 tok/s on.

So `--head ternary`, the default and the only head any speed number in this programme was ever
measured on, **forces `tied = 0`**. The exporter honoured that and wrote an untied head; the
object is therefore a real 10.13 B and not the ten billion the gate demands.

**My §2 was wrong twice about tying, in opposite directions.** First, tying saves *parameters*,
never *charged weights* — the output projection is read every token either way (I caught that one
before the brief was written and the brief's table is already built on the corrected version).
Second, and not caught: at this engine's runnable head, tying is not a lever at all — **it is a
5× slowdown on the head**, which is why the exporter refuses it. **A lever that costs speed had no
business in an arm called `ALL`.**

## The replacement arm

`ALL` is withdrawn and replaced by **`R128`**, which pulls every lever that is *real* on this
engine — `k/v` heads 8 → 2 and the `q/o` rank down to 128 — and hits the integer exactly:

| arm | `F` | `NKV` | head | rank on `q/o` | base charged | total |
|---|---|---|---|---|---|---|
| `R512` | 48,128 | 8 | untied | 512 | 0.419 G | 9,999,220,736 |
| `NKV2` | 48,640 | **2** | untied | 512 | 0.319 G | 9,999,220,736 |
| **`R128`** | **49,152** | **2** | untied | **128** | **0.218 G** | **9,999,220,736** |

`R128` is a *stronger* arm than the withdrawn `ALL` (base 0.218 G against 0.252 G), so this
replacement makes the registered prediction **harder to miss in the direction I predicted**, not
easier. That is the only direction in which a mid-probe substitution is defensible, and the
numbers above are on the record before the object exists.

## What this does to the registered predictions

- **Prediction 1 (all gates fire) is already a MISS** and is scored as one. `G-E40A` went VOID on
  the first build. It is not re-run to a pass — it fired, that is what it is for, and the arm it
  disqualified is gone.
- **Prediction 2's band and number stand as written** (`ATTENTION-LEVERS-EXHAUSTED`, `6.50%`) and
  are now scored on `R128`. Recomputed on the replacement arm the same desk model says **7.21%**;
  **the registered 6.50% is what gets scored**, because the whole point is that the number was
  fixed before the measurement.
- **Prediction 3's floors**: `NKV2` 131.8 tok/s stands unchanged. `ALL`'s 166.9 is withdrawn with
  the arm; the desk model puts `R128` at **192.7 tok/s**, registered here.
- **Predictions 4 and 5** are scored on `R128` in place of `ALL`. Prediction 4 said rank 256 would
  push the base rate to 39–41 G-w/s; **rank 128 is a stronger form of the same prediction and it
  is left at 39–41 rather than widened.**
