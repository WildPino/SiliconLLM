# BRIEF E39 — the same ten billion, put somewhere else

**Pre-registered. Pushed before the runner exists and before the exporter can build the object.**
Nothing here may be edited after the push; the result document scores it as written.

**SPEED ONLY, and the weights are noise.** Same limit as E36, in the same place: this prices a
shape, it does not say a model of this shape is any good. §7 says what makes the shape worth
pricing anyway.

---

## 0. The gap this closes

E36 measured the envelope as two numbers and drew the conclusion in its own §7:

```
time per token = 17.2 ms (attention + head + router) + 0.93 ms per carve group
```

> **"100 tok/s at 4096 wide is not an FFN problem, it is an attention problem."**

The base term alone is 17.2 ms of a 10 ms budget. **E34 said the same thing from the other
side** (`FLOOR-IS-THE-WALL`: at T10's literal shape, 50 tok/s is unreachable with the FFN
deleted). Both probes named the attention shape as the remaining lever, and **neither of them
pulled it.** E34 §7 and E36 §9 item 2 have owed this since.

**And at A10B the base is 82% attention projections**: of `822,083,584` charged weights,
`671,088,640` are `q/k/v/o`, `134,217,728` the head, `16,777,216` the router.

## 1. The question

> **Hold the parameter count EXACTLY fixed at ten billion, move weight out of the attention
> projections into the FFN — where the carve makes it nearly free — and ask what the token
> costs.**

Not "make the model smaller". The file must still be a genuine 10 B.

## 2. The object, and why the comparison is unusually clean

`A10B-R512`: `D = 4096, F = 48128, L = 16, NH = 32, NKV = 8, HD = 128, V = 32768`, untied head,
`--carve 256` (group = **188** neurons), and **`q_proj`/`o_proj` written as the FACTORED kind at
rank 512** (`MK_FACTORED`; `k_proj`/`v_proj` stay dense).

| | `A10B` (E36) | `A10B-R512` |
|---|---|---|
| `F` | 46,080 | **48,128** |
| `q`/`o` | dense `4096×4096` | **rank 512** |
| attention per layer | 41,943,040 | **16,777,216** |
| FFN per layer (total) | 566,231,040 | **591,396,864** |
| **per layer, total** | **608,174,080** | **608,174,080** |
| **parameters** | **9,999,220,736** | **9,999,220,736** |

**The two shapes have the same total to the parameter, and the same per-layer total to the
parameter.** They differ in exactly one thing: *where the weight sits*. That makes E36's measured
`A10B` the matched control for this probe rather than a loose reference — and it is already
measured, twice, under opposite arm orders.

**Charged per token**, with `E·D` for the router charged as E26 requires:

```
A10B      charged(k) = 822,083,584 + 35,389,440·k
A10B-R512 charged(k) = 419,430,400 + 36,962,304·k        ->  1.75x cheaper at k=3
```

## 3. What has to change to build it, stated rather than done quietly

**`synth_export.py:233` refuses `--carve` together with `--rank`**, with the comment *"E26 moves
one axis at a time"*. That rule is correct for **pricing an axis** and this probe's entire
question is the **combination**, so the guard is lifted for the combined case and the comment
rewritten to say which probe owns which. **No accounting is loosened**: `active_weights()` gains
a rank term, and `G-E39B` checks it against an independently written closed form at zero
tolerance.

**The engine needs nothing.** In a `quant==4` file `q/k/v/o` are read by the same tagged reader
that handles `MK_FACTORED` (`donor_engine.c:841`, `:763`), and `matvec` dispatches on `m->rank`
at `:448`. Verified by reading the source before writing this brief — the failure mode E37's
addendum A was written about.

## 4. Arms

`A10B-R512` at `k ∈ {1, 2, 3, 4, 6}`, five interleaved reps, **reps outermost**, one warm token
before `--bench`, **and the arm order reversed on a second run** — E36's run-2 rule, which is now
standing procedure and not a per-probe choice.

Control arm: **`A10B` itself**, re-timed in the same session on the same box, so the comparison
never crosses a session boundary.

## 5. The gates

**`G-E39A` — it is still ten billion, and the file says so.** The parameter count recomputed
**from the exported file's own header** must equal **9,999,220,736 exactly** — the same integer
E36 measured — and bytes on disk must match E1's independently written v4 layout. Zero tolerance.
A shape that got faster by getting smaller answers a different question.

**`G-E39B` — charged accounting, zero tolerance.** At every `k`, `synth_export.active_weights()`
applied to the artifact's header must equal `419,430,400 + 36,962,304·k`.

**`G-E39C` — the planted control, and it must FIRE before any speed number here counts.**
Export the same shape at **rank 4096**, where the factored form moves *more* weights than the
dense one (`2·r·(QO+D) = 67,108,864` against `2·QO·D = 33,554,432`). It must read **SLOWER** than
the dense-`q/o` control. **If a rank that costs more does not cost more, the timing is not
measuring the factored path at all** and every number below it is about something else.

## 6. The verdict cell and the bands, named before the run

**`A10B-R512` at `k = 3`**, tok/s, against `A10B`'s measured **49.96**.

| band | name | what it would mean |
|---|---|---|
| **≥ 100** | `TEN-B-AT-HUNDRED` | the excellent target, reached at a genuine 10 B |
| **80 – 100** | `TEN-B-NEAR-HUNDRED` | the attention lever is real and large; 100 is a `k` away |
| **55 – 80** | `RANK-BUYS-SPEED` | real, smaller than the weight count promises |
| **< 55** | `RANK-DOES-NOT-BUY-SPEED` | the factored form costs back what it saves, and E36 §7's reading is wrong |

Reported alongside, and worth more than the cell: **`k*`, the crossing where `A10B-R512` reads
100 tok/s**, measured twice under opposite arm orders — E36's crossing agreed to one neuron in
forty-six thousand and was the number that survived.

## 7. Predictions — fixed here, before the run

1. **All three gates fire.**
2. **`TEN-B-NEAR-HUNDRED`, and specifically `84.8 tok/s` at `k = 3`.** From E36's two-term model
   with no new parameters: `419,430,400 / 47.254 G-w/s = 8.88 ms` base plus three groups at
   `36,962,304 / 38 G-w/s = 0.97 ms`. I am registering the number, not just the band.
3. **`k* ≈ 1.0`** — 100 tok/s costs about one carve group of 188 neurons, i.e. `0.39%` activation.
4. **E36's two-term model transfers to a factored-attention shape**: base at the full envelope
   rate (47–48 G-w/s), marginal group at ~0.80 of it. **If the base comes in slower, the factored
   matvec has a cost per weight that E25's rank-invariance did not see** — E25 measured the rank
   axis inside a `1.54%` band, and that is the prior this prediction leans on.
5. **`A10B` re-timed in this session lands within 5% of 49.96**, or the session is not comparable
   to E36's and the probe says so instead of reporting a ratio.

## 8. What E39 will NOT be able to claim

- **Nothing about quality.** Synthetic weights. A rank-512 ternary `q/o` is *not* free: E22
  measured it collapsing to `28/160` post-hoc.
- **But it is no longer hopeless, and that is why this shape is worth pricing.** **H0 measured
  that exact object trained back to `111/160`** teacher-forced and to `+0.058` BPB over the
  intact donor — at 1.5 B, `q/o` only, which is precisely the organ E39 factors. **E39 prices the
  shape H0 showed can be trained.** That is the first time this programme's speed side and its
  quality side have pointed at the same object, and it is a reason to measure, not a result.
- **Nothing about the FFN's extra capacity.** `F` grows from 46,080 to 48,128 and at `k = 3`
  only 564 of those neurons are ever read. Whether the extra 2,048 per layer are *useful* is a
  training question this probe does not touch.
- **Nothing about 12 k context.** Same as every speed probe since E30: short context, and E32 §8's
  context-ladder hypothesis is still owed.
