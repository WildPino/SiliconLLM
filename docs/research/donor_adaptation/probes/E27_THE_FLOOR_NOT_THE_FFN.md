# E27 — the floor, not the FFN

**Brief**: `briefs/BRIEF_E27_THE_FLOOR_NOT_THE_FFN.md`, pushed at `808ba43` **before this runner
existed**.
**Code**: `ternary/e27_floor.py`, committed at `f652688`.
**Result**: `engine/results/e27_floor.json` (`engine/results/e27_floor_smoke.json` for the smoke).
**Cost**: 4,345 s, CPU only, one job, twelve arms. `VOID: none`. No timing taken, nothing exported,
no carve and no router anywhere in this probe.

## VERDICT — `FLOOR-IS-NOT-ENOUGH`

E24 ended by naming the lever: at the goal's shape the FFN is nine tenths of the weight, but the
**floor** — attention, the head, and whatever router sits on top — is what decides how much FFN the
budget can still afford. E27 attacked the floor on both of its axes, **width** (rank on `q/o` and on
`k/v`) and **depth** (dropping whole layers), with no carve and no router to hide behind.

**Neither axis opens the goal's shape, and the reason is that the two axes are anti-correlated with
quality in exactly the region where the budget starts to be satisfied.** Every arm that fits the
`0.998 G`-per-token budget is qualitatively broken; every arm that survives qualitatively leaves the
budget a factor of 5–8 out of reach.

| arm | `r_qo` | `r_kv` | `L` | BPB | free | **teacher-forced** | donor active | `T10` floor | `k` of 256 | band |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | — | — | 28 | 0.767595 | 160/160 | **160/160** | 1.5436 G | 2.1978 G | −36.3 | CHEAPER |
| `L28-PASSTHROUGH` | — | — | 28 | 0.767595 | 160/160 | **160/160** | 1.5436 G | 2.1978 G | −36.3 | CHEAPER |
| `QO-512` | 512 | — | 28 | 0.820284 | 26/160 | **144/160** | 1.4995 G | 1.6607 G | −20.1 | CHEAPER |
| `QO-192` | 192 | — | 28 | 1.856378 | 4/160 | 56/160 | 1.4445 G | 0.9899 G | **+0.2** | WORSE |
| `QO-96` | 96 | — | 28 | 2.097275 | 1/160 | 42/160 | 1.4280 G | 0.7885 G | **+6.3** | WORSE |
| `KV-96` | — | 96 | 28 | 1.502497 | 12/160 | 84/160 | 1.5312 G | 1.9210 G | −27.9 | WORSE |
| `QO192+KV96` | 192 | 96 | 28 | 2.013923 | 9/160 | 42/160 | 1.4321 G | 0.7130 G | **+8.6** | WORSE |
| `L24-LAST` | — | — | 24 | 1.916526 | 2/160 | 98/160 | 1.3564 G | 1.8969 G | −31.9 | WORSE |
| `L21-LAST` | — | — | 21 | 2.500083 | 1/160 | 57/160 | 1.2160 G | 1.6819 G | −27.6 | WORSE |
| **`L21-MINRES`** | — | — | 21 | **0.993446** | 14/160 | **113/160** | 1.2160 G | 1.6819 G | −27.6 | **COMPARABLE** |
| `L14-MINRES` | — | — | 14 | 1.824140 | 9/160 | 54/160 | 0.8885 G | 1.1660 G | −10.2 | WORSE |
| `FLOOR-MIN` | 192 | 96 | 21 | 2.192089 | 5/160 | 38/160 | 1.1324 G | 0.5683 G | **+17.3** | WORSE |

`k of 256` is how many FFN groups of 256 the `T10` budget still permits **after** that arm's floor
is paid. A negative number means the floor alone has already eaten the whole per-token budget, with
nothing left for any FFN at all.

**The one real positive result is not on the budget axis at all: `L21-MINRES` reads `113/160`
teacher-forced against `L21-LAST`'s `57/160` at exactly the same depth, the same parameter count and
the same charged cost — a gain of 56 tokens from choosing *which* seven layers to drop.** That is
four times the registered 0–15 band (§5).

---

## §1 — the gates

| gate | what it demanded | reading | fires |
|---|---|---|---|
| `G-F0` | the eval slice is the frozen one, sha `a1a48dc9…`, 51,870 scored bytes, 0 rejected | sha matches, 51,870, 0 rejected | **yes** |
| `G-F1` | `L28-PASSTHROUGH` — the depth machinery run with `n=0` — must be **bit-identical** to `base` | BPB `0.7675949641196624` vs `0.7675949641196625`, free 160/160 both, tf 160/160 both | **yes** |
| `G-F2` | the `QO-512` anchor must reproduce E21's `QO-ACT-512` on disk | `bpb 0.8202837636996289`, `free 26`, `tf 144`, all three match `e21_rank.json` exactly | **yes** |
| `G-F3` | the `T10` floor arithmetic must be recomputed from the shape, not asserted: `r_qo=512, r_kv=256, L=48` ⇒ `0.7130 G` | `0.71303168 G`, `ok: true` | **yes** |
| `G-F4` | a verdict must be issued from the table, not chosen | `FLOOR-IS-NOT-ENOUGH`, best COMPARABLE arm `QO-512`, which permits `k = −20.1` | **yes** |

`G-F1` is the important one. The depth code reindexes `self_attn.layer_idx` and rewrites
`config.num_hidden_layers`; getting either wrong is a **silent wrong answer and not an error**, so
the passthrough arm exists purely to prove the machinery is inert when it is told to drop nothing.
It is inert to the last bit.

`VOID: none` — no arm was discarded, no arm was rerun.

---

## §2 — width: the rank fraction does not transfer across width

E21 found rank 512 on `q/o` nearly free at `D = 1536` (`144/160`, one of the best results in the
programme). The obvious hope was that `r/D` is the invariant, so that the same *fraction* would be
near-free at `D = 4096`. E27 tested the fraction directly by walking `r/D` down at the donor's own
width:

| `r` | `r/D` | teacher-forced | BPB |
|---|---|---|---|
| 512 | 1/3 | **144/160** | 0.820284 |
| 192 | 1/8 | 56/160 | 1.856378 |
| 96 | 1/16 | 42/160 | 2.097275 |

**The curve is a cliff, not a slope.** From `1/3` to `1/8` the model loses 88 of 160 tokens and its
BPB more than doubles. There is no smooth fraction to carry to another width; `r = 512` at `D = 1536`
is a *specific* number that happens to be above this donor's attention rank, and nothing in this
table licenses transposing `1/3` to `r = 1365` at `D = 4096`.

This closes, negatively, the item E21 §8 left open.

### `k/v` resist rank much harder than `q/o`

`KV-96` reads `84/160`. That is `r/D = 1/16` on `k/v` against `42/160` for the same fraction on
`q/o` — so in *relative* terms `k/v` tolerate rank better. But the arithmetic that matters is
absolute, and it runs the other way:

- `k/v` at `T10` are `[1024, 4096]`, **not square**. A rank-`r` factorisation is charged
  `r · (1024 + 4096)`; the dense projection costs `1024 · 4096`. Rank therefore **saves on `k/v`
  only if `r < 819`**, and at `r = 96` it saves `1.9210 → ` almost nothing in donor terms
  (`1.5436 → 1.5312 G`, 0.8%).
- The price for that 0.8% is 76 teacher-forced tokens (`160 → 84`).

So `k/v` are the worst trade on the board: they are the smallest slice of the floor and they are the
most expensive to touch. **Leaving `k/v` in fp32 through E21–E24 was not favourable bookkeeping — it
was avoiding a 76-token cut for a rounding error of savings.** That is the honest reading of what
looked, at the time, like an arbitrary choice.

`QO192+KV96` composes the two and reads `42/160` — the same as `QO-96` alone. The damage does not
add; it saturates, because by then the model is already broken.

---

## §3 — depth: *which* layers, not *how many*

This is the part of E27 that produced something usable.

`MINRES` was defined in the brief, before the run: for each layer, the mean over calibration
positions of `‖block_out − block_in‖ / ‖block_in‖`, and drop the `n` smallest. The measured profile
over the donor's 28 layers:

```
L  0  19.0500   <- the first block dominates; it builds the representation
L  1   1.1846
L  2   0.6431
L  3   0.5442
L  4   0.5085
L  5   0.4806
L  6   0.4744
L  7   0.4534
L  8   0.4108
L  9   0.3834
L 10   0.3778
L 11   0.3706
L 12   0.3282   <-+
L 13   0.3260     |
L 14   0.2903     |  the seven smallest are a CONTIGUOUS MIDDLE BAND
L 15   0.2836     |  and this is what L21-MINRES drops
L 16   0.2882     |
L 17   0.3423     |
L 18   0.3637   <-+
L 19   0.4436
L 20   0.3978
L 21   0.5030
L 22   0.4675
L 23   0.4461
L 24   0.4051
L 25   0.3825
L 26   0.3973
L 27   0.7618   <- the last block rises again
```

**The profile is not monotone.** It falls to a minimum at layers 14–16 and rises again towards the
output. Dropping "the last `n`" — the default move, and the one `L21-LAST` makes — therefore removes
some of the *most* active blocks in the stack, including layer 27 which has the second-highest
residual of all 28.

| arm | layers dropped | teacher-forced | BPB |
|---|---|---|---|
| `L24-LAST` | 24–27 | 98/160 | 1.916526 |
| `L21-LAST` | 21–27 | 57/160 | 2.500083 |
| **`L21-MINRES`** | **12–18** | **113/160** | **0.993446** |
| `L14-MINRES` | 9–18, 20, 24–26 | 54/160 | 1.824140 |

**`L21-MINRES` beats `L21-LAST` by 56 teacher-forced tokens and by 1.51 BPB at identical cost.** Same
21 layers, same 1.2160 G active, same everything except the choice of which seven to remove. It is
the only non-`base` arm in E27 that lands COMPARABLE, and it does so at 75% of the donor's layers
with **no carve, no router, no rank, and no retraining of any kind.**

Depth is not free, though. `L14-MINRES` at half the layers collapses to `54/160`. The usable range
here is narrow: 75% of the stack survives, 50% does not, and E27 did not bisect between them.

---

## §4 — what this means at the goal's shape

The budget arithmetic is the point of the probe, so it is stated in full rather than summarised.

`T10` is the goal's "es 10B" shape: `D=4096, QD=4096, KD=1024, F=14336, L=48, V=32768`. E25 measured
the engine's charged throughput at that shape as **49.9 G active weights/s**, held inside 1.54%
across arms that differ by 15%. 50 tok/s therefore means **≤ 0.998 G active weights per token**.

With the FFN left **dense** — i.e. asking only "what does the floor lever buy on its own?":

| arm | `T10` floor | `T10` FFN kept | total | implied tok/s |
|---|---|---|---|---|
| `base` | 2.1978 G | 8.4557 G | 10.6535 G | **4.68** |
| `QO-512` | 1.6607 G | 8.4557 G | 10.1164 G | **4.93** |
| `QO192+KV96` | 0.7130 G | 8.4557 G | 9.1687 G | **5.44** |
| `L24-LAST` | 1.8969 G | 6.1908 G | 8.0877 G | **6.17** |
| `L21-MINRES` | 1.6819 G | 4.7563 G | 6.4383 G | **7.75** |
| `FLOOR-MIN` | 0.5683 G | 4.7563 G | 5.3247 G | **9.37** |
| `L14-MINRES` | 1.1660 G | 2.1139 G | 3.2799 G | **15.21** |

**The best qualitatively-COMPARABLE configuration E27 found runs the goal's shape at 7.75 tok/s.**
The only arm that clears 10 tok/s is `L14-MINRES`, which reads `54/160` and is not a model. Even
`FLOOR-MIN` — rank 192 on `q/o`, rank 96 on `k/v`, 36 of 48 layers, the most aggressive floor in the
probe — is `9.37 tok/s` with a dense FFN, and reads `38/160`.

To reach 50 tok/s from `L21-MINRES` the FFN would have to fall from `4.7563 G` to **at most
`0.998 − 1.6819 = −0.684 G`**, which is not a number: *the floor alone is 1.68× the entire budget.*
That is the verdict in one line. **At 36 layers with dense `q/o`, deleting the FFN completely still
does not reach 50 tok/s** — it reaches 29.7 tok/s.

And the composition does not rescue it. `FLOOR-MIN` is the arm built specifically to be the cheapest
floor E27 could assemble, and it is the only arm with real room for FFN (`k = 17.3` of 256, 6.8%
activation). It reads `38/160`. **The configuration that fits the budget is the configuration that
does not work.**

---

## §5 — the registered predictions, scored

The brief made five predictions and registered one alternative. They were written before the run and
are scored here without editing.

| # | prediction | outcome |
|---|---|---|
| 1 | `QO-192` and `QO-96` degrade sharply; the rank fraction does **not** transfer | **HIT** — 56 and 42 tokens |
| 2 | `KV-96` is roughly as cheap as `QO-96` in quality terms | **MISS** — `KV-96` is much *better* in quality (84 vs 42) and much *worse* in savings (0.8% vs 7.5%) |
| 3 | `MINRES` beats `LAST` at equal depth, by 0–15 tokens | **HIT on direction, MISS on size** — +56 tokens, far outside the band |
| 4 | `L21` is survivable, `L14` is not | **HIT** — 113 vs 54 |
| 5 | `FLOOR-MIN` lands COMPARABLE and permits `k ≥ 40` | **MISS** — permits `k = 17.3` and reads 38/160 |

The **registered alternative** — "the floor is cheap enough that with an aggressive floor the budget
opens and the FFN question becomes the only question" — **does not fire.** The floor is not cheap
enough, and when it is made cheap enough it stops being a model.

Two of five predictions missed and one missed its magnitude by 4×. The two misses are both
informative and both were only findable because the numbers were committed in advance: I expected
`k/v` to behave like `q/o` (they are opposite on both axes) and I expected the composed floor to
land COMPARABLE (it is the worst arm in the probe bar none).

---

## §6 — what E27 does **not** claim

- **No timing was taken.** Every tok/s in §4 is E25's measured charged-throughput line applied to a
  weight count, carrying E25's own ±5% on any absolute figure. Ratios between rows do not carry it.
- **`MINRES` is a heuristic, not a theory.** It was defined before the run and it worked, but E27
  ran exactly one competing rule (`LAST`). It is not established that `MINRES` is the *best* rule,
  only that it is decisively better than the default one, and that the choice of layers is a
  first-order lever where I had assumed it was a second-order one.
- **The residual profile is the donor's, on one calibration slice.** Whether a 10 B model's profile
  has the same shape — a first-block spike, a middle trough, a rising tail — is untested here.
- **Nothing was retrained.** Every arm is a surgery on frozen weights. A healing pass (the T4 branch,
  `T4_HEALING_PROPOSAL.md`) could move any of these numbers and is the reason `L21-MINRES` is worth
  carrying forward rather than filing.
- **No carve and no router appear in this probe at all.** E27's floors compose with E24's carve
  arithmetically in §4, but that composition has not been *measured* end to end.

---

## §7 — what is owed

1. **`L21-MINRES` deserves a healing run.** It is the best quality-per-weight point the donor branch
   has produced without an oracle, and it is the natural H-arm candidate after H0: 75% of layers,
   `113/160`, no router to fit and nothing to calibrate at inference.
2. **Bisect the depth cliff.** `L21` survives and `L14` does not; `L17`/`L18` is one cheap run and
   would say whether the cliff is at a layer count or at a residual threshold.
3. **A second depth rule.** `MINRES` beat `LAST` by 56 tokens on the first try, which is a strong
   hint that the rule space has not been explored. An obvious next rule: drop by residual but
   forbid dropping contiguous runs.
4. **E21 §8 is now answered negatively** (the rank fraction does not transfer) and can be closed.
5. **E22 §8 remains open**: `k/v` in the factored arms is now measured here (`84/160` at `r/D = 1/16`),
   but the head at 545 M on a 7 B is still untouched.

---

## §8 — the position this leaves the programme in

E24 said the FFN is not the lever and pointed at the floor. E27 attacked the floor with both levers
it has and the floor did not yield: **the goal's shape does not reach 50 tok/s by any combination of
rank and depth applied to a trained donor, because the configurations cheap enough to fit the budget
are not models.**

The honest summary of the donor branch as of E27 is that it has produced one genuinely good result —
**you can drop a quarter of a trained transformer's layers for almost nothing if you drop the right
ones** — inside a branch whose central question is closing negatively. What remains open in the
branch is H0 (no router, no carve by construction, and therefore untouched by E23/E24/E27) and the
healing question generally.

If H0 also fails to move the number, the conclusion the programme should draw is not "cut harder"
but "the 10 B at 50 tok/s is not reachable by surgery on a dense trained donor, and the remaining
path is architectural" — which is what `SCALEUP_ARCHITECTURE.md` already describes and what Phase 64
is already building towards.
