# E27 — the floor, not the FFN

> **SUPERSEDED NUMBER, 2026-09-12, from E28 / E30 / E31.** Every budget figure in this
> document is derived from `THROUGHPUT_G = 49.9 G active weights/s`, which gave **`<= 0.998 G`
> active weights per token for 50 tok/s at T10**. That constant has moved three times since, and
> **the budget now depends on HOW the arm reads its weights**:
>
> | | budget for 50 tok/s at T10 | applies to |
> |---|---|---|
> | as written here (E10/E25) | `0.998 G` charged | — superseded |
> | E28 `CONTAINER-COSTS` | `1.2328 G` charged | E13's layout at the goal's shape, `61.64 G-w/s` |
> | **E30 `AT-THE-WALL`** | **`1.452 G` moved** | the PHYSICAL ceiling: this box's entire `36.30 GB/s` with a perfect kernel. **Nothing can exceed it.** |
> | **E31 `GATHER-COSTS`** | **`1.371 G` in `>= 32 KB` blocks, `0.840 G` at row granularity** | **only arms that ACTIVATE A SUBSET** — carve, MoE, structured sparsity |
>
> **A dense or rank arm streams its weights contiguously and pays no gather penalty**: it is
> priced against E30's line, not E31's. **A carved or routed arm is priced against E31's**, and
> which E31 row applies depends on the exporter's group size.
>
> **For this document specifically**: its arms are rank and depth arms, which stream — so the
> applicable ceiling is E30's `1.452 G`, a `1.45x` widening of the `0.998` used throughout. **The
> qualitative conclusions are unaffected**: every configuration this document rejects is rejected
> by a factor of 5–8, and `1.45x` does not close that.


**Pre-registration. Nothing in §§3–8 has been measured.**

---

## 0. Why this exists, in one paragraph

E24 closed with a sentence that is an instruction: *"The lever that opens the goal's shape is the
one that shrinks the non-FFN floor — rank on `q/o`, and then the head — not more FFN depth."*
E19 had already said FFN-only carving cannot reach the target; E24 put a slope on it (**1.8
teacher-forced tokens per point of activation, and the damage accelerates as the carve deepens**)
and E26's arithmetic put a wall next to it (**with `q/o` dense at `T10` the `k → 0` asymptote is
22.7 tok/s — 50 is impossible with the FFN deleted outright**). So the FFN axis is measured and
the answer is no. **E27 measures the other one.** The floor has exactly two components that any
lever can touch: **how wide a layer is** (rank on `q/o`, on `k/v`) and **how many layers there
are** (depth). Width has been measured at one fraction only. **Depth has never been attacked in
this programme at all**, and it is the goal's own word — *struttura interna*.

## 1. The arithmetic, fixed before the run

At `T10` — the goal's "es 10B" dimensions, `D = 4096`, `QD = 4096`, `KD = 1024`, `F = 14336`,
`L = 48`, `V = 32768` — on E25's **measured** charged throughput of `49.9 G` active weights/s
(band 1.54% over four arms), 50 tok/s means **≤ 0.9980 G active weights per token**. A rank-`r`
factorisation of an `[out, in]` matrix is charged at `r·(out + in)`, so rank saves on `k/v` iff
`r < 819` and on the head iff `r < 3641`.

**`k` permitted of 256, by floor configuration** (router charged at `D·E·L` throughout):

| `r` on `q/o` | `r` on `k/v` | `L` | floor | **`k` of 256** | activation |
|---|---|---|---|---|---|
| dense | dense | 48 | 2.1978 G | **impossible** | — |
| 512 | dense | 48 | 0.9899 G | 0.2 | 0.10% |
| 512 | 256 | 48 | 0.7130 G | 8.6 | 3.37% |
| 256 | 256 | 48 | 0.5117 G | 14.7 | 5.75% |
| 192 | 128 | 48 | 0.3985 G | 18.2 | 7.09% |
| 512 | 256 | 36 | 0.5683 G | 17.3 | 6.78% |
| 512 | 256 | **24** | 0.4236 G | **34.8** | 13.59% |
| 256 | 256 | **24** | 0.3230 G | **40.9** | 15.97% |
| 512 | 256 | **16** | 0.3272 G | **60.9** | 23.80% |
| 192 | 128 | **16** | 0.2223 G | **70.5** | 27.52% |

**Read the last column against what E24 actually measured.** The shallowest FFN depth ever
measured on quality in this programme is `k = 133` of 256 (51.9%), and it read 102/160. **No
width lever at `L = 48` gets the permitted `k` above 18.** Only depth moves it into a range the
carve curve could plausibly survive — and **depth is also the lever that stops the artifact being
a 10 B** (§7).

## 2. What is held fixed

Qwen2.5-1.5B rev `8faed761…`, `attn_implementation="eager"`, fp32, `--threads 6`. The frozen eval
slice 24×512 (ids sha `a1a48dc9…`, 51,870 scored bytes), the five frozen E6 prompts × 32 = 160
positions, the frozen calibration slice (32×512, seed 42424) for every `H = XᵀX`, activation-
weighted rank via `e21_rank.lowrank(W, H, r, weighted=True)` — **imported, not reimplemented**.
Bands inherited unchanged: free-running floor 12, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`; teacher-forced
`> 119` CHEAPER, `107–119` COMPARABLE, `< 107` WORSE. **No carve and no router anywhere in E27** —
this experiment is about the floor, and mixing the FFN axis back in would make every cell
unreadable.

At `D = 1536` the fractions are `r/D = 1/3 → 512`, `1/8 → 192`, `1/16 → 96`. On `k/v`
(`KD = 256`) rank saves iff `r < 219`, so `96` is the fraction-matched cut and `192` would be
nearly free of saving — that is why the `k/v` arm is at 96.

## 3. Arms

| tag | what changes | role |
|---|---|---|
| `base` | — | must read 160/160 |
| `L28-PASSTHROUGH` | the depth machinery, **zero layers dropped** | **planted control on new code**: must be bit-identical to `base` |
| `QO-512` | `q/o`, `r/D = 1/3` | **planted positive**: must reproduce E21's `QO-ACT-512` — `0.8202837636996289`, free 26, tf 144 |
| `QO-192` | `q/o`, `r/D = 1/8` | the fraction `T10-R512` uses. Its quality has never been measured |
| `QO-96` | `q/o`, `r/D = 1/16` | the fraction `T10-R256` uses |
| `KV-96` | `k/v`, `r/D = 1/16` | the floor organ this programme has **never** cut — fp32 and untouched in E21/E22/E23/E24 |
| `QO192+KV96` | both | the composed **width** floor |
| `L24-LAST` | drop the last 4 decoder layers (14%) | depth, positional rule |
| `L21-LAST` | drop the last 7 (25%) | depth, positional rule |
| `L21-MINRES` | drop the 7 layers with the smallest mean relative residual | depth, measured rule |
| `L14-MINRES` | drop 14 (50%) | the depth `T10`'s table needs |
| `FLOOR-MIN` | `QO-192` + `KV-96` + 21 layers | the composed floor |

**`MINRES` is defined here, before it is run**: on the calibration slice, for each decoder layer,
accumulate `mean( ‖block_out − block_in‖ / ‖block_in‖ )` over all positions in one forward pass;
drop the `n` layers with the smallest value. **`FLOOR-MIN`'s depth rule is a registered decision
procedure, not a post-hoc choice**: it uses `MINRES` if `L21-MINRES` reads more than 3
teacher-forced tokens above `L21-LAST`, and `LAST` otherwise.

**Layer dropping reindexes `self_attn.layer_idx`** on the surviving layers, because the KV cache
addresses layers by that field and a gap in it is a silent wrong answer, not an error. `G-F1` is
what proves the reindexing is right.

## 4. Gates

- **`G-F0`** — `base` 160/160 on both metrics and BPB reproduces `0.7675949641196624` to `< 1e-9`.
  Otherwise **VOID**.
- **`G-F1`** — **`L28-PASSTHROUGH` is bit-identical to `base`**: BPB absolute difference `0.0`,
  free 160, tf 160. The depth machinery is new code and **an instrument must fire on a
  known-positive before its nulls count**. Otherwise every depth cell is **VOID**.
- **`G-F2`** — `QO-512` reproduces E21's `QO-ACT-512` to `< 1e-9` (`0.8202837636996289`, free 26,
  tf 144). Otherwise **VOID**: `lowrank` or the `H` capture has drifted and no width cell is
  readable.
- **`G-F3`** — every arm's active weights/token at the donor **and** its `T10` transposition are
  computed by one function from the shapes, printed per arm, and the `T10` row of `QO-512 +
  KV-256 + L48` must reproduce §1's `0.7130 G` exactly. The runner and this brief cannot disagree
  about what an arm costs.
- **`G-F4`** — the verdict, §5.

## 5. Bands — fixed here, before the run

The verdict is about the **floor**, and it is scored on two things at once: the arm must still
rank at the donor, and its `T10` transposition must leave the FFN room the carve could survive.
The threshold `k = 40` of 256 (15.6%) is registered as the bar because it is roughly a third of
the way from zero to `k = 133`, the shallowest depth this programme has ever measured on quality.

- **`FLOOR-OPENS`** — some configuration reads tf **≥ 107** at the donor **and** its `T10`
  transposition permits **`k ≥ 133`**, the shallowest depth with a measured quality number. Then
  the goal is reachable by composing two already-measured results and E28 is the composition.
- **`FLOOR-HELPS`** — some configuration reads tf **≥ 107** at the donor **and** permits
  **`k ≥ 40`**. The goal is not arithmetically closed; what remains is one measurable question —
  *does the FFN survive 16–25% activation with a real router* — and E28 is that measurement at
  exactly that `k`.
- **`FLOOR-IS-NOT-ENOUGH`** — the best configuration that stays COMPARABLE at the donor still
  permits **`k < 40`**. Then, with every lever this programme has measured, **the donor-adaptation
  route to 50 tok/s at a 10 B is closed**, and the honest next move is to say so in those words
  and put the weight on E18's other conclusion — train inside the format at the target shape —
  rather than keep cutting a donor.

## 6. Predictions

1. **`G-F0`, `G-F1`, `G-F2` fire.** `L28-PASSTHROUGH` is bit-identical and `QO-512` reproduces
   E21 exactly.
2. **The rank fraction does NOT transfer: `QO-192` reads `70–90` teacher-forced.** E21 measured
   `r/D = 1/3` at 144 and `r/D = 1/6` at 93; `1/8` is a smaller fraction than `1/6`, so it must
   read below 93, and the interval between 93 and E21's `QO-SVD-256` catastrophe is where I
   expect it. **If this holds, `T10-R512`'s `+14.12%` is a speed number with no quality behind
   it**, which is the disclosure E25 §5 already carried and this would confirm.
3. **`QO-96` reads `< 50`.** `r/D = 1/16` is half the fraction that already cost 51 tokens.
4. **`KV-96` reads `≥ 130`.** `k/v` are 22.0 M of the donor's 1543.6 M active weights, the
   smallest organs in the model, and every previous experiment left them untouched *because* that
   was the favourable choice. If a 44%-cheaper `k/v` costs almost nothing, the favourable choice
   was also nearly free — and that is worth knowing before the next composition charges it.
5. **Depth is the strongest floor lever and it breaks late**: `L24-LAST` reads **≥ 120**,
   `L21-LAST` reads **60–110**, `L14-MINRES` reads **< 30**, and **`MINRES` beats `LAST` at 21
   layers by 0–15 tokens**.

**THE VERDICT I EXPECT — `FLOOR-IS-NOT-ENOUGH`.** §1's table says no width configuration at
`L = 48` permits `k` above 18, and reaching `k ≥ 40` needs `L ≤ 24`, where prediction 5 puts the
donor below COMPARABLE. **I am registering that I expect this experiment to close the donor route
rather than open it**, and registering it *because* that is the outcome I would be tempted to
soften.

**THE REGISTERED ALTERNATIVE.** *If `FLOOR-HELPS` fires — a configuration COMPARABLE at the donor
whose `T10` transposition permits `k ≥ 40` — then the goal is not closed, the carve curve has one
more question to answer and a defined `k` at which to answer it, and E28 is that measurement. In
particular, if `L21-MINRES` or `L24-LAST` holds COMPARABLE, the depth axis is worth a full sweep
of its own and `MINRES` is worth comparing against a trained criterion. Written before the data
exists.*

## 7. What a depth-pruned 10 B is, and is not

**Registered now, so it cannot be argued after the numbers arrive.** A `T10` shape at `L = 16` is
one third of the layers and roughly one third of the parameters: **it is a ~3.5 B, not a 10 B**,
and reporting it as "a 10 B at 50 tok/s" would be dishonest. The goal says *far girare un modello
grande (es 10B)*. So depth results will be reported in two columns — **the `L` that reaches the
budget** and **what fraction of the donor's parameters survive** — and any headline number will
name both. Depth pruning is legitimately one of the goal's *punti attaccabili*; passing off the
result of it as the original model is not.

## 8. What E27 will not be able to claim

- **No speed claim, no timing.** Nothing is exported. `6.79 tok/s` (the real 7.072 B) and `T10`'s
  `4.70 tok/s` stay exact. Every `k` in §1 is arithmetic over E25's measured line, carries ±5%,
  and is **conditional on E26 part B**, which still has no valid record — if a gathered weight
  costs more than a streamed one, §1 is optimistic.
- **One donor, one depth, one vocabulary.** `L = 28` at `D = 1536`. **E16 applies with force
  here**: how many layers a 1.5 B can lose is not how many a 10 B can lose, and this is the axis
  where that objection is strongest. E27 measures the *lever*, not its value at the goal.
- **`MINRES` is a rule I chose.** It is data-driven and cheap and registered in advance, and it
  is still one rule out of many; a null for `MINRES` is not a null for measured layer selection.
- **Nothing about the FFN.** No carve, no router. The `k` column is what the budget *permits*,
  never what the model *survives*.
- **Nothing about H0 or the T4.** H0 is rank-free and depth-free by construction.
