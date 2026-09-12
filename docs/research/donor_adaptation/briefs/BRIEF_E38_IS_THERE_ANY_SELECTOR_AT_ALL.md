# BRIEF E38 — is there any selector at all, or is 1.17% simply not enough capacity?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. The gap this closes, and it is a gap E37 opened

E37 measured that a really-trained donor at **1.17% FFN activation** reads `4.029398` BPB — 0.040
under chance. Then its own control measured something stranger: a router capturing **71.3%** of
the oracle's group mass and one capturing **6.1%** produce the **same model**.

**That leaves the decisive question unasked.** `fitted ≈ random` tells us the *ridge family* is
not the lever. It does **not** tell us whether **any** selector is. Both routers might be sitting
far below a ceiling that a better one could reach — or the ceiling itself might be at the floor.

**Those two readings send the next GPU-hour to opposite places.** If a perfect selector recovers
the model, the T4 ask is *learn a better router*. If even cheating cannot, the ask is *train
inside the format / change the shape*, and every hour spent on routers is wasted. **This probe is
CPU-only and it decides that.**

## 1. The question

> **Give the carve an ORACLE — per-token, exact, unattainable — and see whether the model comes
> back. Is the damage at 1.17% activation a failure of SELECTION, or of CAPACITY?**

## 2. The object, and the one deliberate difference from E37

Donor **Qwen2.5-1.5B** (`S15`), real weights, **fp32, NOT ternarized**, `q/o` untouched. The
carve is simulated in PyTorch by masking the FFN intermediate `gate·up` to the selected groups
before `down_proj` — which is exactly what `ffn_carved` computes, with the format removed.

**Why fp32 and not the engine.** E37 measured selection and format **together**. Here the
question is selection alone, and the engine cannot express an oracle (its selector is a router
matvec). **The cost is that E38's absolute BPB is not comparable to E37's** — only the gaps
*within* E38 are. That is registered as a limitation in §6, not discovered later.

Grouping **A** = the D0c label partition E37 used (`E = 256`, 35 neurons).
Grouping **B** = a seeded random equal partition of the same `F = 8960`.

## 3. The selectors

| name | what it is | attainable? |
|---|---|---|
| **`oracle`** | top-`k` groups by the true per-token group mass `‖gate·up‖²` — the exact quantity E23's and E37's routers regress onto | **no**, it reads the answer |
| `fitted` | E37's ridge routers (`e37_routers_E256.npz`), `x @ R`, top-`k` | yes |
| `random` | the seed-matched synthetic router (`carve_common.router_weights`, seed 26) | yes |
| **`static`** | the top-`k` groups by mass **averaged over the calibration set** — the same groups for every token, **no routing at all** | yes, **and free** |

`static` is in here because if it ties the others, the router matvec E37 charges `E·D·L` for is
buying nothing, and a quality probe will have produced a **speed** result.

## 4. Arms

Grouping A: `k ∈ {256, 64, 16, 3, 1}` × all four selectors.
Grouping B: `k ∈ {64, 3}` × `{oracle, random, static}` (no refit; the point is the partition, not
the router).

## 5. The gates

**`G-E38A` — the planted control, and it must FIRE before any null here counts.** At `k = 256`
every selector keeps every group, so the mask is the identity. All four must reproduce the
**unmasked** fp32 BPB to **< 1e-6**. If masking is not inert when nothing is dropped, no number
below it means anything.

**`G-E38B` — the anchor.** The unmasked fp32 model must read E22's published `base`
**0.767595 ± 0.001** on the frozen 24×512 heldout slice (`ids_sha a1a48dc9…`). Re-measured today
at `0.7675949641`, so this is a live number and not a quotation.

**`G-E38C` — the oracle is an upper bound.** At every `k`, `oracle ≤ fitted` and
`oracle ≤ random` in BPB, by construction: all three rank the same groups by the same criterion
and the oracle ranks them exactly. **A violation means the oracle is mis-computed**, and the
probe reports nothing.

## 6. The verdict cell and the bands, named before the run

**`oracle` at `k = 3`, grouping A** — 1.17% activation, the rate E36's speed requires.
Dense is `0.767595`; chance is `4.069819`.

| band | name | what it would mean |
|---|---|---|
| within **+0.10** of dense | `SELECTION-IS-THE-LEVER` | a perfect selector nearly recovers the model at 1.17%. E37's gap is a **router** problem, and the next GPU-hour goes to routers. |
| oracle beats `random` by **> 0.50** but not within +0.10 | `SELECTION-PARTIAL` | routing buys something real and does not close it; capacity and selection both bind. |
| oracle within **0.50** of `random` | `SELECTION-IS-DEAD` | **cheating does not help.** The damage is CAPACITY. No router can be built that changes this, E37's "how few, not which" holds at the ceiling, and every hour spent on routing is wasted. |

## 7. Predictions — fixed here, before the run

1. **All three gates fire.**
2. **`SELECTION-IS-DEAD`.** I am registering the prediction my own evidence supports rather than
   the comfortable one: E37's control found that 6.1% and 71.3% mass capture give the same model,
   and it is hard to see why 100% would differ. **If this is right, the router branch of this
   programme is closed by a CPU probe and no GPU time is spent finding that out.**
   **If it MISSES** — if the oracle recovers — then mass is the **wrong criterion**, both routers
   were regressing onto a bad target, and E23/E37's whole router family needs rebuilding against
   whatever the oracle is actually tracking. That is the reading I will take, and I am naming it
   now so it cannot be invented afterwards.
3. **`static` will tie `random` at `k ≤ 16`.** If it does, per-token routing buys nothing at this
   rate and the `E·D·L` the router costs in every budget table since E26 can be **deleted** —
   a speed result out of a quality probe.
4. **Grouping B will land within 0.10 of grouping A at `k = 3`.** This is E37 §9 item 2: if the
   damage is "how few, not which", it must be invariant to the partition. **If B differs, E37's
   central reading is wrong** and the label set was doing more work than its own control implied.
5. **The knee will be in the same place as E37's**, i.e. most of the loss already paid by `k = 64`
   (25% activation), for the oracle too.

## 8. What E38 will NOT be able to claim

- **Nothing about the ternary object.** fp32 only, deliberately. E22 and E32 both measured that
  format damage does not compose additively with structural damage, so E38's gaps may not carry
  to the engine's numbers.
- **Nothing about a trained-sparse model.** Same limit as E37: this is a dense-trained donor
  being forced sparse, with no healing.
- **Nothing about 10 B.** 1.5 B, one donor.
- **Nothing about better criteria.** The oracle here maximises **group mass**. If some other
  criterion (gradient-weighted, output-aligned) would pick better groups, this probe does not
  see it — and prediction 2's miss-branch is exactly the door to that question.
