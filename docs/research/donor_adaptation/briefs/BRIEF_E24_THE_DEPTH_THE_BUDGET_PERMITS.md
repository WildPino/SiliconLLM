# E24 — the depth the budget permits, with a router that exists

**Pre-registration. Nothing in §§3–8 has been measured.**

---

## 0. Why this exists, in one paragraph

E23 replaced the oracle router with a closed-form ridge and split the verdict: the carve alone
holds (`V52-LINEAR` tf `110/160`, retention `0.9067`, `ROUTER-HOLDS`) and the composed
configuration does not (`QO512+V52-LINEAR` tf `102/160`, retention `0.7143`, `ROUTER-COSTS`,
band WORSE). `ROUTER-COSTS` obliges, in the brief's own words, that *"the healing target must be
re-derived at a shallower depth."* **E24 is that re-derivation.** It is not a search for a
depth that works — it is a measurement at the three depths the 50 tok/s budget actually permits
once the router is charged, which E23 §7 fixed by arithmetic before this brief existed.

## 1. The depths, fixed by arithmetic and not by choice

The router is `1536 × 256 × 28 = 11,010,048` = `11.0 M`, active every token.

| component | active weights/token |
|---|---|
| `q/o` at rank 512 (`2·D·r` per projection) | 88.1 M |
| `k/v`, untouched | 22.0 M |
| head, untouched | 233.4 M |
| router | 11.0 M |
| **fixed subtotal** | **354.5 M** |
| FFN, full | 1156.1 M |

Against E18 §31's `0.982–1.060 G`, the FFN allowance is `627.5–705.5 M` = **54.28%–61.03%**
activation = **`k = 139 … 156` of 256**. E24 measures the bottom, middle and top of that
interval. **`k = 133` — E19's depth, carried unexamined by E22 and E23 — gives `0.9551 G`, which
is 2.8% BELOW the budget floor**, so it was never the right depth and its shortfall is not
evidence about depths that are.

| `k` | activation | total | role |
|---|---|---|---|
| 133 | 0.5195 | 0.9551 G | **under budget** — E23's arm, carried as the anchor |
| **139** | 0.5430 | 0.9822 G | cheapest in-budget depth |
| **148** | 0.5781 | 1.0228 G | mid |
| **156** | 0.6094 | 1.0590 G | most FFN the budget allows |

## 2. What is held fixed

Qwen2.5-1.5B, the frozen eval slice 24×512 (ids sha `a1a48dc9…`, 51,870 scored bytes), the five
frozen E6 prompts × 32 = 160 positions, D0c's partition `E = 256` read from its own cache, the
frozen calibration slice (32×512, seed 42424) for every fit, ridge `λ = 0.01·mean(diag(XᵀX))`.
**The runner imports `e23_router`'s `fit_routers`/`install` and `e21_rank.lowrank` rather than
reimplementing them**, which is what made E22's and E23's replication gates possible.

## 3. Arms

All composed (`QO512 + carve`), because the composed configuration is the one that failed.

| tag | `k` | router | role |
|---|---|---|---|
| `base` | — | — | must read 160/160 |
| `K133-LINEAR` | 133 | ridge | **planted positive**: must reproduce E23's `1.2014771810176486`, free 9, tf 102 |
| `K133-ORACLE` | 133 | true mass | must reproduce E22/E23's `1.0050386831300866`, free 15, tf 126 |
| **`K139-LINEAR`** | 139 | ridge | verdict |
| **`K148-LINEAR`** | 148 | ridge | verdict |
| **`K156-LINEAR`** | 156 | ridge | verdict |
| `K156-ORACLE` | 156 | true mass | the ceiling at the deepest in-budget depth |
| `K148-STATIC` | 148 | mean calib mass, token-independent | E23 §5's surprise, priced |
| `K148-LOG1P` | 148 | ridge to `log1p(mass)` | the variant I declined in E23 and disclosed |

`K148-STATIC` is charged at **0 M**, not 11.0 M — it needs no per-token score — so if it ties
the ridge it is strictly cheaper and the budget table above shifts in its favour. That is
registered here rather than discovered afterwards.

## 4. Gates

- **`G-U0`** — `base` 160/160 on both metrics. Otherwise **VOID**.
- **`G-U1`** — `K133-LINEAR` reproduces E23 to `< 1e-9` (`1.2014771810176486`, free 9, tf 102)
  **and** `K133-ORACLE` reproduces E22 to `< 1e-9`. Otherwise **VOID**: the router fit or the
  hook has drifted and no depth comparison is readable.
- **`G-U2`** — achieved activation equals `k/256` to `ACT_TOL = 0.002` on every routed arm, so
  depths are compared at exactly their charged cost.
- **`G-U3`** — monotonicity is **not** assumed. If tf is non-monotonic in `k` across the three
  verdict arms by more than 3 tokens, that is reported as the finding and the "deeper is better"
  framing is withdrawn. E16's non-monotonicity in scale is the precedent.
- **`G-U4`** — the verdict, §5.

## 5. Bands — fixed here, before the run

Unchanged and inherited: free-running floor 12, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`. Teacher-forced
(E20 part B): `> 119` CHEAPER, `107–119` COMPARABLE, `< 107` WORSE.

- **`DEPTH-RECOVERS`** — some in-budget depth reads tf **≥ 107** with the ridge router. The
  composed configuration is buildable, `ROUTER-COSTS` was a depth artefact, and the H1/H2 target
  is that depth.
- **`DEPTH-HELPS-NOT-ENOUGH`** — the best in-budget depth reads **103–106**. The direction is
  right and the gap is real; the target must come from somewhere other than depth.
- **`DEPTH-IS-NOT-THE-LEVER`** — the best in-budget depth reads **≤ 102**, i.e. no better than
  `k = 133`. Then the 24-token oracle/ridge gap is not about how much FFN survives, the carve
  axis is exhausted for the composed configuration, and **the T4 request for H1/H2 is withdrawn
  until a different target exists** — H0 is unaffected either way.

## 6. Predictions

1. `G-U0`, `G-U1`, `G-U2` fire; both anchors reproduce exactly.
2. **`K156-LINEAR` reads `103–106` → `DEPTH-HELPS-NOT-ENOUGH`.** Going from 51.95% to 60.94% of
   the FFN is 9 points of activation; E19's own depth curve moved BPB by roughly `0.14` over
   48 points of carve, so 9 points should be worth single-digit teacher-forced tokens, and 102
   needs 5. **This lands right at the boundary, which is why the band has a middle.**
3. **`K156-ORACLE` clears 126.** The oracle at `k = 133` read 126; more FFN with a perfect router
   should not be worse.
4. **`K148-STATIC` lands within 8 tokens of `K148-LINEAR`.** E23 measured 99 vs 110 at `k = 133`;
   I expect the gap to narrow with depth, because more groups kept means fewer decisions matter.
5. **`K148-LOG1P` beats `K148-LINEAR` by 0–4 tokens.** Its pre-run overlap advantage on L27 was
   real (`0.8197` vs `0.7404`) but measured on three layers, and E23 showed overlap and retained
   mass are only loosely coupled.

**THE REGISTERED ALTERNATIVE.** *If `DEPTH-RECOVERS` fires — an in-budget depth reaching tf ≥ 107
with a real, charged router — then `QO512 + carve` is simultaneously inside the 50 tok/s budget,
constructible without an oracle, and COMPARABLE on ranking; E23's `ROUTER-COSTS` was an artefact
of a depth inherited from before routers cost anything; and the H1/H2 target is restored at that
depth with no new mechanism invented. Written before the data exists.*

**AND THE THIRD OUTCOME.** *If `K148-STATIC` ties or beats `K148-LINEAR`, the `11.0 M` router is
not worth its budget and the whole §1 table is recomputed without it — which ADDS `11.0 M` back
to the FFN allowance and moves the permitted depths up again. That would make E24's own depth
grid obsolete on its own result, and the honest response is to say so and re-run, not to keep
the grid. Registered because it is a real possibility, not a hedge.*

## 7. What E24 will not be able to claim

- **No speed claim, no timing. `6.79 tok/s` stays exact.** Nothing is exported; every weight
  count in §1 is arithmetic; `engine.c` implements neither the router nor a factored matvec.
- **No statement about H0 or about healing.** E24 fits regressions and measures depths.
- **One donor, one partition.** `E = 256` from D0c on the 1.5 B. E16 applies.
- **Ridge is still a floor on routers.** A depth null here is a null about *this* router at
  *these* depths.
- **`k/v` stay fp32** in the composed arms, as in E22 and E23 — the comparison remains favourable
  to the configuration, and a like-for-like repeat would widen the gap.

## 8. Cost

One calibration pass (router fit + static accumulator folded into it, as in E23), then nine arms
× (BPB on the frozen slice + 160 free-running + 160 teacher-forced). **Estimated 60–95 minutes,
CPU only, one job.** Smoke first.
