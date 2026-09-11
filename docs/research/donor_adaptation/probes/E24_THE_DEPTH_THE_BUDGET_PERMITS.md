# E24 — the depth the budget permits, with a router that exists

**Brief**: `briefs/BRIEF_E24_THE_DEPTH_THE_BUDGET_PERMITS.md`, pushed at `b78ce4d` **before this
runner existed**.
**Code**: `ternary/e24_depth.py`; `ternary/e23_router.py` (`fit_routers_multi`, the one change,
committed at `1dce754` before the run).
**Result**: `engine/results/e24_depth.json` (`engine/results/e24_depth_smoke.json` for the smoke).
**Cost**: 3,842 s, CPU only, one job, nine arms. No timing taken, nothing exported.

## VERDICT — `DEPTH-RECOVERS`

**The registered alternative fired.** E23's `ROUTER-COSTS` was an artefact of a depth — `k = 133`
— inherited from before routers cost anything, and 2.8% below the budget floor. At the depths the
50 tok/s budget actually permits, the composed configuration recovers with a **real, charged,
oracle-free** router:

| arm | `k` | activation | active/token | BPB | free | **teacher-forced** | band |
|---|---|---|---|---|---|---|---|
| `base` | — | 1.0000 | 1.5436 G | 0.767595 | 160/160 | **160/160** | CHEAPER |
| `K133-LINEAR` | 133 | 0.5195 | 0.9551 G | 1.201477 | 9/160 | **102/160** | WORSE |
| `K139-LINEAR` | 139 | 0.5430 | 0.9822 G | 1.144179 | 13/160 | **105/160** | WORSE |
| `K148-LINEAR` | 148 | 0.5781 | 1.0228 G | 1.104852 | 13/160 | **110/160** | **COMPARABLE** |
| `K156-LINEAR` | 156 | 0.6094 | 1.0590 G | 1.068600 | 15/160 | **118/160** | **COMPARABLE** |
| `K148-LOG1P` | 148 | 0.5781 | 1.0228 G | 1.080834 | 6/160 | **111/160** | **COMPARABLE** |
| `K148-STATIC` | 148 | 0.5781 | 1.0118 G | 1.331382 | 8/160 | **96/160** | WORSE |
| `K133-ORACLE` | 133 | 0.5195 | 0.9551 G | 1.005039 | 15/160 | 126/160 | CHEAPER |
| `K156-ORACLE` | 156 | 0.6094 | 1.0590 G | 0.924102 | 23/160 | 130/160 | CHEAPER |

**`K156-LINEAR` — `1.0590 G`, inside E18 §31's `0.982–1.060 G`, with the 11.0 M router charged and
executed — reads `118/160` teacher-forced, one token below CHEAPER.** This is the first
configuration in this programme that is simultaneously *in budget*, *free of any oracle*, and
*COMPARABLE on ranking*. `k = 133` read 102; the three in-budget depths read 105, 110, 118.

**And the gap the router costs HALVES with depth.** Oracle minus ridge is **24 tokens at `k = 133`
(126 − 102)** and **12 tokens at `k = 156` (130 − 118)**. The shallower the carve, the less the
router's imperfection matters — which is the opposite of the direction the 10 B goal needs to go,
and §6 is about that.

**The counterweight, stated here and not buried.** This is the 1.5 B donor's own *absolute* active
count against an absolute budget. **The recipe does not transfer to the goal's shape as a
fraction.** Transposed to `T10` — the goal's "es 10B" dimensions — `QO512 + K156` is `6.14 G`
active, which on E25's measured line is **≈ 8.1 tok/s, not 50** (§6). E24 moved the quality
question forward; it did not move the goal.

---

## 0. What was owed

E23 split its verdict. The carve alone held with a real router (`V52-LINEAR` tf 110/160, retention
`0.9067`, `ROUTER-HOLDS`); the composed configuration did not (`QO512+V52-LINEAR` tf 102/160,
retention `0.7143`, `ROUTER-COSTS`, band WORSE). `ROUTER-COSTS` obliges, in E23's own words, that
*"the healing target must be re-derived at a shallower depth."*

**E24 is that re-derivation, and it is not a search.** The depths were fixed by arithmetic in
E23 §7, before this brief existed, by charging the router:

| component | active weights/token (Qwen2.5-1.5B) |
|---|---|
| `q/o` at rank 512 (`2·D·r` per projection) | 88.1 M |
| `k/v`, untouched fp32 | 22.0 M |
| head, untouched | 233.4 M |
| **router**, `1536 × 256 × 28` | **11.0 M** |
| fixed subtotal | **354.5 M** |
| FFN, full | 1156.1 M |

Against `0.982–1.060 G` that leaves `627.5–705.5 M` for the FFN = **54.28%–61.03%** activation =
**`k = 139 … 156` of 256**. The runner recomputes this table from the model's own shape rather
than copying it as prose, and reproduces it exactly: `0.9551 / 0.9822 / 1.0228 / 1.0590 G`, with
`k = 133` stamped `in-budget False`.

## 1. What was built, and the one change to shared code

`e23_router.fit_routers` was refactored into `fit_routers_multi(model, ids_cal, layer_ids,
lab_map, targets)`. The accumulation (`XᵀX [D,D]`, `XᵀY [D,E]` in normal-equation form), the
damping (`λ = 0.01·mean(diag(XᵀX))`) and the solve are untouched; only *which function of the
per-group mass* the ridge regresses onto is now a parameter. `fit_routers` keeps its exact
signature and delegates with `targets = ("sqrt",)`. Both targets share **one** calibration pass —
only `XᵀY` is per-target and the solve reuses the same damped `XᵀX`.

**That refactor is not asserted to be inert; `G-U1` proves it.** See §2.

Three things `e24_depth.py` does deliberately, none of them the obvious way:

- **The rank-512 `q/o` factorisation is computed once and swapped in and out**, not recomputed per
  arm. `E21.lowrank` over 56 organs is the expensive part of an arm and eight of nine arms want
  the identical result. `torch.linalg.eigh` consumes no RNG, so this is bit-identical to E23's
  per-arm path — and `G-U1` is what says so.
- **The two anchors are hardcoded AND checked against `e23_router.json` on disk** before the model
  loads; a mismatch stops the run. This caught a real discrepancy: `e23_router.py` carries
  `E22_QO_V52["bpb"] = 1.0050393404796996`, which is **E22's published** number, while E23
  **measured** `1.0050386831300866` for the same arm. They differ in the 7th decimal. Reproduction
  is of the run whose machinery is being reused, so E24 anchors on the measured one.
- **`K148-STATIC` is charged at `1.0118 G`, without the router**, exactly as brief §3 registered —
  it needs no per-token score.

## 2. Gates — all fire

| gate | requirement | result |
|---|---|---|
| **`G-U0`** | `base` 160/160 on both metrics, BPB reproduces `0.7675949641196624` to `< 1e-9` | **FIRES** — 160/160, 160/160, `0.7675949641196625` |
| **`G-U1`** | `K133-LINEAR` reproduces E23's `QO512+V52-LINEAR` to `< 1e-9` | **FIRES** — BPB `1.2014771810176486`, **`abs diff 0.0`**, free 9, tf 102 |
| **`G-U1`** | `K133-ORACLE` reproduces E23's `QO512+V52-ORACLE` to `< 1e-9` | **FIRES** — BPB `1.0050386831300866`, **`abs diff 0.0`**, free 15, tf 126 |
| **`G-U2`** | achieved activation `== k/256` to `0.002` on every routed arm | **FIRES** — exact on all 8 (`0.5195312500`, `0.5429687500`, `0.5781250000`, `0.6093750000`) |
| **`G-U3`** | monotonicity not assumed; a drop `> 3` tokens in `k` is the finding | **MONOTONE** — `(139,105) → (148,110) → (156,118)`, steps `−5, −8`, i.e. every step is a *gain*; worst step `−5` |
| **`G-U4`** | the verdict | **`DEPTH-RECOVERS`**, best `k = 156`, tf 118, `+16` over `k = 133` |

`abs diff 0.0` on both anchors is stronger than the gate asked for: the reproduction is
bit-identical, not merely within `1e-9`. **So `fit_routers_multi` is inert and factorising the
low-rank `q/o` once instead of eight times changes nothing measurable.**

## 3. The depth curve

```
 k      activation   active/token   BPB        teacher-forced   band
 133    0.5195       0.9551 G       1.201477      102/160       WORSE        (out of budget)
 139    0.5430       0.9822 G       1.144179      105/160       WORSE
 148    0.5781       1.0228 G       1.104852      110/160       COMPARABLE
 156    0.6094       1.0590 G       1.068600      118/160       COMPARABLE
```

Monotone in both metrics. Over the 9.0 points of activation from `k = 133` to `k = 156`, BPB falls
`0.132878` and teacher-forced agreement rises **16 tokens — about 1.8 tokens per point of
activation**. E19's own depth curve moved BPB by roughly `0.14` over 48 points of carve; **this
slope is roughly five times steeper, which is why prediction 2 missed** (§5).

Free-running stays AT-FLOOR or barely above (9 → 13 → 13 → 15 against a floor of 12 and a
`RANKS` bar of 80). **E14 §3's rule applies unchanged: the free-running metric has not moved and
nothing here claims it has.** The recovery is entirely in the ranking metric, which is what E20
part B's band was built to read.

## 4. The three side questions the brief registered

**The router earns its 11.0 M.** `K148-STATIC` — the same depth, the same 148 groups fixed per
layer by mean calibration mass, no per-token score at all — reads **96/160** against
`K148-LINEAR`'s 110. A 14-token gap, and `K148-STATIC` is *below* `k = 133`'s 102 despite keeping
15 more groups. **The registered third outcome does NOT fire**: the `11.0 M` is worth its budget
and the §1 depth table stands. (E23 measured 99 vs 110 at `k = 133`; the gap did not narrow with
depth, it widened from 11 to 14.)

**The target axis is worth about one token.** `K148-LOG1P` reads **111** against `K148-LINEAR`'s
110, and BPB `1.080834` against `1.104852` — better on BPB by `0.024`, better on ranking by one
token. Its router is materially different (`‖R‖_F` 19.54 vs 29.26 at layer 0), so this is a real
alternative fit and not a numerical ghost. **One token is inside the noise this programme can
resolve at 160 positions and nothing is promoted on it.** E14 §6 forbids promoting a post-hoc
metric to a gate; `log1p` was registered as prediction 5 in advance, and it landed in its band,
but a one-token win is a null, not a lever.

**The ceiling rises too, and the ridge closes on it.** `K156-ORACLE` reads **130**, clearing
`k = 133`'s 126 as predicted. The oracle/ridge gap is **24 at `k = 133`** and **12 at `k = 156`**.
The router's *relative* damage is halved by going 9 points shallower.

## 5. Predictions — scored, 3 of 5

| # | prediction | outcome |
|---|---|---|
| 1 | `G-U0`, `G-U1`, `G-U2` fire; both anchors reproduce exactly | **HIT** — `abs diff 0.0` on both |
| 2 | `K156-LINEAR` reads `103–106` → `DEPTH-HELPS-NOT-ENOUGH` | **MISS** — it read **118**, twelve tokens above the top of my band, and the verdict is `DEPTH-RECOVERS` |
| 3 | `K156-ORACLE` clears 126 | **HIT** — 130 |
| 4 | `K148-STATIC` within 8 tokens of `K148-LINEAR` | **MISS** — the gap is **14**, and it widened with depth instead of narrowing |
| 5 | `K148-LOG1P` beats `K148-LINEAR` by 0–4 tokens | **HIT** — `+1` |

**Prediction 2 is the instructive miss.** I reasoned from E19's depth curve — `0.14` BPB over 48
points of carve — and concluded 9 points should be worth single digits of teacher-forced
agreement. The actual slope near `k = 148–156` is about five times that. **The carve's damage is
not linear in depth; it accelerates as the carve deepens**, which E19's average over a wide sweep
hid. That also means the same arithmetic, applied *downward* from 51.95% toward the fractions the
goal's shape needs, will *understate* the damage — the error in prediction 2 is in the direction
that makes §6 worse, not better.

## 6. What this does and does not do for the goal

**It does not reach it.** The 1.5 B numbers above are an *absolute* active-weight count compared
with an *absolute* budget (E18 §31: `0.982–1.060 G` for 50 tok/s, which is `9.8–10.6%` **of a
10 B model**). The 1.5 B donor lands in that band at ~58–61% activation only because it is a
1.5 B. Transposing E24's winning recipe to `T10`, the goal's shape, on E25's measured charged
throughput of **49.9 G active weights/s** (band 1.54% over four arms):

| configuration at `T10` | `q/o` | `k/v` | head | router | FFN | total | tok/s |
|---|---|---|---|---|---|---|---|
| dense, with router | 1.6106 | 0.4027 | 0.1342 | 0.0503 | 8.4557 | **10.6535 G** | **4.68** |
| `r = D/3 = 1365`, `k = 156` | 1.0735 | 0.4027 | 0.1342 | 0.0503 | 5.1527 | **6.8134 G** | **7.32** |
| `r = 512`, `k = 156` | 0.4027 | 0.4027 | 0.1342 | 0.0503 | 5.1527 | **6.1426 G** | **8.12** |
| `r = 512`, `k = 148` | 0.4027 | 0.4027 | 0.1342 | 0.0503 | 4.8885 | **5.8783 G** | **8.49** |

**E24's best in-budget recipe, at the goal's shape, is about 8 tok/s.** And the budget runs the
other way: at `T10` the FFN allowance for 50 tok/s is

| `q/o` rank at `T10` | non-FFN floor | FFN allowance at `1.060 G` | permitted `k` of 256 |
|---|---|---|---|
| dense | 2.1978 G | **negative** | impossible |
| `r = D/3 = 1365` | 1.6607 G | **negative** | impossible |
| `r = 512` (`r/D = 1/8`) | 0.9899 G | 0.0701 G | **2.1** |
| `r = 256` (`r/D = 1/16`) | 0.7885 G | 0.2715 G | **8.2** |

**So the goal's shape needs `k ≈ 2 … 8` of 256 — an activation of `0.8%–3.2%` — where E24 measured
quality at `51.9%–60.9%`, and measured it rising with depth at 1.8 tokens per point.** E19 already
concluded that FFN-only carving cannot reach the target; E24 puts a slope on the trade and the
slope points the wrong way. **The lever that opens the goal's shape is the one that shrinks the
non-FFN floor — rank on `q/o`, and then the head — not more FFN depth.** That is E21 §8's open
item (`r/D = 1/3` is validated only at `D = 1536`; `r/D = 1/8` and `1/16` at `D = 4096` are
unvalidated on quality) and E22 §8's (`k/v` still fp32, and the head untouched at 545 M on a 7 B).

**And one number in §6 is conditional.** Every tok/s above assumes E25's charged-throughput
invariant holds for a *carved* arm — that an activated weight costs what a dense one costs.
**That is exactly what E26 part B tests, and E26 part B has not produced a valid record**: its
first attempt ran on a contended box and is VOID (`engine/results/e26_carve_cost_contended.json`).
Until it runs on an idle machine, the `k`-dependent rows above are arithmetic on an untested
assumption, and if a gathered weight costs more than a streamed one they are optimistic.

## 7. What E24 does NOT claim

- **No speed claim, no timing.** Nothing was exported; `6.79 tok/s` (the real 7.07 B) and E25's
  `4.70 tok/s` at `T10` are untouched. Every tok/s in §6 is arithmetic over E25's measured line,
  carries E25's ±5%, and is conditional on E26 part B.
- **Nothing about H0 or about healing.** E24 fits closed-form ridges and measures depths. The T4
  request is unaffected either way; H0 is unaffected.
- **One donor, one partition.** `E = 256` from D0c on Qwen2.5-1.5B, `L = 28`. E16 applies: nothing
  here is a scale claim.
- **Ridge remains a floor on routers.** `DEPTH-RECOVERS` is a statement about *this* router at
  *these* depths. A better router would move `K156-LINEAR` toward the oracle's 130, and the
  measured 12-token gap says there are 12 tokens there to take.
- **`k/v` stay fp32** in every composed arm, as in E22 and E23. The comparison remains favourable
  to the configuration; a like-for-like ternary repeat would widen the gap to `base`.
- **Free-running has not recovered.** 15/160 at the best depth against a floor of 12. Only the
  ranking metric moved, and E14 §3's SCORE/RANK pairing is why both are reported.
- **The `K148-LOG1P` +1 is a null.** Reported because it was pre-registered, not promoted.

## 8. What is owed after this

1. **E26 part B on an idle box** — it prices the assumption every row of §6 rests on.
2. **The rank axis at the goal's width.** §6 says the floor, not the FFN, is what stands between
   this programme and 50 tok/s at `T10`. `r = 512` and `r = 256` at `D = 4096` are `r/D = 1/8` and
   `1/16`; E21 validated `1/3` at `D = 1536`. **Their quality is unmeasured and E24's own miss
   (prediction 2) is evidence that extrapolating a rate across a wide interval fails here.**
3. **The head.** 233.4 M of the 1.5 B's 354.5 M fixed subtotal is the head, and at `T10` it is
   only `0.1342 G` because `T10` carries Mistral's 32,768 vocabulary. A Qwen-vocabulary 10 B
   carries `622 M` and the §6 table would have to be redrawn. E17 is the open question.
4. **A depth grid below `k = 133`**, if §6's `k ≈ 2 … 8` is ever to be answered with data rather
   than an extrapolation the slope says is unsafe.
