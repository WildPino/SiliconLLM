# E38 — is there any selector at all, or is 1.17% simply not enough capacity?

**Read this first: the verdict landed on my registered prediction by 0.0416 BPB.** The band
`SELECTION-IS-DEAD` required the oracle to come within **0.50** of a random selector and it came
within **0.4584**. The band is assigned as registered and I am not dressing that up — **a bar
four hundredths away from the other side is a coin's width**, and §3's *shape* is worth far more
than §2's cell.

**SPEED: none. fp32, not ternary, deliberately** (brief §2). E38's absolute BPB is **not**
comparable to E37's; only the gaps within E38 are.

---

**Verdict: `SELECTION-IS-DEAD`.** Hand the carve a **perfect, per-token, unattainable oracle** at
the activation rate E36's speed requires — 1.17%, `k = 3` of 256 — and it reads **4.131817 BPB**.
Dense fp32 is `0.767595`. Chance is `4.069819`. **The oracle is 0.062 ABOVE the chance line.**

**Cheating does not save 1.17% activation.** No router can be built that changes this, because
the oracle *is* the ceiling of every router that ranks groups by mass.

**But the constructive half is larger than the verdict.** At `k = 64` the oracle beats a random
selector by **2.5546 BPB** and the *fitted* router beats it by **2.3691**. Selection is an
enormous lever at 25% activation and worth nothing at 0.39%. **The value of routing has a peak,
and the goal's rate is past it.**

**Brief**: `briefs/BRIEF_E38_IS_THERE_ANY_SELECTOR_AT_ALL.md`, pre-registered and pushed before
the runner existed. **Runner**: `engine/e38_oracle_ceiling.py`.
**Results**: `engine/results/e38_oracle_ceiling.json`. 4,364 s, 26 arms.

---

## 1. The gates

| gate | what it requires | measured | fires |
|---|---|---|---|
| `G-E38A` | at `k = E` the mask is the identity for **all four** selectors, `< 1e-6` | worst **0.00e+00** | **yes** |
| `G-E38B` | unmasked fp32 == E22's `base` `0.767595 ± 0.001` | **0.767594964**, diff **−3.59e-08** | **yes** |
| `G-E38C` | `oracle ≤ every attainable selector` at every `k`, by construction | **0 violations** | **yes** |

`G-E38A` firing at *exactly* zero matters: it says the labels, the partition, the four selector
paths and the masking hook are **bit-inert** when nothing is dropped, so every number below is
the sparsity and not the harness.

## 2. The sweep — grouping A (the D0c labels E37 used)

| k | activation | oracle | fitted | random | static | **oracle − random** |
|---|---|---|---|---|---|---|
| 256 | 100% | 0.767595 | 0.767595 | 0.767595 | 0.767595 | 0.0000 |
| 64 | 25% | **1.383412** | 1.568866 | 3.938012 | 2.211871 | **2.5546** |
| 16 | 6.25% | 3.449466 | 3.597108 | 4.233244 | 4.338377 | 0.7838 |
| **3** | **1.17%** | **4.131817** | 4.398009 | 4.590205 | 4.574275 | **0.4584** |
| 1 | 0.39% | 4.612284 | 4.654301 | 4.638128 | 4.624240 | 0.0258 |

Dense `0.767595`, chance `4.069819`. **Everything at `k ≤ 3` is over the chance line, the oracle
included.**

## 3. The shape, which is the actual finding

**The value of selection is not monotone in sparsity — it peaks.**

```
oracle - random:   k=256  0.000     (nothing to choose)
                   k=64   2.555     <- the peak
                   k=16   0.784
                   k=3    0.458
                   k=1    0.026     (nothing left to choose between)
```

Both ends are structural. At `k = E` every selector keeps everything, so there is nothing to
choose. At `k = 1` one group of 35 neurons out of 8,960 carries no signal wherever you put it,
so there is nothing worth choosing. **In between there is a regime where choosing well is worth
2.5 bits per byte, and E36's speed requirement sits below it.**

This is the same reading E37 §7 registered from the other side — *a partial FFN is worse than
almost no FFN* — now with the selector axis attached: **at 1.17% the FFN's output is
systematically wrong no matter which neurons produce it.**

## 4. What this says about E37, and it is a correction to an INTERPRETATION, not to a number

E37's addendum B measured `fitted ≈ random` in the ternary engine — 0.0088 BPB apart at `k = 3`,
0.13 at `k = 64` — and I read that as *"a router that finds 71% of the oracle's mass and one that
finds 6% produce the same model."* **E38 says that reading was made in a window too narrow to
support it.**

| | dense | chance | **room** |
|---|---|---|---|
| E37, ternary | 3.475706 | 4.069819 | **0.5941 BPB** |
| E38, fp32 | 0.767595 | 4.069819 | **3.3022 BPB** |

**E37's entire sweep lived inside 0.59 BPB.** The fitted-vs-random gap E38 measures at `k = 64`
is **2.37 BPB — four times E37's whole window.** A difference that large cannot be seen by an
instrument whose dynamic range is a fifth of it, because ternarization had already spent 82% of
the distance to chance before the carve was applied.

**What does NOT change**: E37's verdict cell (`S15-K3 = 4.029398`, `SPARSITY-DEGRADES`) is a
measurement and stands. Addendum B's band was assigned correctly by its own pre-registered rule.
**What changes is what that band licenses**: `ROUTER-IS-NOT-THE-CONSTRAINT` is true *of a
ternarized carved model at 1.17%*, and it is **not** the general statement that routing quality
does not matter. E37's §5 now carries a forward pointer to this section.

**And the honest symmetry**: E38 does not prove ternarization *destroys* routing either. The
floor-compression explanation and the destruction explanation both fit, and separating them
needs a ternary arm with an oracle — which the engine cannot express, for the same reason E38 is
in PyTorch at all.

## 5. Predictions, scored as registered

| # | registered | outcome |
|---|---|---|
| 1 | all three gates fire | **HIT** — and `G-E38A` at exactly 0.00e+00 |
| 2 | **`SELECTION-IS-DEAD`** | **HIT, by 0.0416 BPB.** The oracle beat random by 0.4584 against a 0.50 bar. Registered against my own instinct and it held — but a 4-hundredths margin is not a strong result, and §3 is what should be quoted. |
| 3 | `static` ties `random` at `k ≤ 16` | **HIT below `k = 16`, MARGINAL at it.** `k=3`: 0.016 apart. `k=1`: 0.014. `k=16`: 0.105, and with `static` on the *wrong* side. **Per-token routing buys 0.176 BPB at the goal's rate** (fitted 4.398 vs static 4.574) — not nothing, but both are over chance. |
| 4 | grouping B within 0.10 of A at `k = 3` | **HIT for the oracle** (+0.0310), **MISS for the others** (random −0.2891, static −0.4094). E37 §9 item 2 is answered for the ceiling and not for attainable selectors. |
| 5 | the knee is where E37's was — most loss paid by `k = 64` | **MISS, and informatively.** At `k = 64` the **oracle** has paid `+0.616` of its eventual `+3.364` — **18%**. `random` has paid **83%**. **The knee's position is a property of the SELECTOR, not of the shape**, which no probe in this branch had shown. |

**Prediction 4's miss-half is the interesting one.** At `k = 3` a *random equal partition* is
**better** than the D0c labels for the random and static selectors (−0.289, −0.409). A random
partition spreads high-mass neurons evenly across groups, so an arbitrary pick catches some of
them; the fitted label set concentrates them, which helps a good selector and hurts a bad one.
**`static` at `k = 64` shows the same thing at full strength: 2.2119 on grouping A against
4.5833 on grouping B, a 2.37 BPB swing from the partition alone.**

## 6. What this does NOT say

- **Nothing about the ternary object**, by construction — see §4's closing symmetry.
- **Nothing about a better criterion.** This oracle maximises **group mass**, which is what E23's
  and E37's routers regress onto. If some other criterion picks better groups, E38 cannot see it,
  and prediction 2's near-miss makes that question *more* live rather than less.
- **Nothing about a trained-sparse model.** Dense-trained donor forced sparse, no healing. H0 is
  the other axis and it is the one that moved.
- **Nothing about 10 B.** 1.5 B, one donor, one label family.

## 7. Owed

1. **A criterion sweep.** Mass is one target. Gradient-weighted or output-aligned selection would
   have a different oracle, and §5's 0.0416 margin says the ceiling is close enough to the bar
   that a better criterion could move the verdict.
2. **A ternary oracle arm**, to separate §4's two explanations. Needs an engine that can take a
   selection from outside, which is a small `donor_engine.c` change (`--carve-sel <file>`).
3. **The partition as a design parameter.** §5 found a 2.37 BPB swing from grouping alone at
   `k = 64`. Every carve in this programme has used one label family and nobody has priced the
   partition.
4. **E37 §9 item 1 is now more urgent**, not less: the zero-FFN arm. Both probes point at
   *a partial FFN is worse than almost no FFN*, and neither can test it.
