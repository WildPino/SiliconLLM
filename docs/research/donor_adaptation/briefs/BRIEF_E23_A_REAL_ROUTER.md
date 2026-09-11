# E23 — a real router, because every carve number here is a ceiling

**Pre-registration. Nothing in §§3–7 has been measured.**

---

## 0. The gap, stated plainly

**Every carve number this programme has ever published uses an ORACLE router.** D0, D0c, E19 and
E22 all install the same hook: compute the true squared activation mass per expert group, keep
the top `k`, zero the rest. D0c says so in its own text — *"the router is an oracle: it reads the
true activation mass, so this report is a floor no real router can reach"* (§132–135).

So these numbers are ceilings:

| | BPB | free | teacher-forced |
|---|---|---|---|
| `V52` (E19, E22) | `0.909441` | 12/160 | `117/160` |
| `QO512+V52` (E22) | `1.005039` | 15/160 | **`126/160`** |

`QO512+V52` is the configuration `decisions/T4_HEALING_PROPOSAL.md` proposes spending GPU weeks
on. **Spending them on a ceiling would be healing something that cannot be built.** E23 replaces
the oracle with a router that only sees what a router can see.

## 1. The prior, and it is not favourable

**F1 already tried to predict this donor's FFN activity and found the sparsity absent.** At a 1%
FFN-output-error budget the donor needs `a = 0.9506` of its neurons; **an oracle reading `|h_i|`
exactly still needs `0.836`.** F1's instrument fired on a known positive (C4, 28/28 layers) — the
construction was sound, the donor was not sparse.

**Two things make E23 a different question rather than a repeat.** First, F1 predicted
**per-neuron** activity; E23 predicts **per-group** mass over D0c's 256 co-activation groups of
35 neurons each, and group mass is a far smoother target than individual magnitudes. Second, F1's
tolerance was 1% of FFN output energy; E19 showed the end-to-end tolerance is much looser — `V52`
drops 48% of the FFN for `+0.141846` BPB. **A router can be bad at F1's question and still good
enough at E19's.**

And one prior points the other way: **Probe-4 falsified the hot-pool** — working sets are
approximately i.i.d. across tokens, so *locality comes only from prediction*. A static choice of
experts should be worthless, which is why §4 carries it as the planted negative.

## 2. The router, fixed here

**Linear, per layer, from the block input, closed-form.** For layer `L`, let `x` be the input
`q/k/v` and the MLP norm see (`D = 1536`), and let `m_e = Σ_{i∈e} h_i²` be the true mass of expert
group `e` (`E = 256`). Fit

```
R_L = argmin_R  || sqrt(m) - X R ||_F^2 + lambda ||R||_F^2
```

on the frozen calibration slice (32×512, seed 42424), ridge `lambda = 0.01 · mean(diag(XᵀX))` —
E20's damping, unchanged. At inference the hook scores `x R_L` and keeps the top `k`. **Closed
form, no gradients, no GPU, no hyperparameter search**; `sqrt(m)` rather than `m` because the
target is then on the same scale as the activations that produce it.

**It is charged to the budget.** `R_L` is `1536 × 256` = `393,216` per layer, `11.0 M` over 28
layers, active on every token. Charged ternary at `0.500000` B/weight that moves the
`QO512+V52` configuration from `0.9441 G` to **`0.9551 G`** — still inside E18's
`0.982–1.060 G`. **The router being affordable is not in question; whether it works is.**

## 3. What is held fixed

Same donor (Qwen2.5-1.5B), same frozen eval slice (24×512, sha `a1a48dc9…`), same five E6
prompts, `N_NEW = 32`, same D0c partition `E = 256` read from the cache D0c's own run wrote,
same `k = 133`, same metrics (BPB, free-running greedy, E20 part B's teacher-forced top-1).
**Nothing is re-clustered and nothing is re-implemented**: the hook is `e19_carve_rank.install`
with its scoring function swapped, and the low-rank arms import `e21_rank.lowrank`.

## 4. Arms

| tag | router | role |
|---|---|---|
| `base` | — | must reproduce `results/e6/ref.json` |
| `V52-ORACLE` | true mass | **planted positive**: must reproduce E22 exactly (`0.909441`, 12, 117) |
| `V52-STATIC` | the `k` groups with the largest mean mass over the calibration slice, same for every token | **planted negative**: Probe-4 says this should be worthless |
| `V52-RANDOM` | a fixed random `k`-of-`E` per layer, seed pinned | second planted negative, and the floor for "how bad is bad" |
| **`V52-LINEAR`** | §2's ridge router | **verdict arm** |
| `QO512+V52-ORACLE` | true mass | E22's composed ceiling, re-read |
| **`QO512+V52-LINEAR`** | §2's ridge router | **the configuration the T4 proposal is about** |

## 5. Gates

- **`G-T0`** — `base` at `160/160` free and `160/160` teacher-forced. Otherwise **VOID**.
- **`G-T1`** — `V52-ORACLE` reproduces E22's `0.909440994415161` to `< 1e-9`, free `12`, tf `117`,
  achieved activation `0.51953125`. Otherwise **VOID**: the harness has drifted and nothing else
  can be read.
- **`G-T2`** — the planted negatives must actually be bad. If `V52-STATIC` or `V52-RANDOM` scores
  within `2` teacher-forced tokens of `V52-LINEAR`, **the instrument has no resolution** and the
  verdict is withheld — a router that cannot beat picking at random has not been shown to route.
- **`G-T3`** — the verdict: `V52-LINEAR` and `QO512+V52-LINEAR` on both metrics, against §6.
- **`G-T4`** — achieved activation must match `0.51953125` to `ACT_TOL = 0.002` for **every**
  routed arm, so all arms are compared at the same cost.

## 6. Bands — fixed here, before the run

Free-running unchanged since E17/E18: floor `12`, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`.
Teacher-forced, E20 part B's measured ternary band as E21 and E22 used it: `> 119` CHEAPER,
`107–119` COMPARABLE, `< 107` WORSE. On top of those, the question E23 exists to answer:

**Router retention**, against its own oracle rather than an absolute:

```
retention = ( tf(LINEAR) - tf(RANDOM) ) / ( tf(ORACLE) - tf(RANDOM) )
```

- **`ROUTER-HOLDS`** — `retention ≥ 0.80` **and** `tf(V52-LINEAR) ≥ 107`. The carve survives
  contact with a real router and the T4 target stands as written.
- **`ROUTER-COSTS`** — `0.40 ≤ retention < 0.80`, or `tf` in `80–106`. The carve survives but the
  healing target must be re-derived at a shallower depth.
- **`ROUTER-BREAKS-IT`** — `retention < 0.40` or `tf < 80`. **Every carve number in this
  programme is then a ceiling with no floor under it**, E19's and E22's verdicts need restating
  as such, and the T4 request is withdrawn until a different target exists.

## 7. Predictions

1. `G-T0` and `G-T1` fire; the oracle reproduces E22 to the token.
2. **`V52-STATIC` and `V52-RANDOM` both land `AT-FLOOR` free-running and below `40`
   teacher-forced.** Probe-4's i.i.d. finding says a token-independent choice carries no
   information; if these two are *not* bad, Probe-4 is wrong about this donor and that is the
   finding instead.
3. **`V52-LINEAR` reads `ROUTER-COSTS`, `retention` in `0.40–0.80`.** A linear map from the block
   input cannot recover group mass that depends on the gate's nonlinearity; F1 says the per-neuron
   version of this fails badly, and grouping should recover some but not most of it.
4. **`QO512+V52-LINEAR` is below `V52-LINEAR` by more than the fp32 pair was** (E22: 126 vs 117,
   a *gain* of 9). Registered because E22's composition surprised me in the favourable direction
   and I am not assuming it does so twice.
5. `G-T4` holds exactly for every arm — the top-`k` is a hard count, so achieved activation is
   `k/E` by construction whatever the scores are.

**THE REGISTERED ALTERNATIVE.** *If `V52-LINEAR` reads `ROUTER-HOLDS` — retention ≥ 0.80 and
teacher-forced ≥ 107 from a ridge regression with no gradients and no tuning — then the carve is
not an oracle artefact, the `0.9551 G` configuration is buildable as specified, and the T4
proposal's §2 precondition is discharged in one CPU afternoon. That would make `QO512+V52` the
first configuration in this programme that is simultaneously inside the 50 tok/s budget, best-in-
class on ranking, and constructible without an oracle. It is written here before the data exists.*

**And the third outcome.** *If `G-T2` fails — the planted negatives score near the verdict arm —
E23 decides nothing, because an instrument that cannot separate a trained router from a random one
cannot certify either. Registered as a real possibility rather than left available afterwards.*

## 8. What E23 will not be able to claim

- **No speed claim, no timing. `6.79 tok/s` stays exact.** Nothing is exported. The router's
  `11.0 M` is priced in §2 by arithmetic, not measured, and `engine.c` implements neither the
  router nor a factored matvec.
- **Linear is a floor on routers, not the best one.** A two-layer router, a router reading the
  previous layer's activity, or one trained jointly during healing could all beat it. A null here
  bounds *this* router; only `ROUTER-HOLDS` would be a positive result about routers in general.
- **One donor, one depth.** `k = 133` on the 1.5 B. E16's non-monotonicity in scale applies.
- **No healing.** E23 fits a regression; it does not train the model.

## 9. Cost

One calibration pass collecting `(x, sqrt(m))` per layer (28 × `1536 × 256` targets over 16,384
tokens), then 28 ridge solves of a `1536 × 1536` system, then seven arms × (BPB on the frozen
24×512 heldout + 160 free-running + 160 teacher-forced). **Estimated 50–80 minutes, CPU only,
one job.** Smoke first.
