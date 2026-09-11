# E22 — does cheap compose? The first configuration inside the 50 tok/s budget

**Pre-registration. Nothing in §§4–8 has been measured.** §2 is arithmetic over already-measured
quantities and is published here, before the run, as E19 part A and E21 §2 were.

---

## 0. Why this and why now

E21 produced the first structural cut that is nearly free: `QO-ACT-512`, activation-weighted
rank-512 on all 28 layers' `q_proj`+`o_proj`, **`+0.052689` BPB, 144/160 teacher-forced**. E19
produced the deepest FFN carve this donor's own budget requires: `V52`, **`+0.141846` BPB**, under
an oracle router.

**Separately, each is the cheapest thing on its axis. Together — and this is arithmetic, §2 —
they put the 1.5 B donor at `0.944 G` active weights per token, inside E18's 50 tok/s budget of
`0.982–1.060 G`, for the first time in this programme.**

So the question is no longer "can anything get inside the budget". It is **"does the thing inside
the budget still work"**, and E21 supplied a reason to doubt it: its one composition arm,
`BOTH-ACT-256`, read **48/160** teacher-forced where its halves read 68 and 93 — *worse than
either* — while its BPB came in `0.342498` **better** than the additive prediction. **One datum,
two metrics, opposite directions.** E22 is the second datum, on the pair that matters.

## 1. Inheritance, exactly

| from | what E22 reuses | how |
|---|---|---|
| E21 | the activation-weighted low-rank construction, `H` capture, damping `λ = 0.01·mean(diag H)` | `ternary/e21_rank.py`'s own `lowrank`, imported |
| E19 | the oracle top-`k` carve and D0c's cached partition | `density/e19_carve_rank.py`'s own `install`/`load_labels`, imported |
| E20 part B | teacher-forced top-1 as the standing second metric, and its bands | unchanged |
| E18/E17 | free-running floor `12`, margin `2`, `AT-FLOOR ≤ 14`, `RANKS ≥ 80` | unchanged |
| E6 | the five frozen prompts, `N_NEW = 32`, 160 greedy positions | unchanged |

**Nothing is re-implemented.** Both verdict arms are compositions of code that has already been
run and published, which is what makes §5's replication gates meaningful rather than decorative.

## 2. The budget, arithmetic, published before the run

Every weight charged as **ternary** (`0.500000` B/weight, recomputed from the artefact by E18's
runner). Budget for 50 tok/s: **`0.982–1.060 G`** active weights/token (E18 §31). A `D×D`
projection at rank `r` costs `2·D·r`, i.e. `2r/D` of dense — at `r = 512`, `D = 1536`, **two
thirds**.

| Qwen2.5-1.5B | dense | after `QO512` | after `QO512` + `V52` |
|---|---|---|---|
| FFN | `1156.1 M` | `1156.1 M` | **`600.6 M`** (`51.953125%`, E19's achieved value) |
| `q + o` | `132.1 M` | **`88.1 M`** | `88.1 M` |
| `k + v` | `22.0 M` | `22.0 M` | `22.0 M` |
| `lm_head` | `233.4 M` | `233.4 M` | `233.4 M` |
| **total active** | `1.5436 G` | `1.4995 G` | **`0.9441 G`** |

> **`0.9441 G` is inside the 50 tok/s budget** — 3.9% under its low end, 11% under its high end.
> On E18's own derivation that is **52.0–56.1 tok/s**. This is the first configuration in the
> donor programme that the speed side does not immediately reject.

**And it does not transfer to 7 B, which must be said in the same breath.** On the Coder-7B the
same two cuts give `q+o` `719.3 → 479.5 M` and the FFN `5703.2 → 2963.1 M`, for **`4.090 G` — 4.0×
the budget.** Worse: with `q/o` already cut, `attn + head` alone is `1.127 G`, so **the budget
permits `−107 M` of FFN, i.e. none at all.** The 1.5 B fits because its head is `233 M`; the 7 B
does not because its head is `545 M` against the same fixed budget. **E22 is therefore a test of
whether a budget-feasible configuration works, on the only donor where one exists — not a claim
that the goal's size is reachable.**

## 3. The precision axis, composed for the first time

Every E21 arm is fp32; the engine ships ternary. A factored `q/o` that actually runs is
`W ≈ A·B` with **both factors ternary**. E22 builds it: `A = W H^½ Bᵣ` (`n_out×r`),
`B = Bᵣᵀ H^-½` (`r×n_in`), each ternarized with `t2_rules.r3_actsearch`, the **shipped** rule.

`B` sees `x`, whose per-channel RMS is already captured. **`A` sees `Bx`, and its RMS is available
in closed form from the same `H`** — `rms_A = sqrt(diag(B H Bᵀ)/n)` — so no second calibration
pass is needed and no RMS is guessed. Recorded here because it is the one place E22 computes
something E21 did not.

## 4. Arms

| tag | what | role |
|---|---|---|
| `base` | untouched donor | must reproduce `results/e6/ref.json` |
| `QO512` | E21's `QO-ACT-512`, rerun | **replication anchor**: BPB `0.820284`, free `26`, tf `144` |
| `V52` | E19's carve, `E=256`, `k=133`, oracle | **replication anchor**: BPB `0.909440994415161`, free `12` |
| **`QO512+V52`** | both, composed | **verdict arm A — the §2 configuration** |
| `QO512-T` | the low-rank factors ternarized (§3) | rank × precision, attention only |
| **`QO512-T+V52`** | verdict arm A with ternary factors | **verdict arm B — what would actually ship** |
| `STACK` | `QO512-T` + `V52` + ternary FFN (`R3`) + ternary head (`R3`) | the whole runnable model, end point |

`V52`'s teacher-forced number **has never been measured** — E19 and E18 both owe it. E22 measures
it as a by-product of the anchor arm, discharging that item.

## 5. Gates

- **`G-S0`** — `base` at `160/160` free **and** `160/160` teacher-forced, BPB `0.767595`.
  Otherwise **VOID**.
- **`G-S1`** — `QO512` reproduces E21's BPB to `< 1e-9` and its `26`/`144`. Otherwise **VOID**.
- **`G-S2`** — `V52` reproduces E19's `0.909440994415161` to `< 1e-9` and its `12/160`, and its
  achieved activation matches `0.51953125` to `ACT_TOL = 0.002`. Otherwise **VOID**.
- **`G-S3`** — the verdict: `QO512+V52` and `QO512-T+V52` read on **both** metrics against §6.
- **`G-S4`** — teacher-forced fires on the known-positive before any null is read (E20 part B's
  instrument, planted-control law).

**Two independent replication gates against two different prior runs is the strongest control
this programme has been able to build**, and it is only possible because §1 imports rather than
reimplements.

## 6. Bands — fixed here, before the run

**Free-running** (unchanged since E17/E18): floor `12`, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`.
**Teacher-forced** (E20 part B's measured `107`–`119` for eight ternary heads, as E21 used it):
`> 119` = `CHEAPER`, `107`–`119` = `COMPARABLE`, `< 107` = `WORSE`.

**Additivity — new, and it needs both halves (E14 §3: every SCORE metric needs a RANK partner).**

```
excess = BPB(A+B) - [ BPB(A) + BPB(B) - BPB(base) ]
```

- **score side**: `|excess| ≤ 0.020` (4·σ_seed) = `ADDITIVE`; `< −0.020` = `SUB-ADDITIVE`;
  `> +0.020` = `SUPER-ADDITIVE`.
- **rank side**: `tf(A+B) ≥ min(tf(A), tf(B))` = `RANK-SUB-ADDITIVE`; below it =
  `RANK-SUPER-ADDITIVE`.

For reference, E21's single datum read **`SUB-ADDITIVE` (`−0.342498`) and `RANK-SUPER-ADDITIVE`
(48 against 68 and 93)** — the two halves disagreeing is itself the thing being tested again.

## 7. Predictions

1. `G-S0`, `G-S1`, `G-S2` fire; both anchors exact.
2. **`QO512+V52` is `SUB-ADDITIVE` in BPB** (`excess < −0.020`, additive prediction `0.962130`).
   Mechanism: both constructions preserve logit geometry, and two geometry-preserving
   perturbations overlap rather than stack.
3. **`QO512+V52` is `RANK-SUPER-ADDITIVE`** — teacher-forced below `V52`'s own. E21's only datum
   says so, and it is registered even though it is the outcome that kills §2's configuration.
4. **`V52` reads teacher-forced ≥ `107`** — i.e. at least `COMPARABLE`. This is what *BPB ordering*
   predicts: `V52` costs `+0.141846`, less than the best ternary head's `+0.170414`, and those
   read `107`–`119`. **E21 §4a says BPB does not order across axes and therefore predicts this
   will miss.** Two published laws, opposite predictions, one measurement — registered as such so
   whichever way it falls is a result and not a rationalisation.
5. **`STACK` reads `AT-FLOOR` free-running and `< 107` teacher-forced.** E18 already measured the
   all-ternary rung at the floor; nothing here should rescue it.

**THE REGISTERED ALTERNATIVE.** *If `QO512+V52` reads teacher-forced ≥ 107 — inside or above E20's
ternary band — then the first configuration this programme has derived inside the 50 tok/s budget
is also no worse per step than ternarizing a single organ, and the healing target for the T4
sessions becomes **the assembled 1.5 B**: a model that is already budget-feasible, already right
two-thirds of the time per step, and small enough to heal in weeks rather than months. That is a
concretely scoped GPU job instead of an open-ended one, and it is written here before the data
exists.*

**And the third outcome.** *If `G-S1` or `G-S2` fails, E22 decides nothing about composition and
becomes a report on a reproducibility failure across three runners. Registered as a real
possibility, not left available as an excuse.*

## 8. What E22 will not be able to claim

- **No speed claim, no timing. `6.79 tok/s` stays exact.** §2 is a budget derivation. Nothing is
  exported, `engine.c` still has no factored matvec (two GEMVs with an intermediate of size `r`)
  and `QWENDON1` no kind for one, and **the carve's router is an ORACLE** — it reads the true
  activation mass, so it is a ceiling. A null under an oracle is strong; a pass is **not** a
  runnable result.
- **Not the goal's scale.** §2 says plainly that the same two cuts leave the 7 B at `4.090 G`, 4.0×
  the budget, and permit *negative* FFN once `attn+head` is counted. E22 is about whether a
  budget-feasible configuration can work at all.
- **No healing.** E22 fits and composes; it does not train.
- **Not an optimum.** `r = 512` and `k = 133` are two published points, not a swept frontier.

## 9. Cost

One `H` capture (57 organs, ~220 s, E21's pass), D0c's cached partition read from disk, then seven
arms × (BPB on the frozen 24×512 heldout + 160 free-running + 160 teacher-forced positions).
**Estimated 50–80 minutes, CPU only, one job.** Smoke first, then the run.
