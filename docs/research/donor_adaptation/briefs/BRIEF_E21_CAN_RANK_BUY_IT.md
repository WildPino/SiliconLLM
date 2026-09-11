# E21 — can RANK buy what precision and sparsity could not?

**Pre-registration. Nothing in §§3–7 has been measured.** §2 is arithmetic over already-measured
quantities and is published here, before the run, exactly as E19 part A was.

---

## 0. Why this experiment and not healing

Healing/QAT has been the first owed item since E18 §9 and needs GPU. **It is not being skipped —
it is being aimed.** E19 proved that every carve this programme has built is FFN-only and that
FFN-only carving cannot reach 50 tok/s at any depth: on the Coder-7B `attn+head` alone are
**`1.367 G`** active weights against a 50 tok/s budget of **`0.982–1.060 G`**, so a 7B with its FFN
deleted outright still runs at 35.9–38.8 tok/s. **A carve that reached the target would have to cut
attention and the output head too, and that has never been attempted.**

So before spending GPU weeks healing a structure, the structure has to exist. E21 asks whether one
does.

**And E20 supplied the instrument that makes the question answerable.** Free-running greedy
agreement has no resolution below the floor — E17, E18 and E19 all read `3`–`12` of 160 and could
not order the interventions. E20 part B's teacher-forced top-1 spreads the same arms over
`107`–`119` of 160, fires exactly on the known-positive (`160/160` for `base` and `ID`), costs
`459 s` for ten arms, and is tied to free-running by an exact identity (`G-B1`, `50/50`). **E21 is
the first experiment that can rank structural interventions rather than only reject them.**

## 1. The axis nobody here has priced

Three axes have been measured: **precision** (T1, T2, T2b, E12, E15, E16, E18, E20 — the format is
the constraint), **sparsity/conditional activation** (D0, D0c, Probe-4, E19 — FFN-only, cannot
reach the budget), and **basis** (T3 — rotation hurts; D2 — spectra only).

**Rank has never been measured end-to-end.** D2 computed spectra for `q_proj`, `o_proj`,
`gate_proj`, `down_proj` on layers 0, 13, 27 of the 1.5B and stopped there: no BPB, no ranking, no
`lm_head`, no `k/v`. Nothing in this programme has ever replaced a donor matrix with a low-rank
factorisation and measured what came out.

## 2. Part A — the budget, arithmetic, published before the run

Every weight charged as **ternary** (`0.500000` B/weight, recomputed from the artefact by E18's
runner, not assumed). Budget for 50 tok/s: **`0.982–1.060 G` active weights/token** (E18 §31).

| organ | Qwen2.5-1.5B | Qwen2.5-Coder-7B | share of attn |
|---|---|---|---|
| FFN | `1156.1 M` (74.9%) | `5703.2 M` (80.7%) | — |
| attention, of which | `154.1 M` (10.0%) | `822.1 M` (11.6%) | — |
| — `q + o` | `132.1 M` | `719.3 M` | **86–88%** |
| — `k + v` | `22.0 M` | `102.8 M` | 12–14% |
| `lm_head` | `233.4 M` (15.1%) | `545.0 M` (7.7%) | — |
| **`attn + head`** | **`0.388 G`** | **`1.367 G`** | **1.29–1.39× the whole budget at 7B** |

**`q` and `o` are 86–88% of attention**, so cutting attention means cutting `q/o`. What a low-rank
factorisation costs at rank `r` (`D·r + r·D` for a square projection, `V·r + r·D` for the head):

| cut | 7B dense | r = 256 | r = 512 |
|---|---|---|---|
| `q/o`, all 28 layers | `719.3 M` | **`102.8 M`** (−616.6) | `205.5 M` (−513.8) |
| `lm_head` | `545.0 M` | **`39.8 M`** (−505.2) | `79.7 M` (−465.3) |

**At `r = 256` on the 7B, `attn + head` falls from `1.367 G` to `0.245 G`** — from 1.29–1.39× the
entire budget to **23–25% of it** — leaving `0.74–0.82 G` for the FFN, i.e. the FFN carved to
**13–14%**, which is inside the range D0c already builds routers for.

> **This is the first configuration this programme has derived that lands inside the 50 tok/s
> budget on a 7B-class donor.** E18 priced precision and found no rung both fast and good; E19
> priced FFN carving and found it cannot reach the budget at any depth. **Neither priced rank.**
> Whether the configuration is *usable* is what §§3–6 measure, and the honest prior is in §3.

## 3. The prior, stated before the run, and it is not favourable

D2's measured spectra (1.5B, weights only, three layers):

| matrix | rank for 90% energy | for 99% | stable rank |
|---|---|---|---|
| `L00.q_proj` | 445 | 918 | 7.42 |
| `L13.q_proj` | 547 | 1009 | 82.95 |
| `L27.q_proj` | 425 | 883 | 64.83 |
| `L00.o_proj` | **937** | 1262 | 101.14 |
| `L13.o_proj` | 634 | 1056 | 92.63 |
| `L27.o_proj` | 596 | 1026 | 47.75 |

**These matrices are not low-rank.** A plain rank-256 truncation discards well over 10% of the
Frobenius energy of every one of them, and `L00.o_proj` needs `937` of `1536` for 90%. On weight
error alone, `r = 256` is expected to be **comparable to or worse than ternarization**, which E20
measured at relative weight error `0.4449–0.6697`.

**Which is exactly why the arms are not plain SVD.** D4 established on this donor that weight-space
reconstruction is the wrong objective: its Hessian ablation recovered `0.483` with `real_H`,
**`0.0`** with `identity_H`, `−1.595` shuffled. E20 part B measured how anisotropic the relevant
input covariance is — the head's `H = E[h hᵀ]` spans eigenvalues `1.48e-1` to `3.00e8`. **A matrix
that is not low-rank in weight space can still be low-rank in the directions the data actually
occupies**, and activation-weighted low-rank is to rank what `R3`/`R5` are to precision.

So E21's verdict arms minimise **`‖(W − Wᵣ)X‖_F`**, not `‖W − Wᵣ‖_F`. Construction, fixed here:
take the calibration second moment `H = E[x xᵀ]`, form `H^{1/2}` by eigendecomposition with the
same damping `λ = 0.01·mean(diag H)` E20 used, SVD `W·H^{1/2}`, truncate to `r`, and map back —
`Wᵣ = (truncate(W·H^{1/2})) · H^{−1/2}`. This is the exact minimiser of the activation-weighted
error at rank `r`, so the plain-SVD arm is a genuine control and not a straw man.

## 4. Arms

All arms leave every untouched weight **bit-exact in fp32**. The question here is rank alone;
composition with ternarization is §7's owed item, not this run's.

| tag | what | control for |
|---|---|---|
| `base` | untouched donor | must reproduce `results/e6/ref.json` |
| `FULL` | factorise at `r = min(shape)`, i.e. no truncation, through the same code path | **planted control**: must be token-identical to `base` |
| `H-SVD-256` / `H-SVD-512` | `lm_head`, plain SVD | isolates *data* from *rank* |
| **`H-ACT-256` / `H-ACT-512`** | `lm_head`, activation-weighted | **verdict arm for the head** |
| `QO-SVD-256` | all 28 layers' `q_proj`+`o_proj`, plain SVD | |
| **`QO-ACT-256` / `QO-ACT-512`** | all layers' `q_proj`+`o_proj`, activation-weighted | **verdict arm for attention** |
| **`BOTH-ACT-256`** | head **and** `q/o` together | **the §2 configuration, measured** |

`tie_word_embeddings = True` on this donor, so the head is untied and cloned first, exactly as
`t2b_organs.py:149` does. `H` for each organ is captured over the frozen calibration slice
(32×512, seed 42424) — the same pass and the same code path E20 used, `t2b_organs.capture`.

## 5. Gates

- **`G-R0`** — `base` reproduces E6's reference at `160/160` and BPB `0.767595`. Otherwise **VOID**.
- **`G-R1`** — `FULL` is **token-identical** to `base` with BPB difference `0`. Proves the
  factorise-and-reassemble path is lossless at full rank. Otherwise **VOID**.
- **`G-R2`** — the activation-weighted construction beats plain SVD **in BPB** at equal rank, on
  both organs. If it does not, the construction is suspect and that is **reported as a finding**,
  not absorbed — a method that cannot win on the objective it optimises has not been shown to work.
- **`G-R3`** — **the verdict**: `BOTH-ACT-256` read on **both** metrics, against §6's bands.
- **`G-R4`** — teacher-forced top-1 for every arm, with `base`/`FULL` required at `160/160` before
  any null is read (E20 part B's instrument, fired on its known-positive first).

## 6. Bands — fixed here, before the run

Free-running, unchanged since E17/E18 and used by E18, E19 and E20: floor **`12/160`** (E18 part A,
best constant-token predictor), margin **`2`** (E17), **`AT-FLOOR ≤ 14`**, **`RANKS ≥ 80`**.

Teacher-forced top-1 is new as a *banded* metric and needs its band fixed now, from measured
quantities only. E20 measured eight ternary heads at **`107`–`119` of 160** (67–74%) and the intact
donor at `160/160`. So:

- **`RANK-IS-CHEAPER`** — `BOTH-ACT-256` teacher-forced **> 119**, i.e. above every ternary head
  E20 measured.
- **`RANK-IS-COMPARABLE`** — teacher-forced in **`107`–`119`**, inside E20's ternary band.
- **`RANK-IS-WORSE`** — teacher-forced **< 107**.

## 7. Predictions

1. `G-R0` and `G-R1` fire. `FULL` is exactly lossless (it is an orthogonal factorisation).
2. **`G-R2` holds and by a large margin** — larger than the quantization analogue's, because §3
   says the weight spectra are unfavourable while §3 also says the data covariance is extremely
   anisotropic, and that gap is precisely what the weighting exploits.
3. **`H-ACT-256` lands `AT-FLOOR` free-running** and **inside `107`–`119` teacher-forced.** The
   head survived ternarization at 110/160 with a `0.45–0.62` relative weight error; rank-256 is a
   comparable perturbation by a different mechanism.
4. **`QO-ACT-256` is *less* damaging than `H-ACT-256`**, because attention output is summed over
   28 layers and a residual stream tolerates per-layer error that a final argmax does not.
5. **`BOTH-ACT-256` → `RANK-IS-COMPARABLE`.**

**THE REGISTERED ALTERNATIVE.** *If `BOTH-ACT-256` reads `RANK-IS-CHEAPER` — teacher-forced above
119 — then rank is a strictly better axis than precision for this donor, the §2 configuration is
both inside the 50 tok/s budget and less damaged than anything ternarization produces, and the
healing target for the T4 sessions changes from "a ternary donor" to "a low-rank donor", which is
a different and much better-conditioned optimisation. This would be the first structural result in
this programme that moves toward the goal rather than closing a route, and it is written here
before the data exists.*

**And the third outcome.** *If `G-R2` fails — activation weighting does not beat plain SVD — then
E21 decides nothing about rank and is a report on a mis-specified construction. Registered as a
real possibility rather than left available as an excuse afterwards.*

## 8. What E21 will not be able to claim

- **No speed claim.** Nothing is exported and no timing will be taken. **`6.79 tok/s` is
  untouched.** §2 is a budget derivation, not a measurement, and a low-rank factorisation has its
  own kernel cost that `engine.c` does not currently implement — two GEMVs instead of one, with an
  intermediate of size `r`. That cost is **not** priced here and is §9's first owed item.
- **No composition with ternarization.** Every arm is fp32. Rank × precision is the product that
  matters and it is not measured here.
- **One donor, one scale.** 1.5B. E16's non-monotonicity in scale applies to any claim made here.
- **No healing.** E21 fits; it does not train.
- **Not an optimum.** Activation-weighted truncation is the best rank-`r` approximation *of that
  objective*, not a proof that no better rank-`r` model exists. A null is evidence, not a theorem.

## 9. Cost

One calibration pass (`t2b_organs.capture`, ~210 s), then per-organ `H^{1/2}` and SVDs — the head
is 151936×1536, `q/o` are 56 matrices of 1536×1536 — then nine arms × (BPB on the frozen 24×512
heldout + 160 free-running positions + 160 teacher-forced positions). **Estimated 90–150 minutes,
CPU only, one job.**
