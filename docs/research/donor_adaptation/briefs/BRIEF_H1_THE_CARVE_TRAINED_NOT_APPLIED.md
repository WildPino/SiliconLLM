# BRIEF H1 — the carve TRAINED rather than APPLIED

**Pre-registered. Pushed BEFORE the trainer exists and before one GPU-second is spent.**
Nothing below may be changed after a number is seen; changes go in a dated addendum, as in E43.

Requested in `COMMUNICATION.md` APERTO 0 and granted by the user on 2026-09-12.
**Cost: two T4 sessions of 2.8 h (~5.6 GPU-h).** Both are needed and §7 says why.

---

## 0. The question, in one sentence

**E37 measured what it costs to APPLY the ternary+carved FFN to a trained donor. H0 measured
that a ternary format can be TRAINED INTO. Nobody has put those two facts in one run — H1 does,
and asks whether training with the mask ON recovers what applying it destroys.**

### 0.1 Gate discipline, inherited and binding

Two rules carry over and are in force here:

1. **From E43 §0.1** — *every gate below is either ORDINAL (direction only, no tolerance) or uses
   an anchor this programme has already measured on this exact instrument.* **No gate in H1
   invents a number.** Every threshold in §5 is a BPB already on disk.
2. **From H0 run 3 (`T4_HEALING_PROPOSAL.md` §12)** — *a gate is a floor, not a progress meter.*
   H0 was nearly closed early because its progress was read off the gate metric, which had
   saturated. **H1 therefore registers its progress metrics separately from its gate**, and they
   are the continuous one (BPB) and the one with headroom (free-running). §5.3.

---

## 1. What is actually broken — and it is NOT mostly the carve

The programme has been saying "the carve costs +0.5537 BPB". True, but it is the **small half**,
and stating it alone has been misleading. Read off `results/e37_sparsity_cost.json`, one
instrument, the frozen 51,870-byte slice:

| state | BPB | damage added |
|---|---|---|
| fp32 dense donor | **0.767595** | — |
| ternary FFN, **all 256 groups on** (`K256`) | **3.475707** | **+2.708112** |
| + carve to `k=16` / 6.25% (`K16`) | 3.986801 | +0.511094 |
| + carve to `k=3` / 1.17% (`K3`) | 4.029398 | +0.553692 |
| chance | 4.069819 | — |

**Ternarising the FFN post-hoc costs 2.708 BPB; carving it then costs another ~0.51.** The carve
ladder is nearly FLAT from `k=64` (3.927) to `k=1` (3.990) — every rung sits just under chance.
So the object H1 must heal is **+3.22 BPB of combined damage**, of which the carve is 16%.

**This is why H1 is the informative run and not a formality.** H0 healed +2.045 BPB of damage
(ternary `q/o`) down to +0.042, removing 97.9%. H1 asks whether the same mechanism survives a
damage 1.6× larger, on the organ that holds nine tenths of the weights, with a mask on top.

---

## 2. Feasibility, computed BEFORE asking for the hours

Training the FFN means fp32 masters + grads + two AdamW moments over the trained organs.

| config | FFN params | AdamW state | fits a 16 GB T4? |
|---|---|---|---|
| Qwen2.5-1.5B, **all 28 layers** | 1,156,055,040 | **17.2 GB** | **NO** |
| Qwen2.5-1.5B, **8 of 28 layers** | 330,301,440 | 4.9 GB | yes |
| Qwen2.5-0.5B, all 24 layers | 313,786,368 | 4.7 GB | yes |

**The obvious version of H1 does not fit and would have died mid-session.** Registered choice:
**Qwen2.5-1.5B with the FFN trained in 8 of 28 layers**, because it keeps H0's donor and H0's
frozen eval slice, so H1's number is directly comparable to H0's `0.810022`. The 0.5 B variant
is architecturally complete but throws away every anchor this programme owns.

**Which 8 layers is registered NOW, because choosing them after the fact is how you flatter a
result.** Evenly spaced across depth: **3, 6, 9, 12, 15, 18, 21, 24.** Not E27's `MINRES` band
(12–18), which is the *least active* quarter and therefore the easiest place to carve — picking
it would be choosing the arm that wins.

**The cost of that choice, stated:** the carve exists in 8 layers, E37's anchors carve all 28, so
**E37's numbers are NOT the comparison.** §4 re-measures a matched anchor.

---

## 3. What H1 trains

Resumes `q/o` from **H0 run 3's output** (`h0_trained3.npz`, 1000 steps, BPB 0.810022) — those
organs are already healed and re-healing them is not what the hours are for. Added on top, in
the 8 registered layers only:

* **ternary `gate` / `up` / `down`**, straight-through, the same R3 rule and STE as H0 — 24 new
  organs, 330,301,440 masters;
* **the carve ON during training**, `E = 256` groups from `density/results/d0c_labels/labels_E256.npz`
  (equal groups of 35, the partition every carve result in this programme uses);
* **a router trained JOINTLY with the experts** — §6.

`k = 16` of 256 = **6.25% activation**. Registered because it is what the box now affords at
10 B after E39/E40 moved the floor (~4.5–6%), and because E37 has a measured applied anchor at
exactly that rate. Not `k=3`: that was the budget before E39, and the ladder is flat anyway.

**Everything else is frozen** — embeddings, the tied head, norms, attention outside `q/o`, and
the FFN of the other 20 layers.

---

## 4. The anchors, all re-measured here and none quoted

Per the programme's rule, every anchor is re-run on `h0_eval.py` at the time of the verdict
rather than copied out of an old file:

| anchor | what it is | expected |
|---|---|---|
| `intact` | the fp32 donor — **planted control, must fire at 160/160** | 0.767595 |
| `h0-run3` | H0's trained state, **no FFN touched** — H1's start line | 0.810022 |
| **`applied-8L`** | **the matched control: the same 8 layers, ternary + carved at `k=16`, APPLIED post-hoc, NOT trained** | **unmeasured — §5.1** |
| `chance` | — | 4.069819 |

**`applied-8L` is the number H1 is actually arguing with**, it does not exist yet, and it is
**free** — CPU only, no GPU. It is measured BEFORE the T4 runs, and its value is recorded in an
addendum at that time so it cannot be adjusted afterwards.

---

## 5. Gates and bands

### 5.1 Planted controls — these fire BEFORE any H1 null counts

Per the standing law (*an instrument must fire on a known-positive before its nulls count*):

| gate | demand | why it exists |
|---|---|---|
| `G-H1a` | with **`k = E = 256`** (all groups on) the carved forward is **BIT-IDENTICAL** to the uncarved ternary forward | the only thing that rules out a silently mis-wired mask or a permutation bug — the E27 `L28-PASSTHROUGH` trick |
| `G-H1b` | the **first optimizer update is actually APPLIED** (read from AdamW's own state, not from step count) | this is H0's `G-H0e`; it caught a real bug that would have faked H0's null |
| `G-H1c` | the FFN masters **move** after that first applied update | H1 trains a different organ than H0; the same trap is open |
| `G-H1d` | `applied-8L` **reproduces E37's direction** — carving 8 layers is strictly worse than not carving them, on this instrument. **ORDINAL, no tolerance** | proves the matched anchor is measuring the carve and not an artifact of the 8-layer restriction |

**If `G-H1a` does not fire, H1 has no result** and the hours are not spent — it would mean the
mask is not doing what the measurement claims.

### 5.2 The gate — ORDINAL, and the anchors are measured

**`G-H1`: trained BPB < `applied-8L` BPB.** Direction only, no tolerance, both measured on the
same instrument in the same session. This is the minimum honest claim — *training beats
applying* — and it is the whole point of the run.

Bands, every threshold an anchor already on disk (E43 §0.1):

| band | condition | reading |
|---|---|---|
| `CARVE-NOT-TRAINABLE` | BPB ≥ `applied-8L` | training buys nothing; the T4 branch closes |
| `TRAINING-HELPS` | `3.475707` ≤ BPB < `applied-8L` | the carve heals, ternarisation does not |
| `CARVE-IS-TRAINABLE` | `0.810022` < BPB < `3.475707` | both heal — **the result that opens 10 B** |
| `CARVE-IS-FREE` | BPB ≤ `0.810022` | indistinguishable from H0, i.e. the FFN carve costs nothing under training |

### 5.3 Progress metrics, registered SEPARATELY from the gate

**Because of H0 run 3.** The gate above is a one-shot comparison at the end. The question *"is
it still improving, does session 2 buy anything"* is answered **only** on:

* **BPB** (continuous), and
* **free-running generation** out of 160 (H0 run 3: 8 → 15; the intact donor: 160).

**Teacher-forced `tf` is recorded and is NOT a progress signal.** No decision to continue, stop
or extend H1 may cite `tf`, and no in-training fp16 probe may be used for anything except
confirming the run is alive and finite.

---

## 6. The router — and this is the MoE question, named as such

Pointed out by the user on 2026-09-12: *"mi sembra che non stai considerando i MoE."* Checked,
and the criticism lands. **Everything this programme has measured is post-hoc MoE**: in
`engine/carve_common.py` the "experts" are slices of ONE shared pretrained FFN (the `F` axis cut
into 256 equal groups and permuted contiguous), and the router has only ever been an **oracle**,
a **seeded synthetic matrix**, or a **post-hoc ridge** (E23, E37) — **never trained with the
experts.** E37/E38/E40 close *that*, and the documents have been saying "the carve is closed"
without the qualifier. It is the post-hoc carve that is closed.

**H1 therefore trains the router JOINTLY with the experts** — per-layer, block-input → group
score, top-`k` with straight-through on the gate so the router receives gradient. That is the
first real MoE on this branch.

**It is deliberately ONE arm and not an ablation.** A frozen-router arm would re-ask a question
E38 already answered negatively (a per-token **oracle** router is not enough post-hoc), and
5.6 GPU-h does not buy two arms. Consequence, registered: **if H1 fails, it cannot say whether
the carve or the router failed.** That is the price of the budget and it is stated in advance
rather than discovered in the write-up.

**The ceiling on all of this, so nobody over-reads a win.** At `R128`/`k=3` the FFN is 34.2% of
the charged weights and head+attention are 60.7%. **If the FFN were free the engine gains
1.52×** (112.73 → ~171 tok/s). MoE is necessary and **not sufficient**; that is why E43 went
after the head.

---

## 7. Why BOTH sessions, and it is not padding

**H0 run 3 is the evidence.** After 500 steps H0 looked flat on the metric being watched. It was
not: the *second* 500 steps removed **a quarter of the damage that survived the first 500** and
nearly doubled free-running. **A one-session H1 would be read the way a one-session H0 was read,
and that reading was wrong.**

Registered: **session 1 produces no verdict.** Its output is a checkpoint and a BPB, reported as
an interim point on a curve with two points. The gate in §5.2 is evaluated **after session 2**.

---

## 8. Predictions — registered, falsifiable, and some will be wrong

1. `applied-8L` lands between **3.10 and 3.55** — a fraction of E37's all-28-layer damage roughly
   in proportion to 8/28, plus the carve.
2. **`G-H1` fires**: trained beats applied. *(If this misses, the T4 branch closes and I say so.)*
3. H1 lands in **`TRAINING-HELPS`**, not `CARVE-IS-TRAINABLE` — 1000 steps healed +2.045 BPB in
   H0; here the damage is +3.22 on a bigger organ with 20 layers frozen.
4. **Free-running stays in `AT-FLOOR` or `PARTIAL` (≤ 40/160).** The generator is the axis that
   has never recovered, in any arm, at any scale.
5. The jointly-trained router beats `k/E` recall (0.0625) by a wide margin — **but its
   contribution is not separable from the experts'**, per §6.
6. **Weakest prediction, flagged as such:** that 8 trained layers out of 28 move the whole-model
   BPB at all. Twenty layers keep a dense fp32 FFN; they may simply dominate the measurement and
   leave H1 unable to resolve its own effect. **If §5.2's bands all collapse inside the
   dispersion of the 8-layer restriction, H1 is UNRESOLVABLE and says so.**

---

## 9. What H1 cannot say, whatever it returns

* **Nothing about 10 B.** 1.5 B, 8 layers. Like H0 it is a **structure validation**.
* **Nothing about speed.** No timing is taken, no artifact is exported, `112.73 tok/s` is
  untouched. H1 is a quality question end to end.
* **Nothing about the head**, which E43 measured as the largest single term of the floor and
  which no trained arm has ever touched.
* **Nothing about `STACK`** — the shippable object also ternarises the head, which H1 does not.
* **It cannot separate the carve from the router** (§6), and it cannot separate ternarisation
  from carving unless `applied-8L` and a `k=E` trained arm are both taken — and the second does
  not fit in the budget.

---

## 10. Deliverables and the stop

1. This brief, pushed **before** the trainer exists.
2. `s1/h1_qat.py` — H0's trainer plus the carve, the FFN organs and the joint router. `G-H1a`–
   `G-H1d` live in its self-test.
3. `applied-8L` measured on **CPU, free**, and recorded in **addendum A before the T4 runs**.
4. A CPU smoke that fires every gate on a toy shape.
5. The bundle, its `MANIFEST.json` with sha256s, and **`RUN.md` with the exact command.**
6. **STOP. I do not launch the T4 sessions** — I hand over the ready command, per the standing
   rule that the user launches the long runs.

---

# ADDENDUM A — `G-H1c` was MIS-SPECIFIED, and the joint router is UNSTABLE

**Written and pushed BEFORE the T4 runs and before any H1 number on the real donor exists.**
Everything here comes from a CPU smoke on a *toy* shape (`D=32, F=128, E=8, k=2`, random
weights), whose only job is to check wiring and find design faults cheaply. **No number in this
addendum is a result about the donor, and none of them may be quoted as one.**

## A.1 What fired, and stands

| check | result |
|---|---|
| `G-H1a` — `k = E` bit-identical to the uncarved ternary FFN | **FIRES**, max abs difference **exactly 0.0** |
| the carve actually masks at the real `k` | **FIRES** — output differs, `k*GSZ` neurons live, gate values exactly `{0, 1}` at init |
| gradient reaches `gate`, `up`, `down` **and** `router` | **FIRES** — all four non-`None`, all non-zero |

## A.2 `G-H1c` as registered in §5.1 is WRONG, and it is replaced

§5.1 registered *"the router learns: recall@k strictly above `k/E`"*. **That gate measures the
wrong thing and it must not be used.** `recall@k` scores agreement with the top-`k` groups **by
activation mass**, and a jointly trained router is not trying to match mass — it is minimising
loss, and it is free to pick a different set and compensate with its gate weights. Measured on
the toy: after training, recall sat at chance (0.2646 vs `k/E` = 0.2500) **while the router was
simultaneously beating the exhaustive best fixed pair on reconstruction loss.** A gate that a
working instrument fails is not a gate — that is E42's error, and E43 §0.1 exists to stop it.

**Worse, the yardstick is one this programme has already retired.** E38 handed the carve a
per-token mass **oracle** and it still read above chance. Scoring a router by how well it
imitates that oracle asks it to reproduce a known-insufficient criterion.

**`G-H1c` is REPLACED, ORDINAL, no tolerance:**

> **`G-H1c` (revised)** — on **held-out** tokens, the jointly trained router must be strictly
> better than **`STATIC`**: the same model with the top-`k` groups chosen once by global
> activation mass over the calibration stream and used for *every* token.

That is E23's comparison (`V52-STATIC` read 99/160 against the linear router's 110) and it
prices the only thing a router can sell — **the per-token decision.** If it cannot beat a fixed
selection, the router is not earning its 5.1% of the charged weights.

## A.3 The instability, found before the GPU and not after it

Training the router is **not stable**, and this would have consumed a whole T4 session. Toy,
held-out loss, `STATIC` control at ~0.015:

| router LR | 50 steps | 300 steps |
|---|---|---|
| `3e-2` | 0.104 | 0.132 |
| `3e-3` | **0.0141 — beats `STATIC`** | 0.089 |

It is **not overfitting**: 64 and 2,048 training sequences behave identically. It is divergence.
Adding the standard Switch load-balancing auxiliary loss spreads occupancy as intended (max
group share 0.23 → 0.15) but **does not fix it** — held-out loss still walks 0.0136 → 0.0861
between steps 50 and 300, because on a random-weight toy the best policy is to keep picking the
heavy groups and load balancing actively forbids that.

**Registered consequences:**

1. **The router gets its OWN optimizer param group with its own LR**, separate from the experts.
2. **That LR and the auxiliary-loss coefficient are fixed by a CPU smoke on the REAL donor
   before the T4 runs, and recorded in an addendum at that time.** They are **not** tuned on
   the T4, and **not** tuned on the toy — the toy has random weights, no structure to route on,
   and §A.3's numbers are therefore evidence about *stability*, not about *what setting to use*.
3. **If the CPU smoke cannot find a setting where `G-H1c` (revised) fires on the real donor,
   H1 does not launch** and the T4 hours are handed back unspent. A router that cannot beat a
   fixed selection on CPU will not learn to on a T4.

## A.4 What does NOT change

The question (§0), the damage decomposition (§1), the 8 registered layers and the reason for
them (§2), `k = 16` (§3), the `applied-8L` matched control (§4), the **ORDINAL gate `G-H1` and
its bands** (§5.2), the separation of progress metrics from the gate (§5.3), both sessions
(§7), and every scope limit in §9. **§6's disclosure is now sharper, not weaker:** H1 still
cannot separate the carve from the router, and §A.2 means a *failure* of `G-H1c` would at least
say the router half is the part that did not earn its place.
