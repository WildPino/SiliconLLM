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

---

# ADDENDUM B — addendum A misnamed the gate it was correcting

**Same day, before any run. Correcting my own error rather than editing §A.2 in place.**

**§A.2 says "§5.1 registered `G-H1c` as recall@k". It did not.** §5.1 registers `G-H1c` as
*"the FFN masters **move** after that first applied update"* — the H0-style trap check for the
new organ. **That gate is correct, it is untouched, and §A.2 had no business replacing it.**

The recall-vs-mass claim was **prediction 5 in §8**, a prediction, not a gate. So:

1. **`G-H1c` stands exactly as first registered** (FFN masters move). `G-H1a`, `G-H1b` and
   `G-H1d` are likewise unchanged.
2. **The router gate introduced by §A.2 is renamed `G-H1e`** and added to §5.1 rather than
   replacing anything:

   > **`G-H1e`** — on **held-out** tokens the jointly trained router must be strictly better
   > than **`STATIC`** (top-`k` chosen once by global activation mass and used for every
   > token). ORDINAL, no tolerance. E23's comparison; it prices the per-token decision, which
   > is the only thing a router sells.

3. **Prediction 5 of §8 is RETIRED, not scored.** It predicted the router would beat `k/E`
   recall "by a wide margin" on a yardstick §A.2 then showed to be the wrong one. A prediction
   measured on a retired metric cannot be marked hit or missed, and marking it either way would
   be scoring myself against a ruler I had already thrown out. **It is replaced by nothing** —
   `G-H1e` is a gate, and gates are not predictions.
4. Everything else in addendum A stands: the gate-formula fix, §A.1's three wiring checks, and
   §A.3's instability finding with all three of its registered consequences.

**Why this is written down rather than quietly fixed.** Addendum A was pushed at `35b755b`. An
addendum that misstates which gate it is overriding is exactly the kind of small, plausible
error this programme fails on — and the fix costs one section, whereas discovering at verdict
time that two different gates share a name costs the run.

---

# ADDENDUM C — `applied-8L` MEASURED, and it BREAKS §5.2's band table

**Written and pushed BEFORE the T4 sessions, as §10 item 3 requires.** Everything here is CPU,
fp32, free, on the frozen 24×512 slice. `benchmarks/donor_adaptation/s1/h1_applied.py`,
`results/h1/h1_applied_8L.json`.

## C.1 The instrument first, then the number

Three checks before any arm counts, all exact:

| check | read | demanded |
|---|---|---|
| frozen slice | 51,870 scored bytes, `4.22945205479452` B/token | the registered `ids_sha a1a48dc9…` |
| `intact` | **0.767595** | E12's dense anchor — **planted control, FIRES** |
| `h0-run3` | **0.810022** | H0 run 3's own reading — H1's start line reproduced |
| **`G-H1a` on the REAL donor shape** | max abs diff **0.0e+00** | bit-identical at `k = E`, zero tolerance — **FIRES** |

`G-H1a` had only ever fired on a toy. It now fires on `D=1536, F=8960, E=256, GSZ=35` with the
H0 organs installed, which is the configuration the T4 run will train.

| arm | BPB | vs previous |
|---|---|---|
| `intact` | 0.767595 | — |
| `h0-run3` (q/o trained, FFN untouched) | 0.810022 | +0.042427 |
| `ternary-8L` (ternary FFN on 8 layers, `k = E`, no carve) | **0.947851** | **+0.137829** |
| **`applied-8L`** (`k = 16`, E37's own fitted router, hard gate) | **1.096636** | **+0.148785** |

**`applied-8L = 1.096636` is the number H1's gate argues with. It is registered here and may
not be adjusted afterwards.**

## C.2 `G-H1d` FIRES

`applied-8L` 1.096636 **>** `ternary-8L` 0.947851, **+0.148785**, ordinal and no tolerance. The
matched control is reading **the carve** and not an artifact of the 8-layer restriction. For
scale: this programme's own seed constant is **σ ≈ 0.005** (R1 calibration), so the carve cost
is about **30 σ** — comfortably measurable, which is what `G-H1d` had to establish.

## C.3 §5.2's BAND TABLE IS BROKEN, and the repair makes it HARDER

As registered, with `applied-8L = 1.096636` substituted in:

```
CARVE-NOT-TRAINABLE   BPB >= 1.096636
TRAINING-HELPS        3.475707 <= BPB < 1.096636      <-- EMPTY: 3.475707 > 1.096636
CARVE-IS-TRAINABLE    0.810022 <  BPB < 3.475707      <-- OVERLAPS the first band
CARVE-IS-FREE         BPB <= 0.810022
```

**`TRAINING-HELPS` is empty and `CARVE-IS-TRAINABLE` overlaps `CARVE-NOT-TRAINABLE`.** The
cause is exactly the category error this brief named when it invented `applied-8L` in the first
place: **`3.475707` is an ALL-28-LAYER number** (E37's ternary-everywhere arm) used as a
threshold in an **8-layer** comparison. §2 wrote *"E37's all-28-layer anchors are NOT the
comparison"* and then §5.2 used one as a band edge.

**The repair is to swap that one threshold for its matched counterpart**, `ternary-8L`,
measured in the same session, on the same instrument, on the same 8 layers:

| band | condition | reading |
|---|---|---|
| `CARVE-NOT-TRAINABLE` | BPB ≥ **1.096636** (`applied-8L`) | training buys nothing; the T4 branch closes |
| `TRAINING-HELPS` | **0.947851** (`ternary-8L`) ≤ BPB < 1.096636 | the carve partly heals; ternarisation does not |
| `CARVE-IS-TRAINABLE` | **0.810022** (`h0-run3`) < BPB < 0.947851 | the carve fully heals AND some ternarisation does — **the result that opens 10 B** |
| `CARVE-IS-FREE` | BPB ≤ **0.810022** | indistinguishable from never touching the FFN |

Now monotone — `applied-8L > ternary-8L > h0-run3` — with every edge a number measured today.

**This is a TIGHTENING, and that is the point.** Under the registered table a trained result of
`1.05` would have fallen in `CARVE-IS-TRAINABLE`, i.e. *"the result that opens 10 B"*, because
that band ran all the way up to 3.475707. Under the repair the same `1.05` reads
`TRAINING-HELPS` — a far more modest claim. The old table would have **over-claimed on almost
any outcome**; the repair makes the good-news band roughly **twelve times narrower** (0.138 wide
instead of 2.666). A gate change that makes the answer harder, registered before the run, is
not a goalpost move — and **`G-H1`'s gate itself is untouched: trained BPB < `applied-8L`.**

## C.4 A defect in my own run, and the sensitivity arm it turned into

**I launched the first `applied-8L` WITHOUT `--stats`**, so `h1_qat.build_ffn` took its
`rms = ones` fallback — and that function's own docstring says what that means: *"the R3 rule
is activation-weighted and using ones here would silently change the format from the one the
engine ships."* That run measured a **different quantization** from E37's and **cannot be the
anchor.** It is kept, labelled, as `results/h1/h1_applied_8L_ONES_sensitivity.json`, and
`h1_stats.py` now captures the real calibration by importing `qwen_export.capture_act_rms` —
the definition the exporter and E1 already share — over T2/E23/E37's pinned
`(calib, 32, 512, 42424)`, on the plain donor, because E37 calibrated there.

**How much the defect was worth, measured rather than assumed:**

| arm | R3 (anchor) | `ones` | difference |
|---|---|---|---|
| `intact` | 0.767595 | 0.767595 | 0.000000 |
| `h0-run3` | 0.810022 | 0.810022 | 0.000000 |
| `ternary-8L` | 0.947851 | 0.946799 | **+0.001052** |
| `applied-8L` | 1.096636 | 1.100281 | **−0.003645** |

**Both differences are below this programme's seed constant σ ≈ 0.005**, and they point in
opposite directions. So on this donor at this shape the R3 activation calibration is worth
**less than noise** — it neither rescues nor damages the carve.

**And there is a derivation that explains why the effect is this small, which is the part worth
keeping.** `r3_actsearch` is **invariant to a global rescaling of `act_rms`**: with `ww → c·ww`,
the scale `a = (w·q·ww²)/(kept·ww²)` is unchanged and the error `(r²·ww²)` scales by `c²`
uniformly, so the argmin over `D_GRID` does not move. The two arms therefore differ **only
through the per-column profile inside a layer**, never through its magnitude — and the measured
answer is that the profile is worth under 0.004 BPB here. **This does NOT rescue the `ones` run
as the anchor**: E37 used the real profile and a control that does not match E37 is not a
control. It does mean the defect cost wall time, not correctness.

## C.5 What the number does to H1's stated motivation, and this is uncomfortable

**The prize is 0.287 BPB, not the 3.22 the brief's own header advertises.** `h1_qat.py`'s
docstring says *"H1 must heal +3.22 BPB combined, 1.6× what H0 healed"* — and `+3.22` is an
**all-28-layer** figure taken from E37. On the 8 layers H1 can actually afford, the entire
damage between H1's start line and its control is:

```
applied-8L 1.096636  −  h0-run3 0.810022  =  0.286614
   of which the carve  +0.148785
   and ternarisation   +0.137829
```

**Registered consequences:**

1. **The `+3.22` framing is RETIRED.** No H1 document may quote it as what this run heals. It
   describes a 28-layer object H1 does not build.
2. **The measurement is still comfortably resolvable** — 0.287 BPB is ~57 σ and the carve half
   alone is ~30 σ — so the smaller prize is a smaller *claim*, not a weaker instrument.
3. **A limitation that is now sharper, and it is NOT new, only quantified.** Eight layers of
   twenty-eight is what a 16 GB T4 affords (§2), and the damage does not look additive: E37
   reads +2.708 for ternarising 28 layers while 8 layers here read +0.138, which is far less
   than `28/8 ×`. Whatever H1 measures is therefore **a statement about training the carve at
   all**, not a per-layer coefficient to multiply up to 28 layers or to 10 B. §9's scope limits
   stand and this is the number that makes them concrete.
4. **`G-H1` is unchanged and still the right gate.** *Does training beat applying* is exactly
   the question 0.287 BPB of headroom can answer.

---

# ADDENDUM D — the router smoke: layer 3 says GO, and the SELECTION RULE is registered here, before the other seven layers run

**Written and pushed BEFORE layers 6–24 are measured.** Layer 3 is done and in
`results/h1/h1_router_smoke_L3.json`. This addendum exists because the remaining layers will
decide which setting ships, and the rule for reading them must be on disk first.

## D.1 What layer 3 says

`h1_router_smoke.py`, CPU fp32, real donor, H0's q/o installed, the real R3 calibration,
`E = 256`, `k = 16`, 4,096 held-out tokens, 300 steps per setting, 12,685 s.

**The smoke's own planted control FIRES first**: router gradient `1.361e+00`, router moved
`1.000e-03` on the real donor. Without that, "no setting beat STATIC" would be
indistinguishable from "nothing was ever trained" — and that reading would hand back the T4
hours for a wiring reason.

| | value |
|---|---|
| layer 3's uncarved output power | 0.13067 |
| **STATIC** (top-`k` once by global mass) | **0.108585 — 83.1% of the power** |
| untrained router (zeros, index tie-break) | 0.124098 |

**The carve at `k=16` destroys 83% of this layer's FFN output energy**, which is worth stating
plainly: `applied-8L`'s +0.149 BPB is what that costs after the residual stream and 27 other
layers absorb it.

| lr | aux | held-out | ratio vs STATIC | `occ_max` | |
|---|---|---|---|---|---|
| 3e-4 | 0 | 0.097016 | **0.8935** | **0.818** | beats |
| 3e-4 | 0.01 | 0.101815 | **0.9377** | **0.161** | beats |
| 3e-4 | 0.1 | 0.114619 | 1.0556 | 0.151 | |
| 1e-3 | 0 | 0.103940 | **0.9572** | **0.909** | beats |
| 1e-3 | 0.01 | 0.105409 | **0.9708** | **0.162** | beats |
| 1e-3 | 0.1 | 0.116434 | 1.0723 | 0.149 | |
| 3e-3 | 0 / 0.01 / 0.1 | 0.121–0.124 | 1.118–1.142 | 0.968 / 0.254 / 0.253 | |
| 1e-2 | 0 / 0.01 / 0.1 | 0.133–0.157 | 1.228–1.443 | **1.000** / 0.613 / 0.528 | |

**GO on layer 3: 4 of 12 settings beat STATIC**, and they are exactly the two lowest learning
rates at aux 0 and 0.01. This reproduces addendum A's instability finding on the real donor —
the router trains, and it trains only in a narrow corner.

## D.2 The BEST RATIO IS NOT THE SETTING TO SHIP, and why

Read `occ_max`, the fraction of tokens selecting the single most-used group. Uniform at
`k = 16` of `E = 256` is **0.0625**.

* **`aux = 0` wins the ratio and very nearly COLLAPSES**: 0.818, 0.909, 0.968, and at lr 1e-2
  exactly **1.000** — one group selected by *every* token.
* **`aux = 0.01` still beats STATIC and stays near-balanced**: 0.161, 0.162 — about 2.6×
  uniform.

**A collapsed router is a STATIC router with extra steps.** The whole thing a router sells is
the per-token decision (addendum B, and E23's `V52-STATIC` before it); a router that picks the
same group for 82–91% of tokens has mostly re-derived the control it is being scored against,
and in H1's real run — where the experts train too — it would starve every group it stops
selecting. Taking `0.8935` because it is the smallest number would be optimising the proxy and
losing the thing the proxy stands for.

## D.3 The selection rule, REGISTERED NOW

Applied to the eight-layer table once it exists, in this order:

1. **Eligible** = settings whose `occ_max` ≤ **0.5**, averaged over the measured layers. (0.5
   is 8× uniform and half of total collapse — a generous bar chosen to exclude *collapse*, not
   to pick a winner; layer 3's eligible set is exactly the `aux = 0.01` pair, and its
   ineligible set is exactly the `aux = 0` pair.)
2. Among eligible settings, take the one that **beats STATIC on the most layers**.
3. Ties broken by the **best mean ratio**.
4. **If NO eligible setting beats STATIC on a majority of layers**, that is recorded as such
   and the shipped setting is the best *eligible* one with the shortfall stated — not the best
   collapsed one.
5. **If no setting of either kind beats STATIC on any layer**, addendum A consequence 3 fires:
   **H1 does not launch and the hours go back unspent.**

**`G-H1e` itself is unchanged.** It is scored end-to-end by `h1_qat.py` at the end of the T4
run, on held-out LM loss, against STATIC. This smoke is a **local proxy** whose only job is to
choose `--router-lr` and `--aux`; a local win need not survive composition through 8 layers and
a head, and `G-H1e` may still fail.

## D.4 The grid is NARROWED for layers 6–24, and this does not force the answer

One layer cost **3.5 hours**; eight at twelve settings is ~28 h. The grid for the remaining
seven layers drops the settings that lost on layer 3 **by more than 11%** — every `lr ≥ 3e-3`
cell (ratios 1.118–1.443, all collapsing) and every `aux = 0.1` cell (1.056–1.443) — leaving

> **lr ∈ {3e-4, 1e-3} × aux ∈ {0, 0.01}**, ~6 h.

**Why this is a search narrowing and not a result narrowing:** all four survivors are kept,
including both `aux = 0` cells that §D.3 rule 1 makes **ineligible**. So the remaining layers
can still return NO-GO for every eligible setting, and the eligible pair can still lose to the
collapsed pair without that changing what ships. `--steps 300`, `--cap 4096` and the seeds are
held **identical to layer 3**, so all eight layers stay comparable cell by cell.

---

# ADDENDUM E — the rule was registered, the eight layers landed, and the rule FIRES ITS SHORTFALL CLAUSE

All eight layers are measured. `results/h1/h1_router_smoke_L3.json` (12,685 s) and
`results/h1/h1_router_smoke_L6-24.json` (29,316 s), 11.7 h of CPU in total, both with the
smoke's own planted control FIRING first (router gradient `1.361e+00` and `3.983e-01`, router
moved `1.000e-03`, on the real donor).

§D.3's rule is executed by `s1/h1_router_select.py` — a script that makes **no choices of its
own**; it reads the two JSONs and applies the five clauses in order. Its output is frozen at
`results/h1/h1_router_select.txt`.

## E.1 The eight-layer table

| setting | layers won | mean `occ_max` | eligible | mean ratio | Σ error | vs STATIC |
|---|---|---|---|---|---|---|
| lr 3e-4, aux 0 | **2 of 8** — 3, 24 | 0.7610 | no | 1.0082 | 1.478007 | **−1.43%** |
| **lr 3e-4, aux 0.01** | **2 of 8** — 3, 24 | **0.3014** | **YES** | 1.0379 | 1.533524 | **+2.27%** |
| lr 1e-3, aux 0 | 1 of 8 — 3 | 0.8096 | no | 1.0996 | 1.810509 | +20.74% |
| lr 1e-3, aux 0.01 | 1 of 8 — 3 | 0.2930 | YES | 1.0714 | 1.615651 | +7.75% |

**Rule 1** → eligible = the two `aux = 0.01` settings. **Rule 2** → `lr 3e-4, aux 0.01`, 2 layers.
**Rule 3** → not needed. **Rule 4 FIRES**: 2 of 8 is below the majority of 5.

> **SHIPPED: `--router-lr 0.0003 --aux 0.01`, with the shortfall stated — it beats STATIC on
> two of eight layers and loses on six.**

**Rule 5 does not fire** (settings do beat STATIC on layers 3 and 24), so addendum A
consequence 3 does not fire either: **H1 launches.** §E.4 says on what basis.

## E.2 THE RUNNER'S OWN HEADLINE IS NOT THE ANSWER, and this is the E43 trap again

`h1_router_smoke.py` printed `GO. Best setting: --router-lr 0.0003 --aux 0`. That line is
**superseded and must not be quoted.** Three reasons, all structural:

1. It ranks by layers-won only. The occupancy criterion lives in this brief, not in the script
   — deliberately, since §D.3 was written after the script.
2. The L6–24 process **never saw layer 3**, so its "1 of 7" tallies are over the wrong
   denominator.
3. Its 1-of-7 four-way tie was broken by list order, not by anything measured.

This is the same class of trap as E43's stale `G-E43A ... VOID` runner line: **a script's
printed verdict outlives the specification it was written against.** Recorded here, in
`results/h1/h1_router_select.txt`, and in the log note beside the JSON.

## E.3 THE RANK METRIC AND THE SCORE METRIC DISAGREE, and I am reporting both

E14 §3: every SCORE metric needs a RANK partner. Here the pair is registered (layers-won) and
computed (Σ error over the 8 layers) — and **they point opposite ways**:

* The shipped eligible setting is **+2.27% WORSE than STATIC in aggregate error mass.**
* The only setting that is *better* in aggregate (**−1.43%**) is `aux = 0`, which is
  **ineligible** — mean `occ_max` 0.761, and 0.998 on layer 24.

E14 §6 forbids promoting a post-hoc metric to a gate, so Σ error does **not** change what
ships. But it changes what may be *claimed*: **this smoke does not establish that a trained
router beats a static one in this regime.** It establishes that a trained router can beat
STATIC on 2 of 8 layers while losing aggregate error, and that the version which wins
aggregate error has collapsed onto one group. Anyone reading only §E.1's "SHIPPED" line would
have the wrong impression, which is why this section is not a footnote.

**A defect in my own rule, found by applying it.** Rule 1 averages `occ_max` *across* layers,
so a setting can pass the bar while collapsing *on* a layer — exactly what the shipped setting
does: `occ_max` 0.161 on layer 3 (a clean win) and **0.998 on layer 24** (a collapsed one). Its
two wins are one of each. Sensitivity run in the same script: under **per-layer** eligibility
the rule ships **the same setting**, because every alternative loses its wins too. The defect
is real, it is recorded, and **it is not load-bearing here** — I am not re-specifying the rule
after seeing the data.

## E.4 What the eight layers say that the rule did not ask about — WHERE THE CARVE ACTUALLY HURTS

| layer | uncarved power | STATIC error | error / power |
|---|---|---|---|
| 3 | 0.13067 | 0.108585 | 0.8310 |
| 6 | 0.12701 | 0.109205 | 0.8598 |
| 9 | 0.10499 | 0.093466 | 0.8902 |
| 12 | 0.09241 | 0.077691 | 0.8408 |
| 15 | 0.08570 | 0.072258 | 0.8431 |
| 18 | 0.16527 | 0.134604 | 0.8145 |
| 21 | 0.38677 | 0.287808 | 0.7441 |
| 24 | 0.97868 | 0.615886 | **0.6293** |

Two facts, neither of which H1's design anticipated:

1. **The damage is nowhere near evenly spread.** Layer 24 alone carries **41.1%** of the total
   carve error and layers 21+24 carry **60.3%**, because their uncarved output power is 3–8×
   the mid-stack's. `H1_LAYERS = (3, 6, 9, 12, 15, 18, 21, 24)` is an *even* spread over a
   *very uneven* target.
2. **Relative damage FALLS with depth** (0.89 at layer 9 → 0.63 at layer 24) while absolute
   damage rises. The carve is relatively kindest exactly where it costs most.

Both are single-run, single-donor observations on a frozen-expert proxy. They are **not** gates
and nothing is re-planned on them; they are logged because they are the first map of where the
`k = 16` carve's error lives, and they name an obvious follow-up (**a depth-weighted layer
choice**) that H1 will not take, because changing `H1_LAYERS` now would break comparability
with the `applied-8L` anchor 1.096636 that §C registered.

## E.5 Why H1 still launches on a 2-of-8 router result

The honest statement of what was and was not shown:

* The smoke is a **frozen-expert local proxy**: it trains the router alone against a per-layer
  reconstruction target, with the experts held fixed. H1's real run trains **both**, against LM
  loss. A router that cannot help frozen experts may still help jointly — it changes *which
  experts receive gradient*, which is the one mechanism the proxy is structurally blind to.
* That cuts both ways and is stated as such: it is equally the reason this GO is **weak
  evidence**, and it is why `h1_qat.py`'s refusal to start at `--router-lr 0` (addendum A
  cons. 2) stands rather than being relaxed to "just freeze the router".
* **H1's primary gate is `G-H1`, not `G-H1e`.** `G-H1` asks whether the *trained* carve beats
  `applied-8L` 1.096636 — that is about the **experts**. The router smoke only ever chose two
  hyper-parameters.

**Registered prediction, before the T4 hours are spent:** on this evidence I expect **`G-H1e`
to FAIL** — the trained router will not beat STATIC end-to-end. I am writing that down now so
that a failure cannot later be presented as expected-all-along, and so that a *pass* counts for
something: it would mean the joint regime does what the frozen-expert proxy says it cannot,
which is a result about MoE on this branch and not a hyper-parameter.

---

# ADDENDUM F — THE NULL CONTROL PRICED THE GATE FORM AT +0.80 BPB, AND TWO REGISTERED GATES WERE MEASURING IT INSTEAD OF TRAINING

**Written and pushed BEFORE the T4 sessions, and before any trained weights exist.** That
timing is the whole justification for changing anything, and §F.4 says so explicitly.

`h1_nullbundle.py` builds a bundle that is the donor's own experts plus E37's own routers with
**zero training** — by construction, exactly the model `h1_applied.py` measured as
`applied-8L`. Run through `h1_eval.py` it must decompose to zeros. It does:

```
arm-E   1.096636   (untrained experts + E37 router  + HARD gate)   experts  +0.000000
arm-ER  1.096636   (untrained experts + E37 router  + HARD gate)   router   +0.000000
trained-8L 1.898014 (same weights     + same router + SOFT gate)   gate form +0.801377
                                                        residual 0.0e+00
```

**The decomposition control FIRES.** And the third term is the finding.

## F.1 The soft gate costs +0.801377 BPB by itself

Nothing is trained in that bundle. The only difference between `1.096636` and `1.898014` is
the gate form: E37's hard `{0, 1}` against the renormalised

    g_e = k * p_e / sum_{j in sel} p_j ,   sum_sel g = k

The mechanism is not mysterious. The gates AVERAGE 1 across the selected groups but are not
each 1: with a peaked router a dominant group is amplified by up to `k * p_max / sum_sel p`,
which at `k = 16` can be a factor of ten or more, and the experts underneath were never
trained to expect it. At **~160 σ** against the programme's seed constant 0.005 this is not a
subtlety — it is the largest single effect measured anywhere in H1's apparatus.

**This is exactly the confound the decomposition was built to find, found at the only moment
it could be acted on honestly.**

## F.2 `G-H1` as registered compares gate FORM, not training

`G-H1` reads *trained-8L < applied-8L*. As implemented, the trained arm ran the **soft** gate
and `applied-8L` is **hard**. So H1 would have had to recover **+0.80 BPB of gate-form
handicap before training counted for anything** — a bar that has nothing to do with the
question §0 asks.

And there is a second, independent reason the soft number is the wrong one: **`engine.c` runs
the hard gate.** `carve_common` passes selected groups at exactly 1.0; `h1_applied.py`'s own
docstring says so and that is why the control was built hard. A soft-gated BPB measures an
object the deliverable cannot execute.

> **`G-H1` is scored on the HARD-gated trained model.** Same weights, same router, same `k`,
> gates in `{0, 1}` — like-for-like with `applied-8L`, and the form the engine ships.
> The soft-gated number is still measured and still reported, as the diagnostic it is.

**Training stays SOFT and that is not a contradiction.** A hard gate has no gradient to the
router at all — that is precisely why addendum A's first gate formula failed its planted
control. Train soft, evaluate hard, and *measure the gap*: `trained-8L(soft) − arm-ER(hard)`
is already a term of the decomposition, so the train/eval mismatch is quantified in every run
rather than assumed away. **If that term is large at the end of training, the trained model is
leaning on gate magnitudes the engine discards, and that is a result about the format.**

## F.3 `G-H1e` as registered CANNOT isolate routing, and is repaired

Worse than `G-H1`, because it is not merely handicapped — it is measuring the wrong thing
entirely. `STATIC` installs a fixed set of `k` groups with **gates exactly 1**. The router arm
ran the **soft** gate. So `G-H1e` compared *soft router* against *hard fixed selection*, and
the null control shows what that is worth: **router 5.564279 vs STATIC 3.435791 nats/token,
"FAILS", with zero training and E37's own router.** A 62% gap produced by gate form alone.

> **Both arms of `G-H1e` are measured with the HARD gate.** Only the *selection* differs —
> per-token router against a fixed set — which is the only thing `G-H1e` ever claimed to test
> (addendum B, and E23's `V52-STATIC` before it).

This does not rescue the router. Addendum E's expectation stands and is unchanged: the CPU
smoke gave 2 of 8 layers and +2.27% aggregate error, so **`G-H1e` is still expected to FAIL** —
but it will now fail, or pass, on routing.

## F.4 Why this is a legitimate change and not moving the goalposts

The test this programme applies to itself (E36 run-2 rule, E40 addendum A precedent, addendum
C's own repair) is whether a result could have influenced the change. Here:

1. **No trained weights exist.** H1 has not run. There is no H1 number to make this choice
   flattering or unflattering to.
2. The evidence is a **NULL bundle** — untrained weights, whose decomposition is required to be
   zeros and *is* zeros. The instrument certified itself and then reported the confound.
3. The change is derived from a **property of the deliverable** (`engine.c` runs hard gates),
   not from a preference about the outcome.
4. It cuts **both ways**: it removes a handicap from `G-H1`, and it removes a spurious 62%
   advantage that `G-H1e`'s STATIC arm was enjoying. One gate gets easier to pass, the other
   gets harder.
5. `G-H1`'s **threshold does not move**: `applied-8L` is still `1.096636`, still read back from
   its own file and refused if it has drifted. The bands are untouched.

**What would NOT have been legitimate:** discovering this after H1 returned `1.9` and then
deciding the soft gate was unfair. It is on disk now, with the run that produced it
(`results/h1/h1_eval_NULLCONTROL.json`), before the hours are spent.

## F.5 What changes in the code

* `h1_eval.py` — `G-H1` scored on the hard-gated arm; `trained-8L(soft)` kept and reported;
  `G-H1e` measured hard on both arms.
* `h1_qat.py` — `g_h1e` hard-gates the router arm; the `--every` progress line prints the
  held-out BPB **both ways**, so the watched number is the deployable one and the gap is
  visible while the run is in flight rather than only at the end.
* The null control is re-run against the repaired specification, and `h1_pack.py` will not pack
  until it fires again.

---

# ADDENDUM G — THE ROUTER SMOKE HAD THE SAME CONFOUND. ADDENDUM E's SELECTION IS RETIRED, AND THE RE-RUN COSTS 1.1 HOURS INSTEAD OF 33

**Written and pushed BEFORE the re-run.** Addendum F found that the soft gate is worth
+0.801377 BPB by itself and repaired `G-H1` and `G-H1e`. Checking whether anything else shared
the defect found that the router smoke — all 11.7 hours of it, and the selection addendum E
shipped — had it too.

## G.1 What was wrong

`h1_router_smoke.py` never set `hard_gate`. So per cell:

* the **STATIC** arm ran `set_static(pick)`, which installs gates of **exactly 1** — hard;
* the **trained router** arm ran `m(xev)` at the default, i.e. the **renormalised soft gate**.

Every `ratio` in addendum D's and addendum E's tables is therefore *soft router ÷ hard STATIC*.
It was never a measurement of selection.

**And the distortion is not a constant offset — it interacts with the parameter being chosen.**
The soft gate's damage grows with router peakedness, because `g_e = k·p_e / Σ_sel p_j`
amplifies a dominant group. `aux` exists precisely to keep the router flat. So:

* `aux = 0` cells (collapsed, `occ_max` 0.82–1.00) were **maximally** penalised;
* `aux = 0.01` cells (`occ_max` ≈ 0.13–0.16) were **barely** penalised.

The comparison that decided what ships — balanced against collapsed — is exactly the
comparison the artefact corrupts. This is not a small correction to a verdict; it removes the
basis for the verdict.

## G.2 What is retired, and what survives

**RETIRED — may not be cited:**

* addendum D.1's layer-3 table and its "GO, 4 of 12 settings";
* addendum D.4's **grid narrowing**, which dropped eight settings for losing "by more than
  11%" on distorted numbers;
* addendum E.1's eight-layer table, the shipped `--router-lr 3e-4 --aux 0.01`, the "2 of 8
  layers" shortfall, and the +2.27%/−1.43% error-mass figures;
* addendum E's registered expectation that `G-H1e` fails, **which was derived from those
  numbers**. It is withdrawn rather than carried forward, because a prediction inherited from a
  void instrument is not a prediction.

**SURVIVES:**

* the **planted control** — the router moves on the real donor (grad 1.361e+00 / 3.983e-01).
  It is a wiring check and is gate-independent;
* `init` and `denom`, which are measured at a **zero** router where soft and hard coincide
  exactly, so they were never confounded;
* **the selection RULE of addendum D.3**, which is about eligibility and layers-won and says
  nothing about gates. It is re-applied unchanged to the new table — and re-applying a rule
  registered before either dataset is the strongest form this can take;
* the **structural** reading of §E.4 — the carve's damage concentrates in deep layers (layer 24
  = 41.1% of total, 21+24 = 60.3%). Those are `STATIC` and uncarved-power numbers, both
  hard-gated, both unaffected.

## G.3 Why the re-run is cheap, and why that matters here

The smoke freezes the experts and trains only the router — yet `_quant` re-ran `r3_actsearch`
every step: a 10-point grid over three `[8960, 1536]` tensors, ~3.3 G element-ops per forward.
That was **92% of the file's cost**. With the experts frozen, `ste()` is a constant.

> Measured on the real shape: **4.164 s/step → 0.138 s/step, 30×**, and the cached tensors are
> `torch.equal` to what `_quant` returns. The run **asserts** that equality per layer and
> refuses to start if it does not hold — speed is not traded against correctness on a promise.

Consequence: the **full 12-setting grid on all 8 layers is ~1.1 h**, against 33 h before. So the
re-run does **not** inherit addendum D.4's narrowing. Every setting the first pass discarded on
distorted evidence is measured again, correctly gated. **The narrowing objection disappears
rather than being argued about.**

## G.4 What changes in the runner

1. The trained arm is evaluated with `hard_gate = True`, matching STATIC. **Only the selection
   differs.** Training stays soft — a hard gate has no router gradient.
2. The soft number is kept per cell as a diagnostic, so the gate-form gap is measurable per
   setting instead of being invisible.
3. The quantisation cache, with its bit-exactness assertion.
4. **The trained routers are SAVED** (`*_routers.npz`). The first version discarded them, which
   is why a mis-gated *evaluation* forced a full *retrain*. A measurement that cannot be
   re-scored is a measurement that must be re-run, and that is a defect in the harness, not
   bad luck.

## G.5 The registered prediction for the re-run

Two things, written before the numbers exist:

1. **More settings will beat STATIC than before**, because the handicap is removed from the
   router arm and from nothing else.
2. **`aux = 0` will gain more than `aux = 0.01`**, because it was penalised more. So the
   collapsed settings may now win on ratio where they previously lost — and **rule 1 still
   excludes them on `occ_max`, unchanged, for the reason §D.2 gave: a collapsed router is a
   STATIC router with extra steps.** That rule was registered before any of this and is not
   being touched now.

If instead the new table looks like the old one, then the gate form was not load-bearing in the
smoke and addendum E's selection returns on its own merits — stated here so that outcome is
available rather than embarrassing.
