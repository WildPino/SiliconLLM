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

---

# ADDENDUM H — THE CORRECTED GRID SHIPS THE SAME SETTING. BOTH OF MY REGISTERED PREDICTIONS ARE WRONG, AND A THIRD DEFECT TURNED UP THAT WAS NOT A GATE

The full 12-setting × 8-layer grid re-ran in **4542 s (1.26 h)** with the hard-gated evaluation,
the bit-exactness-asserted quantisation cache, and the trained routers saved. Addendum G said
what to expect. It was wrong twice, and the honest summary is one line:

> **The registered rule, applied verbatim to the corrected data, ships
> `--router-lr 0.0003 --aux 0.01` — the same setting, on the same 2 of 8 layers `[3, 24]`, with
> the same rule-4 shortfall. Not one of the 96 cells changed its beats-STATIC verdict.**

That is the branch §G.5 wrote in as the uncomfortable outcome: *"if the new table looks like the
old one, gate form was not load-bearing in the smoke and addendum E's selection returns on its
own merits."* It does.

## H.1 Prediction 1 — FALSIFIED

Registered: *"more settings will beat STATIC than before, because the handicap is removed from
the router arm and from nothing else."*

Measured: **identical tally.** `lr=3e-4,aux=0` and `lr=3e-4,aux=0.01` win 2 of 8; `lr=1e-3,aux=0`
and `lr=1e-3,aux=0.01` win 1 of 8; the other eight settings win 0 of 8. **Cells whose verdict the
gate repair flipped: 0 of 96.**

## H.2 Prediction 2 — FALSIFIED IN DIRECTION, and the mechanism I gave was wrong

Registered: *"`aux = 0` will gain more than `aux = 0.01`, because it was penalised more"* — the
reasoning being that `aux=0` collapses the router (`occ_max` 0.82–1.00) and a collapsed router
is what the soft gate punishes.

Measured gate-form gap, `(soft − hard)/hard`, mean over the 8 layers:

| lr \ aux | 0 | 0.01 | 0.1 |
|---|---|---|---|
| **0.0003** | −0.25% | −0.25% | −0.40% |
| **0.001** | +0.16% | −0.07% | −0.26% |
| **0.003** | +0.70% | +1.68% | +0.77% |
| **0.01** | **+7.53%** | **+13.49%** | **+18.69%** |

Pooled over lr: `aux=0` **+2.03%**, `aux=0.01` **+3.71%**, `aux=0.1` **+4.70%**. Largest single
cell: layer 24, `lr 0.01 aux 0.1`, hard 0.798996 against soft 1.065250, **+33.3%**.

So the gap is governed by **lr**, not by `aux` — and at fixed lr it **grows** with `aux`, the
opposite of what I registered.

**Why my mechanism was wrong, stated plainly because it is reusable.** The soft gate is
`g_e = k·p_e / Σ_{j∈sel} p_j`. Its distortion is set by the spread of `p` **inside the selected
k**, and by nothing else. `occ_max` measures something different — how many *tokens* share a
group. `aux` flattens the token-level load `f`, and the way a router achieves that is by
discriminating *more* sharply per token, pushing different tokens to different groups. Flatter
occupancy and sharper within-top-k probabilities are the same behaviour seen from two sides.
**I used `occ_max` as a proxy for a quantity it does not proxy.** The right diagnostic for the
soft gate is the within-top-k probability spread, which nothing here logs.

This also scopes addendum F's headline honestly. The **+0.801377 BPB** gate-form cost was
measured with **E37's post-hoc fitted router**, which is sharply peaked. A router trained here at
`lr 3e-4` gives a gap under **0.4%**. Both numbers are right; they are about different routers.
The engine ships the hard gate either way, so hard is the correct measurement regardless of how
big the gap happens to be.

## H.3 The defect the re-run actually caught — and it was not a gate

The original grid ran as **two processes with different `--layers`**: one for layer 3, one for
layers 6–24. `capture_inputs` runs a forward through the model **with the carved FFNs already
installed**, so the activations reaching layer 6 depend on whether layer 3 is carved. The L6-24
process captured with **layer 3 uncarved** — i.e. against a model H1 will never run.

| layer | power old → new | STATIC old → new |
|---|---|---|
| 3 | 0.130669 → 0.130669 **identical** | 0.108585 → 0.108585 **identical** |
| 6 | 0.127010 → 0.117859 | 0.109205 → 0.102206 |
| 9 | 0.104992 → 0.103310 | 0.093466 → 0.090933 |
| 12 | 0.092406 → 0.092600 | 0.077691 → 0.077131 |
| 15 | 0.085702 → 0.084618 | 0.072258 → 0.070932 |
| 18 | 0.165270 → 0.161773 | 0.134604 → 0.130777 |
| 21 | 0.386770 → 0.364360 | 0.287808 → 0.271097 |
| 24 | 0.978680 → 0.884009 | 0.615886 → 0.536754 |

**Layer 3 reproduces to every printed digit across two separate processes**, which is a free
control: the harness is deterministic, so every downstream difference is the carve composing and
not run-to-run noise. Total STATIC error mass **1.499502 → 1.388415, −7.41%**.

§E.4's structural finding survives with its magnitude trimmed: layer 24 alone **41.1% → 38.7%**
of total error, layers 21+24 **60.3% → 58.2%**, and relative damage still falls monotonically
with depth (STATIC/power 0.8310 at L3 → 0.6072 at L24). The reading does not change; two of its
points were borrowed from a model missing a carve.

**The general rule, which is the part worth keeping:** *a grid split across processes is only one
experiment if every process installs the same model.* Here the split silently changed the
measurement target, and it is invisible in the per-cell numbers — only the re-run with all eight
layers installed at once exposed it.

## H.4 Addendum D.4's narrowing is vindicated after the fact

G.2 retired the narrowing because it dropped eight settings on distorted numbers, and that was
the right call on the evidence available. Measured correctly, all eight win **0 of 8**:

`lr 3e-4 aux 0.1` 1.0939 · `lr 1e-3 aux 0.1` 1.1179 · `lr 3e-3 aux 0` 1.1554 ·
`lr 3e-3 aux 0.01` 1.1276 · `lr 3e-3 aux 0.1` 1.1474 · `lr 1e-2 aux 0` 1.1823 ·
`lr 1e-2 aux 0.01` 1.1931 · `lr 1e-2 aux 0.1` 1.1798 (mean ratio vs STATIC).

**It discarded nothing.** Retiring it was still correct — a decision made on void evidence is
void even when it lands on the right answer — but the record should say that the narrowing cost
us no information, only the right to cite it.

## H.5 What ships, and the re-registered expectation

`h1_router_select.json`: `launch = true`, `router_lr = 0.0003`, `aux = 0.01`, layers won
`[3, 24]`, `rule4_shortfall = true`, mean `occ_max` 0.2974 (eligible), mean ratio 1.0399, error
mass **+2.67% worse than STATIC** in aggregate. The only setting better in aggregate is
`lr=3e-4,aux=0` at **−2.02%**, and rule 1 excludes it at mean `occ_max` 0.7720 — collapsed, with
`occ_max` 1.000 on the very layer it wins. The per-layer eligibility sensitivity ships the same
setting, so rule 1's averaging is not load-bearing here.

**Re-registered, on a valid instrument this time:** `G-H1e` is expected to **FAIL**. Addendum G
withdrew addendum E's version of this prediction because it was derived from a void table; this
one is derived from the corrected table, is written before the T4 run, and rests on the same two
facts as before — 2 of 8 layers, and +2.67% aggregate error against STATIC. A PASS would be a
result about *joint* training, not about a hyper-parameter.

`G-H1` — the experts, `trained-8L < applied-8L = 1.096636`, hard gate — is untouched by all of
this and remains the primary gate.

## H.6 What 1.26 hours bought

Stated flatly, because the temptation is to dress it up: **the gate repair changed no decision.**
It bought (a) a selection that rests on a matched instrument instead of a confounded one, which
is worth having before spending GPU hours on it; (b) the split-process defect, which moved the
aggregate baseline 7.4% and would otherwise have gone into the record permanently; (c) 96 saved
routers, so nothing here ever needs retraining to be re-scored; and (d) two falsified predictions,
including a mechanism error — `occ_max` is not a proxy for soft-gate damage — that I would
otherwise still believe.

---

# ADDENDUM I — THE CPU SMOKE OF THE TRAINER KILLED SESSION 2 AT STARTUP, AND IT WAS THE SAME ROOT CAUSE A FOURTH TIME

`h1_qat.py` had never been run end to end on the real bundle path. It has now been, twice: a
fresh session 1 and a `--resume` session 2, eight layers, tiny slices, 2 steps, CPU fp32.

Session 1 was clean. **Session 2 died at startup:**

```
G-H1a  k=E bit-identical to uncarved : FAILS  (max |diff| 3.222e-02)
G-H1a FAILS -- the mask is not wired to what the measurement claims.  STOP.
```

The bundle was not mis-wired. **The check was asking the wrong question**, and it is the same
wrong question as everywhere else in this brief.

## I.1 The root cause, stated once for all four instances

`G-H1a` says: at `k = E` the carved forward is bit-identical to the uncarved one, because the
gates are then exactly 1. That is true under **either** of two conditions and **neither** of them
is unconditional:

* **(a) the HARD gate**, for any router at all — `sel` is all-ones, gates are exactly 1;
* **(b) the SOFT gate, but only at a ZERO router** — `p` is uniform, so
  `g = k·p/Σ_sel p = 1` exactly, for every `k`.

Half of H1's apparatus was written while the router happened to always be zero, so (b) looked
unconditional. It is not. A `--resume` starts from a **trained** router, and (b) then fails **by
construction**.

The same root cause has now produced **four distinct defects** in this brief:

| # | where | what it did | found by |
|---|---|---|---|
| 1 | `G-H1` | scored a soft-gated trained model against a hard-gated control, charging H1 ~0.80 BPB before training counted | null control (addendum F) |
| 2 | `G-H1e` | pitted a soft router against a hard fixed selection; read "FAILS" with zero training | null control (addendum F) |
| 3 | `h1_router_smoke.py` | same, for 11.7 h, on the comparison that chose what to ship | audit after F (addendum G) |
| 4 | `h1_qat.py`'s `G-H1a` | **aborted session 2 at startup, on the command RUN.md prints** | this CPU smoke |

Defect 4 is the expensive one. It fires **after session 1 has already spent its 2.8 GPU-h**, on
a machine the user is paying for, with a message that says the model is broken when it is not.

## I.2 The repair, and its planted control

`h1_qat.py` now checks **both** forms and requires **both** to fire:

```
G-H1a  k=E identical, HARD gate, any router : FIRES  (max |diff| 0.000e+00)
G-H1a  k=E identical, SOFT gate, router=0   : FIRES  (max |diff| 0.000e+00)
carve actually masks at k=16                : FIRES  (max |diff| 2.575e+01)
```

Per the planted-control law, a repair is not trusted until the **wrong** question is shown to
fail on a **known-good** module. `h1_selftest.py` gained **T8**, on a toy whose router is
deliberately non-zero:

```
T8a  the WRONG form (soft gate, trained router) FAILS as it must (max|d| 3.011e+00)  FIRES
T8   HARD gate, any router: bit-identical at k=E (max|d| 0.0e+00)                    FIRES
T8b  SOFT gate at a ZERO router: bit-identical at k=E (max|d| 0.0e+00)               FIRES
```

All of T1–T8 fire.

## I.3 The resume is real, and here is the number that proves it

`T7` proved a resumed module's *forward* is bit-identical to the saved one, on a toy. It did not
prove that a resumed **run** continues rather than silently restarting. The smoke gives that for
free, and it is a clean ordinal control:

| | held-out BPB at step 0 |
|---|---|
| session 1, fresh | **1.442147** |
| session 1, after its 2 steps | **1.431063** |
| session 2, `--resume`, step 0 | **1.431063** |

Session 2 opens exactly where session 1 closed, to all six digits. A silent restart would have
read 1.442147. **This is the check the two-session design actually rests on**, and until now
nothing had run it.

## I.4 Two smaller things the smoke caught

**A label that lied.** The progress line hardcoded `"(fp16, GPU, PROGRESS not the gate)"` and
printed it next to a CPU fp32 number. True on the T4, false on the smoke — the exact defect
class this brief keeps finding, in miniature. It now reports what actually ran, and the record
carries `progress_precision`.

**A warning that does not mean what it looks like.** Torch prints
`None of the inputs have requires_grad=True. Gradients will be None` from
`torch/utils/checkpoint.py`. It comes from the **eval** pass — gradient checkpointing under
`no_grad` — not from training. `model.enable_input_require_grads()` is called for the training
path, and `G-H1c` demonstrates the point rather than asserting it: masters move at **both depth
extremes**, `L03.gate` and `L24.down`, i.e. the backward chain traverses the full carved stack.
The warning is now explained in the CPU-run output so nobody stops a T4 session over it.

## I.5 What this says about the two-session plan

Nothing changes in the plan: `--factors` stays H0's bundle in both sessions, `--resume`
continues the FFN masters and the router, the seeds still differ (1717, 2718), and the shipped
router setting is still `--router-lr 0.0003 --aux 0.01` as addendum H confirmed.

What changes is that **the session-2 command has now actually been executed** rather than
reasoned about. It was wrong, it is fixed, and the fix has a control that fails on purpose.

## I.6 And then the smoke caught a regression I had just introduced

Fixing the lying label in §I.4 defined `prec` inside `main()` and referenced it inside `save()`,
which is a separate function. The next resume run died at the **first periodic checkpoint**:

```
File "h1_qat.py", line 896, in save
    "bpb_fp16_gpu_step0": bpb0, "progress_precision": prec,
NameError: name 'prec' is not defined
```

Worth recording rather than quietly fixing, for three reasons. It was a **fresh** defect,
introduced ~40 minutes earlier while repairing a different one — session 1 had run that exact
code path twice, successfully, before the patch. It sat on the `--every` checkpoint path, so on
the T4 it would have destroyed **the whole session's output** at the first 250-step save while
the run appeared healthy up to that moment. And it was invisible to `py_compile`, to
`ast.parse`, and to the self-test, all of which passed.

`prec` is now an explicit parameter of `save()`, and the function was scanned for any other free
name that could do the same thing (**none**). The general point is not about this variable:
**a patch that adds a field to a record written by another function is a cross-function change,
and "it compiles" is not evidence about it.** The only thing that caught it was running the
program.

## I.7 A gate value nobody could reproduce

Running the same bundle three times printed `carve actually masks at k=16` with
`max |diff|` **2.082e+01**, **2.575e+01**, **2.231e+01**. The verdict is robust — the point is
only that the carve *does* mask — but that number is **stored in the run record** as
`carve_is_live.max_abs_diff`, and it came from an unseeded `torch.randn` probe. A gate value
that changes run to run is not evidence, even when the verdict it carries is. The probe is now
seeded (90210); the verdicts do not depend on it.

## I.8 Final state of the trainer, verified by running it

| | session 1 (fresh) | session 2 (`--resume`) |
|---|---|---|
| `G-H1a` hard gate, any router | FIRES 0.0e+00 | FIRES 0.0e+00 |
| `G-H1a` soft gate, router = 0 | FIRES 0.0e+00 | FIRES 0.0e+00 |
| `carve_is_live` | FIRES | FIRES |
| `G-H1b` first applied update | step 1, 0 declined | step 1, 0 declined |
| `G-H1c` masters moved | L03.gate, L24.down, L03.router | same |
| step-0 held-out BPB | 1.442147 | **1.431063** = session 1's last |
| `resumed_ffn_from` in the record | — | `h1_smoke_s1.npz` |
| periodic save + final save | both | both |
| `complete` | true | true |

`G-H1e` reads FAILS in both (4.195582 / 4.192291 against STATIC 4.115758 / 4.116145) — expected
and meaningless at 2 steps on 64 sequences, and already re-registered as the expectation for the
real run in §H.5. The gate that decides H1 is `G-H1`, scored afterwards by `h1_eval.py` on CPU
fp32 against `applied-8L = 1.096636`.


---

# ADDENDUM J — BOTH SESSIONS HIT THE TIME CAP AT STEP 195, AND THE FOUR BANDS ARE NOT SYMMETRIC UNDER A TRUNCATED BUDGET

**2026-09-14. Written after the operator handed over the two session artefacts and BEFORE
`h1_eval.py` has been run on either of them.** Not one CPU fp32 number exists yet. The reading
rule below is therefore registered against an unknown outcome, which is the only condition under
which registering a reading rule means anything.

The operator reported the two sessions raw and explicitly declined to read them against the
bands: *"Te lo riporto così com'è, senza interpretarlo contro le bande ... quella lettura è
tua."* That is `feedback_no_anchoring_producer` honoured from the other side, and it is why the
numbers below can be used at all.

## J.1 What actually came back

| | session 1 | session 2 |
|---|---|---|
| seed | 1717 | 2718 |
| `resumed_ffn_from` | `null` (fresh carve) | `h1_trained_s1.npz` |
| wall | 10101.43 s | 10088.85 s |
| `steps_requested` | 4000 | 4000 |
| **steps completed** | **not recorded** (see J.4) | **195** — `TIME CAP 2.8 h reached at step 195 -- stopping cleanly` |
| `--every` | 250 | 250 |
| history entries | step 0 only | step 0 only |
| step-0 held-out BPB (fp16, GPU) | 1.2052601751146157 | 0.9795840560215258 |
| `nonfinite_microbatches` | 0 | 0 |

**The history is empty past step 0 because the logger interval is 250 and the cap landed at
195.** That is the whole explanation, and it is a logging artefact, not a training failure. The
trainers ran: `G-H1b` reads the first APPLIED optimizer update at step 1 with 0 declines in both
sessions, `G-H1c` shows the masters moved, and the saved bundles are the trained weights.

So the registered budget bought **≈390 cumulative optimizer steps against 4000 — 9.75%.**

## J.2 Every planted control fired, in both sessions

| control | s1 | s2 |
|---|---|---|
| `G-H1a` k=E identical, HARD gate, any router | FIRES, max\|d\| **0.0** | FIRES, max\|d\| **0.0** |
| `G-H1a` k=E identical, SOFT gate, router=0 | FIRES, max\|d\| **0.0** | FIRES, max\|d\| **0.0** |
| `carve actually masks at k=16` (discrimination) | FIRES, max\|d\| 23.125 | FIRES, max\|d\| 27.219 |
| `G-H1b` first APPLIED update | step 1, 0 declined | step 1, 0 declined |
| `G-H1c` masters moved | FIRES | FIRES |
| `G-H1e` router vs STATIC, both HARD-gated | 2.882662 < 3.082641 → FIRES | 2.821667 < 3.076994 → FIRES |

`G-H1a` at 0.0 **with** `carve_is_live` at 23.1/27.2 is the pair that matters: the mask is
provably inert at `k = E` and provably active at `k = 16`, so it is wired correctly and the
instrument discriminates. Per `feedback_planted_controls` the apparatus has earned the right to
have its nulls counted.

**The shortfall is budget, not correctness.** Nothing in these artefacts is broken.

## J.3 The fp16 step-0 pair IS paired — and still may not be read against the bands

My first instinct was to refuse the 1.2053 → 0.9796 comparison as confounded by the seed change.
That is wrong, and the code says so: `heldout_nats` is a deterministic full sweep over the whole
held-out array at `bs=1` with no sampling, so **the seed does not enter the evaluation.** Session
2's step 0 is session 1's end state measured on the identical stream at identical precision. It
is a legitimate paired before/after of session 1's 195 steps: **−0.2257 BPB.**

**It still may not be compared to a band edge, for a different and decisive reason.** The band
edges — `applied-8L` 1.096636, `ternary-8L` 0.947851, `h0-run3` 0.810022 — are **CPU fp32 on the
frozen 24×512 slice**. The 1.2053/0.9796 pair is **fp16 on GPU over `h1_heldout.npz`**. Different
corpus, different precision, different device. Noting that 0.9796 "falls between `ternary-8L` and
`applied-8L`" would be a scope error of exactly the kind `feedback_charged_vs_moved_bytes` and
E58's scope rule exist to stop: *a bound is a number with a scope*, and these two numbers do not
share one. §5.3 registered this metric as PROGRESS precisely so it could not be spent as a gate.

**What the pair does license:** the direction of travel is down, and the trainer moved the model
a long way in 195 steps. That is a statement about conduct, not about H1's verdict.

## J.4 A defect: the artefact records what was ASKED, not what was DONE

`h1_trained_s*.json` carries `steps_requested: 4000` and no `steps_completed`. The only place
the real count exists is one stdout line, and **session 1's log is 0 bytes** — lost to the
cp1252/tqdm encoding crash the operator diagnosed. So session 1's completed step count is
**unrecoverable from the artefacts**; the ≈195 above is inferred from its wall time matching
session 2's to within 13 s on the same shape and the same cap, and it is marked as an inference
wherever it is used.

This is `feedback_measure_the_run_not_just_the_cell` again: *a number that describes the CONDUCT
and not the result escapes every gate.* A trainer that cannot say how far it got is not fully
auditable. **Fix, for any future session:** `steps_completed`, `stop_reason` and `seconds_per_step`
go in the JSON, which survives the log.

## J.5 THE READING RULE, registered here

The four bands were written for a run that completed its registered budget. At 9.75% of it they
are **not symmetric**, and the asymmetry is not a matter of taste:

* `TRAINING-HELPS`, `CARVE-IS-TRAINABLE` and `CARVE-IS-FREE` are **achievement** bands. They say
  *training reached this level*. Reaching one of them on 9.75% of the budget is the claim made
  **a fortiori** — a shorter run that gets there is stronger evidence, not weaker. These are read
  exactly as registered.
* `CARVE-NOT-TRAINABLE` is **not** an achievement band. It says *training buys nothing and the T4
  branch closes.* That is a claim about the **limit** of training, and 390 of 4000 steps cannot
  support it. Reading it here would be the `feedback_gate_is_not_a_progress_meter` failure in its
  floor form: treating a counter that never got off the floor as though it had measured a ceiling.

**Registered rule:** if `h1_eval.py` returns trained BPB **≥ `applied-8L` (1.096636)**, the
outcome is recorded as **`H1-UNDERTRAINED`** — an OWED result requiring the remaining budget —
and **not** as `CARVE-NOT-TRAINABLE`. The negative band stays unassigned until a run completes a
materially larger fraction of its registered steps.

**Why this is not a goalpost move.** It is registered before any CPU fp32 number exists; it can
only ever **weaken** a conclusion I am allowed to draw, never strengthen one; and it cannot
manufacture a positive, because the three positive bands are untouched and `G-H1`'s gate — trained
BPB < `applied-8L` — is untouched. The direction of a legitimate mid-flight change is the one
that costs the author something (addendum C.3 set that precedent by making the good-news band
twelve times narrower).

## J.6 What runs now, and what does not

1. `h1_eval.py` on **`h1_trained_s2.npz`** — the cumulative artefact, ≈390 steps, both sessions'
   training. CPU fp32, frozen slice. **This is the gate.**
2. `h1_trained_s1.npz` is retained as the mid-point but is **not** a second measurement of the
   same thing and will not be reported as one.
3. **The weights do not enter git.** 1.33 GB each; they stay in one place on disk per the
   standing rule that weights live in exactly one location. The JSON sidecars and the surviving
   log go into `results/h1/`.
4. `G-E63d` is still `VOID` and OWED. H1 does not touch it, and neither does E64.


---

# ADDENDUM K — ADDENDUM J PICKED THE WRONG DENOMINATOR. THE RULE SURVIVES; ITS GROUNDING DOES NOT

**Still before `h1_eval.py` returns.** Written minutes after J, after opening §2 and §7 — which
J should have opened before it chose a denominator. No CPU fp32 number exists yet.

## K.1 The error

J.1 and J.5 said the run bought **"9.75% of budget"**: ≈390 cumulative steps against
`steps_requested: 4000`. **`4000` was never the registered budget.** §2 costs H1 in *memory* and
§7 costs it in *sessions*:

> *"Cost: two T4 sessions of 2.8 h (~5.6 GPU-h). Both are needed and §7 says why."*
> *"Registered: session 1 produces no verdict... The gate in §5.2 is evaluated **after session 2**."*

`--steps 4000` was an upper bound the `--max-hours 2.8` cap was always going to truncate.
**Both registered sessions ran, both to the cap. The registered budget was spent IN FULL.** By
the registered design this is the moment the gate is read, not a short delivery.

This is `feedback_gate_is_not_a_progress_meter` in its denominator form for the **fourth** time,
and the second time in two days: I divided by a quantity that was never the budget, and it made
the run look like a 90% shortfall when it was a completed one. The tell was available for free
in the brief's own header line.

## K.2 The grounding that actually holds, and it is stronger

§7 does not justify two sessions by hours. It justifies them by a **step count**, taken from H0:

> *"After 500 steps H0 looked flat on the metric being watched. It was not: the second 500 steps
> removed a quarter of the damage that survived the first 500... A one-session H1 would be read
> the way a one-session H0 was read, and that reading was wrong."*

Measured, from the artefacts rather than assumed:

| | steps in one 2.8 h session | s/step | cumulative over two sessions |
|---|---|---|---|
| **H0 run 3** (`h0_trained3.json`, 10094 s) | **≥ 500** (history logs step 500; `--every` 250 so the true figure is 500–749) | **≤ 20.2** | ~1000, and §7 says only the second half was informative |
| **H1** (`h1_trained_s2`, 10089 s) | **195** (`TIME CAP 2.8 h reached at step 195`) | **51.7** | **≈390** |

**H1's two full sessions delivered fewer steps than H0's ONE** — 390 against ≥500. The run sits
*below* the mark §7 explicitly names as the one that was read wrongly. The two-session design was
calibrated in sessions while its rationale lived in steps, and H1 costs **≥2.6× per step** than
the run the rationale was borrowed from, because it trains 330,301,440 FFN masters plus a router
where H0 trained q/o factors.

**Nobody mis-ran anything.** The sessions did what was asked; the asking was mis-costed, by me,
in the brief.

## K.3 What changes, and what does not

**J.5's reading rule STANDS, unchanged**, and is now grounded in §7 instead of in a denominator
I invented:

* the three achievement bands — `TRAINING-HELPS`, `CARVE-IS-TRAINABLE`, `CARVE-IS-FREE` — are
  read exactly as registered, a fortiori;
* **`CARVE-NOT-TRAINABLE` may not be assigned**, because it is a claim about the *limit* of
  training and this run is below the step count §7 registered as insufficient to read. That
  outcome is recorded as **`H1-UNDERTRAINED`** and stays OWED.

`G-H1`'s gate — trained BPB < `applied-8L` — is untouched, as are §5.2, the band edges, and §8's
predictions.

**What does change is the character of the T4 ask.** Under J it read as *"finish the budget you
already granted"*. It is not that. The granted budget is spent. **§6 of this addendum is a NEW
request**, and it must be put to the user as one.

## K.4 The ask, sized on measured throughput

To reach parity with the H0 run whose step count §7 borrows — 1000 cumulative steps:

```
  needed        1000 - 390  =  610 steps
  measured      51.7 s/step  (10089 s / 195 steps, session 2)
  time          610 x 51.7   =  31,540 s  =  8.8 GPU-h
  sessions      8.8 / 2.8    =  3.1       ->  FOUR sessions of 2.8 h (11.2 GPU-h),
                                               which lands at ~1170 cumulative steps
```

Against the standing 30 GPU-h/week/account across three accounts (90 GPU-h/week), **11.2 GPU-h is
one week's work on a single account** and the relay is already verified unattended. This is the
clause in the standing goal that covers it — *"se ti serve possiamo fare sessioni brevi
(settimane) su T4 per rifiniture, in quel caso però melo devi comunicare"* — and it is being
communicated rather than assumed. It goes in `COMMUNICATION.md`.

**It is not requested yet, and nothing is launched.** `h1_eval.py` is running on the ≈390-step
bundle first, because if it already lands in one of the three achievement bands the extra hours
buy a sharper number rather than the verdict, and that is a different and much weaker case for
spending them.

## K.5 The defect this leaves in the trainer

Three fields would have prevented both J's error and the unrecoverable session-1 count, and they
cost nothing:

```
  steps_completed     the number the gate's readability depends on
  stop_reason         "time-cap" | "steps" | "nonfinite"
  seconds_per_step    what any future budget must be costed on
```

They belong in the JSON, which survives when the log does not. Registered as owed before the
next session is asked for.

---

# ADDENDUM L — THE RESULT: `G-H1` PASSES, AND K.4's FOUR-SESSION ASK IS CUT TO ONE

**2026-09-14, after both bundles were scored on CPU fp32.** Full reading:
`probes/H1_THE_CARVE_TRAINED.md`; ledger §64. This addendum records only the verdict, the
prediction scoring that belongs to the brief, and the correction to K.4.

## L.1 The verdict

```
trained-8L  0.962593  <  applied-8L  1.096636      delta -0.134043   -> TRAINING-HELPS
```

`G-H1` **PASSES** at both checkpoints (s1 ~195 steps: 0.983337; s2 ~390: 0.962593). Band
`TRAINING-HELPS`, which J.5/K.3 registered as an achievement band readable *a fortiori*. Every
planted control fired on the gate instrument as well as on the T4 side.

## L.2 Predictions, scored

| # | outcome |
|---|---|
| 1 `applied-8L` 3.10–3.55 | **MISSED** (1.096636) — already scored in addendum C |
| 2 `G-H1` fires | **TAKEN**, both checkpoints |
| 3 lands in `TRAINING-HELPS` | **TAKEN post-repair, UNSCOREABLE pre-repair** — §5.2's original table made this band EMPTY (C.3); a prediction naming an empty band is not a clean hit |
| 4 free-running ≤ 40/160 | **NOT MEASURED** — `h1_eval.py` has no generation arm. **OWED.** |
| 5 router contribution **not separable** | **second clause FALSIFIED** — `arm-E` prices it directly, additively, residual 0.0 |
| 6 8 of 28 layers move whole-model BPB at all | **TAKEN** (−0.134043) |
| **E.5** `G-H1e` **FAILS** | **FALSIFIED**, favourable direction, at BOTH checkpoints (margin 0.1996 → 0.2551) |

**Two of seven were falsified in the FAVOURABLE direction.** This brief's errors ran
pessimistic — the comfortable direction to be wrong in, and therefore the one to watch.

## L.3 K.4's ask is CUT from four sessions to ONE, and the curve is why

K.4 sized **four** sessions against H0's 1000-step mark. That was computed **before any CPU
fp32 number existed**, i.e. against an unread curve. With the curve in hand it is the wrong ask.

| | steps | `trained-8L` | bought |
|---|---|---|---|
| `applied-8L` | 0 | 1.096636 | — |
| s1 | ~195 | 0.983337 | −0.113299 |
| s2 | ~390 | 0.962593 | **−0.020744** |

**No curve is fitted to two points**, and the two increments are **not comparable**: both
sessions carry `adam_state_restarted: true`, so session 2 begins with the optimizer state
discarded and session 1 does not. Restart transient and genuine flattening cannot be separated
from what exists.

**One session is decisive either way**, which four were not:

1. it yields the **second post-restart increment**, directly comparable to −0.020744 — two
   comparable increments are the minimum that can say whether the curve is decaying;
2. the gap to `CARVE-IS-TRAINABLE` is **0.014742**, *smaller than one session-2 increment*, so a
   non-decaying curve **crosses the band**;
3. the **router** — `+0.001884 → −0.005871`, the only term that accelerated between the two
   checkpoints, and the one carrying the MoE question — gets a third point.

**Registered: ONE session of 2.8 h.** Asking for four now would be buying hours against a trend
nobody has read. If the third point shows the curve still descending, the case for more is made
*on evidence* and asked for then.

## L.4 Still owed, unchanged by this result

* **Free-running** (prediction 4) — no generation arm exists in `h1_eval.py`.
* **`steps_completed` / `stop_reason` / `seconds_per_step`** in the trainer JSON (K.5).
* **The `k=E` trained arm** that would separate ternarisation from carving — §9 registered it as
  not fitting the budget, and it still does not. Until it exists, H1's result prices the
  **smaller** of the two damage axes: §62.12 puts the half-byte format at **82%** of the
  dense→chance damage and the carve at **17%**.
* **`G-E63d` is `VOID` and OWED.** H1 does not touch it.

---

# ADDENDUM M — session 3, and the 2.8 h cap was MINE, not the platform's

**Pre-registered 2026-09-14, pushed before the bundle is rebuilt and before session 3 runs.**
**The T4 budget was granted this day: 90 GPU-h/week across three accounts, T4x2.**

## M.1 The thing I got wrong, and it is not the one addendum K found

K re-grounded the undertraining argument on §7's step count after J divided by a budget that
was never registered. Both addenda took **2.8 h** as the unit of a session and argued about
**how many** of them to ask for — K said four, L said one.

**Neither asked why a session is 2.8 hours.** It is not a platform limit:

```
scripts/kaggle_run.py:21   "The batch kernel runs to the 12 h session limit and stops."
scripts/kaggle_ops.py:14   "...before a 12 h job is staked on it."
h1_qat.py:611              ap.add_argument("--max-hours", type=float, default=2.8)
```

**`2.8` is a default I chose.** The platform allows **12 h**. So the entire "how many sessions"
argument was conducted in a unit that was four times smaller than it needed to be, and H1's
undertraining — the finding of addenda J, K and L — is substantially **an artefact of my own
flag**, not of the budget.

This is [[feedback-dont-size-against-an-unread-number]] in a third form, and the worst of the
three: J divided by an unregistered budget, K sized against a measurement still running, and
**M finds that the UNIT both of them were counting in had never been checked against the
platform that defines it.** The error again has no sign — here it made me ask for too little,
twice.

## M.2 What one long session buys, arithmetically

From the artefact: `10088.85 s / 195 steps` = **51.7 s/step** (`h1_trained_s2.json`, and
addendum K.3's table).

| plan | steps | Adam restarts | vs H0's ≥500 per session |
|---|---|---|---|
| what H1 actually got (2 x 2.8 h) | **~390 total** | **2** | below, in EVERY session |
| L's ask (1 more x 2.8 h) | ~585 cumulative | 3 | still below, per session |
| K's ask (4 more x 2.8 h) | ~1,170 cumulative | 6 | still below, per session |
| **M: 1 x 11 h** | **~766 in ONE session** | **1** | **above, in a single session** |

**One 11 h session beats K's four-session ask on the axis that matters** (steps under a
continuous optimizer) while costing **11 GPU-h instead of 11.2**, and it removes the defect
that made L's reading provisional: both existing increments carry `adam_state_restarted: true`,
so they are **not comparable to each other** and nothing may be extrapolated from the two.
A single uninterrupted session has no such seam.

## M.3 Why 11 h and not 12

**The trainer's own cap must fire BEFORE the platform's, or the run is lost.** `h1_qat.py:861`
breaks on `--max-hours` and only then reaches the final `np.savez` of a 1.33 GB artefact
(line 892). If the Kaggle session is killed at 12 h first, that save never happens.
**`--max-hours 11.0` leaves an hour of margin for the final eval, the save and the upload.**

Partial loss is bounded independently: `save()` is already called at every `--every` (250)
steps with `done=False`, so a hard kill costs at most ~250 steps. **That path has never
executed** — sessions 1 and 2 both stopped at step 195, before the first periodic save — so
session 3 is also the first exercise of the checkpoint path.

## M.4 Apparatus changes, and they are the OWED ones

`h1_qat.py` records three fields it never recorded, and the reason is J's error:

```
"steps_completed":  step        # what the run DID
"stop_reason":      "time-cap" | "steps-exhausted" | "in-progress"
"seconds_per_step": el/step
```

**And a correction of meaning, not just an addition.** `complete: true` is written by the final
save **whether or not the time cap truncated the run**. Addendum J read `complete: true`
alongside `steps_requested: 4000` and concluded a 90% shortfall from a fully-spent session.
The flag means *the script finished cleanly*; `stop_reason` is now the field that says whether
the **training** did. Self-tests re-run and all fire.

Nothing else changes: same layers `[3,6,9,12,15,18,21,24]`, `k=16` of `E=256`, `bs 2`,
`accum 8`, `lr 2e-4`, `router-lr 3e-4`, `aux 0.01`, `--every 250`, resume from
`h1_trained_s2.npz`, seed **3141** (new, and the seed does not enter the eval — `heldout_nats`
is a deterministic full sweep, addendum J.6).

## M.5 Predictions — registered, falsifiable

1. **`steps_completed` ≥ 700** and `stop_reason == "time-cap"`. *If s/step degrades badly on a
   long run — thermal, or the periodic save — this is where it shows.*
2. **The gate crosses into `CARVE-IS-TRAINABLE`: trained BPB < 0.947851.** The distance from
   s2 is **0.014742**, which is **less than one 195-step increment** (session 2 moved
   −0.020744), and session 3 is ~3.9x longer with no Adam seam. **This is the prediction that
   can fail**, and the reason it might: the experts are flattening (−0.115183 → −0.128172,
   i.e. the second session added only −0.012989 on that term).
3. **The router term keeps accelerating relative to the experts** (§64.2's law): the router's
   share of the session's movement exceeds session 2's **37%**. *Falsifies §64.2 if it does not.*
4. **`G-H1e` fires again and its margin grows** beyond session 2's 0.255112.
5. **The first periodic checkpoint (step 250) is written and is loadable.** Never exercised
   before; a silent failure here would cost the session.

## M.6 What session 3 may NOT conclude

1. **Nothing about rate.** No tok/s. **`G-E63d` stays `VOID` and OWED.**
2. **Nothing about the 10 B, and nothing about the RANK axis.** H1 trains a **carve** on a
   **ternary** FFN — §62.12 prices that as the **smaller** of the two damage axes, and E65
   sized the rank hole at **5.18x** the one H1 works against. A win here does not transfer.
3. **No promotion of the two-point curve.** With one continuous session the three points still
   differ in optimizer continuity, so the curve is read ordinally, not fitted.
4. **Nothing that re-reads sessions 1 and 2.** Their numbers stand as published in addendum L.

