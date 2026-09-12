# H0 — does straight-through training move the ternary factored form at all?

**This is the branch E37 leaves standing, and it is the only post-hoc route this programme has
never tested.** Every other one has closed: precision (E18), sparsity (E19), rule (E20), rank
(E21), composition-into-the-format (E22), and — as of today — **carve-to-the-speed-rate (E37)**.
H0 asks the opposite question: not *can a trained dense model be converted*, but **can the format
be trained into.**

**Plan**: `decisions/T4_HEALING_PROPOSAL.md` §3 (H0), gate registered there long before the run:
*"`QO512-TB`'s teacher-forced rises from `28/160` to `≥ 48`. Below that, stop."*

---

**Verdict: `GATE PASSES`, and by a margin the bar was not designed to measure.**
**Teacher-forced `28 → 111` of 160.** The bar was 48. **BPB `2.812226 → 0.825358`** — the intact
donor reads `0.767595`, so **97.2% of the damage is gone.**

**And it did that on 12.5% of the training it asked for.** The T4 job stopped itself at its 2.8 h
wall after **500 of 4000 steps**, ~4.1 M of 16 M tokens. The checkpoint is usable by design; the
gate does not need the rest.

**The caveat, stated here and not in a footnote: free-running generation did NOT recover.**
`8/160` against the intact donor's `160/160`, and *below* E22's fp32 `QO512+V52` at `15/160`.
**Teacher-forced and BPB came back; the generator did not.** That is the same split this
programme has hit at every scale, and H0 does not resolve it.

---

## 1. What was trained, and what was not

`QO512-TB`: `q_proj` and `o_proj` of all 28 layers replaced by **ternary low-rank factors**
`ternarize(A)·diag(s)·ternarize(B)` at rank 512, straight-through. **88,109,056 trainable
parameters.** Everything else — `k/v`, the FFN, the head — stays fp32 dense.

**No FFN carve and no router at all.** That was a design choice made when H0 was built, so that
what the gate measures is one thing. It also means H0's object is **not** the shippable one:
`STACK` (ternary factors + `V52` + ternary FFN + ternary head) reads `4/160` post-hoc, and
nothing here says training closes that.

## 2. The instrument, and its two planted anchors

Measured on **this CPU, in fp32**, with the same instrument every published number in this
programme used — E12's chance line, E17/E18's free-running bands, E20 part B's teacher-forced
top-1, the frozen 24×512 heldout slice (`ids_sha a1a48dc9…`) and the five frozen E6 prompts.
Porting it to a GPU would make H0's number incomparable with E19–E23, which is exactly why the
gate is a number from this table and not a loss curve.

| anchor | what it must read | measured |
|---|---|---|
| **`init`** — the factored model **before** training | E22's published `QO512-TB`: BPB `2.812226`, free `1`, tf `28` | **`2.8122382`, free `1`, tf `28`** |
| **`intact`** — the donor untouched | E22's `base`: BPB `0.767595`, free `160`, tf `160` | **`0.7675949641`** (diff **−3.6e-08**), free **160**, tf **160**, mean rank **1.00** |

**Both anchors fire, and they bracket the result from below and above.** The low one reproduces
E22's `QO512-TB` to six decimals; the high one reproduces E22's `base` to **3.6e-08** with
perfect 160/160 on both metrics and mean rank exactly 1.00. That is the planted control the law
requires — an instrument that reads 160/160 on a known-good model and 28 on a known-broken one
has earned the right to report 111 on this one.

**The training stream is not the eval slice, and that is recorded in the artifact rather than
asserted**: `h0_train.json` carries `part: "calib"`, `seed 90011`, its own `ids_sha256`
(`9bb5229f…`), a different `corpus_sha256`, and a field literally named
`eval_slice_sha256_NOT_THIS`. The calibration seed E23 pins (`42424`) was **deliberately
avoided**.

## 3. The numbers

| | BPB | vs intact | free | teacher-forced | mean rank | rank ≤ 5 |
|---|---|---|---|---|---|---|
| `base` (intact, E22) | **0.767595** | — | 160 | **160** | 1.00 | 160 |
| `QO512-TB` **init** | 2.812226 | +2.044631 | 1 | **28** | 1476.4 | 68 |
| **`QO512-TB` trained** | **0.825358** | **+0.057763** | **8** | **111** | **7.10** | **144** |
| *(reference)* E22 `QO512+V52`, **fp32 factors** | 1.005039 | +0.237444 | 15 | 126 | 2.03 | 148 |
| *(reference)* E21 fp32 ceiling for this rank | — | — | — | 144 | — | — |

**Read the third row against the fourth.** The trained **ternary** factored form reads a *better*
BPB than E22's **fp32** factored configuration — `0.825358` against `1.005039` — and 111
teacher-forced against 126. Those are different objects (`QO512+V52` also carries the FFN carve),
so this is a comparison of *configurations*, not a controlled ablation. What it licenses is the
narrow claim H0 was built to test: **straight-through training moves this object, and it moves it
most of the way back.**

**Mean rank `1476 → 7.10`** and **rank≤5 `68 → 144` of 160** are the rank partner E14 §3 demands
for the BPB score, and they move together.

## 4. What the trainer's own gates say

`G-H0e`, the planted control for the trainer itself — *"a null from H0 only means something if
the instrument can be shown to move the masters"* — fired on the real run:

* **first update APPLIED at step 5**, after **4 declined** by `GradScaler` (scale `65536 → 4096`,
  exactly the four halvings a cold fp16 scaler takes);
* masters moved **8.2e-06** on all three probes — real, not zero;
* **`nonfinite_microbatches: 0`** across the whole run.

**This is the run-1 defect, fixed and verified.** Run 1 died at step 1 because the gate read the
masters without checking that an update had been *applied* — and `GradScaler` declines the first
step by construction. The repaired gate waited for the real update instead of firing on an
unverified precondition. The second run-1 defect (the quantizer cache, filled by step 0's
`no_grad` probe, pulling `A` and `B` out of the graph and leaving only `s` — 512 numbers of
88.1 M — trainable) **would have manufactured H0's null**; both fixes went in at `c613e5f` with
two new planted controls.

## 5. What this does NOT say

* **Nothing about the shippable object.** `QO512-TB` is ternary `q/o` only. The FFN, the head and
  the carve are all still fp32 or absent here. `STACK` is the thing that ships and it is untested
  under training.
* **Nothing about generation.** Free-running is `8/160`, in the `AT-FLOOR` band. The model scores
  well and does not generate. **BPB and teacher-forced are not a generator**, and E22 §, E17 and
  E18 all said so before this run.
* **Nothing about 10 B.** This is 1.5 B, and §0 of the proposal says it in its own words: what
  the T4 hours buy is a **structure validation**, not the goal.
* **Nothing about the full schedule.** 500 of 4000 steps. Where the curve goes with the other 3500
  is unmeasured; the last two checkpoints read tf 110 → 112 on the GPU's own fp16 probe, which is
  flat, and flat at step 500 is not evidence about step 4000 either way.

## 6. Where the weights are

**These cannot be regenerated** — the T4 instance that produced them is gone, and unlike the
init bundle they are not a function of pinned inputs. So there are now **two copies**, verified
identical by hash:

```
benchmarks/donor_adaptation/s1/results/h0_kaggle_run2/h0_trained.npz   (working copy)
D:\_ktmp\h0_run2_weights\h0_trained.npz                                (backup, + the .json)
352,967,686 bytes   sha256 6d3fd3e3d0a15f5c60cd63a38563f4a4994c35e28005290de06ecb6374b54764
280 keys = 28 layers x 2 organs x 5 tensors (A, s, B, rms_in, rms_A), 0 non-finite of 280
```

Gitignored by size, anchored to the place and not to a type (per the `.gitignore` audit), and
named here and in `COMMUNICATION.md` so the pointer survives this session.

## 7. Owed

1. **The other 3500 steps**, which is the cheap question now that the gate is passed.
2. **`STACK` under training** — the object that actually ships. H0 deliberately did not test it.
3. **Free-running recovery**, which is the one axis that did not move and the one the goal's
   sentence is actually about.
4. **The carve, trained rather than applied** — E37 measured that applying it costs +0.55 BPB
   with the router provably irrelevant. H0 says the format can be trained into. Nobody has put
   those two facts in the same run.
