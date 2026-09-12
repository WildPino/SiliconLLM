# BRIEF E41 — is the PARTITION the lever? Can a better grouping make ~6% activation usable?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

**QUALITY ONLY, on a REALLY TRAINED donor, in fp32.** No speed, no synthetic weights. This is the
half of the goal that E37–E40 kept handing to a GPU.

---

## 0. Why this, and why it is worth a CPU afternoon before any GPU hour

E40 closed the shape axis: with every real attention lever pulled, **50 tok/s buys ~6% FFN
activation at ten billion parameters** and 110 tok/s buys 1.17%. E38 measured what a really
trained donor does at those rates — **3.597 BPB at 6.25%**, against a dense **0.767595** and a
chance line of **4.069819** — and concluded no *router* can fix it, because even a per-token
oracle cannot.

**But E38 also found something it could not follow up, and left it owed in its §7 item 3.** The
same static selector, on the same donor at the same `k`, reads **2.2119 on the D0c labels and
4.5833 on a random equal partition** — **a 2.37 BPB swing from the grouping alone, which is four
times the entire window E37's whole sweep lived in.**

> **Every carve in this programme — E19, E26, E36, E37, E38, E39, E40 — has used ONE family of
> labels, and nobody has ever asked whether it is any good.**

E38 proved no router beats the oracle *of a given partition*. **It says nothing about a different
partition, whose oracle is a different function.** If the grouping is the lever, ~6% becomes
usable, the goal is reachable without training, and the T4 ask in `COMMUNICATION.md` should be
withdrawn. If it is not, the ask is justified by one more independent measurement instead of by
argument.

**This costs a CPU afternoon and no GPU.** That is the whole reason it goes first.

## 1. The question

> **Hold the donor, the rate and the selector recipe fixed. Change only HOW NEURONS ARE GROUPED
> into the 256 carve groups, and ask whether any grouping makes the rate the box affords
> produce a model.**

## 2. Setup, inherited unchanged from E38 so the numbers compose

Qwen2.5-1.5B (`S15`), **fp32**, `D=1536, F=8960, L=28`, `E=256` groups of **35** neurons.
Frozen heldout slice `("heldout", 24, 512, 1234)`, `ids_sha`
`a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`, 51,870 scored bytes.
Dense anchor **0.767595**, chance **4.069819**. Calibration on a **separate** slice
(`"calib"`, seed 424242) — never the eval one.

**Nothing is reimplemented.** The masking hook is E38's `Carve`; the ridge router is
`e23_router.fit_routers`, **refit per partition** with that partition's own label map, which is
the same code E23 and E37 used.

## 3. The partitions

| arm | how the 8,960 neurons are grouped | why it is here |
|---|---|---|
| **`D0C`** | E37's label family | **the control** — what every carve in this programme has used |
| `RAND` | random equal partition, seed 38038 | E38's grouping B, so one cell reproduces |
| **`CONC`** | rank by calibration mass, group 0 = top 35, group 1 = next 35, … | **maximum concentration**: if you may keep only `k` groups, put the heavy neurons together |
| `STRIPE` | rank by calibration mass, deal round-robin | **maximum balance** — every group carries near-identical mass |
| `COACT` | cluster by co-activation correlation on the calib slice | neurons that fire *together* land together, which is the structure a router can predict |
| `PERM` | `D0C` with the 256 group IDs permuted, seed 41041 | **the planted control** — see `G-E41C` |

Selectors: **`oracle`** (per-token, the ceiling of every router on that partition) and
**`fitted`** (the attainable ridge router, refit for each partition).

`k ∈ {16, 3}` — **6.25%, what E40 measured 50 tok/s affording at ten billion**, and 1.17%, what
~110 tok/s affords. Plus `k = 256` for the inertness gate.

## 4. The gates

**`G-E41A` — inertness.** At `k = E` every partition must be bit-inert: BPB equal to dense within
`1e-6`. A partition that changes the answer while dropping nothing is not a partition, it is a
bug. (E38's `G-E38A`, re-run per partition.)

**`G-E41B` — the anchor.** Unmasked fp32 must read E22's `base` `0.767595 ± 0.001`, or the
session is not measuring the published object and stops.

**`G-E41C` — the planted control, and it must be BIT-EXACT.** `PERM` is `D0C` with the group IDs
permuted: the *sets of neurons* are identical, only their names change. **Under the `oracle` it
must therefore read EXACTLY `D0C`'s BPB — difference 0.0, not "small".** If permuting names moves
the number, the instrument is selecting on labels rather than on neurons and **every partition
comparison below it is meaningless.** (Under `fitted` it may legitimately differ, because the
router is refit and ridge is not permutation-equivariant in general; that difference is reported,
not gated.)

## 5. The verdict cell and the bands, named before the run

**The cell is the BEST partition's `fitted` BPB at `k = 16`** — the attainable selector, at the
activation E40 says 50 tok/s buys — against `D0C`'s measured **3.597108**.

| band | name | what it would mean |
|---|---|---|
| **≤ 1.005** | `PARTITION-IS-THE-LEVER` | **usable at the rate the box affords — the goal is reachable post-hoc and the T4 ask is withdrawn** |
| 1.005 – 2.50 | `PARTITION-HELPS-A-LOT` | a large real gain; worth a second probe before GPU |
| 2.50 – 3.45 | `PARTITION-HELPS` | beats `D0C`'s own **oracle** (3.449466) with an attainable router |
| 3.45 – 3.597 | `PARTITION-HELPS-A-LITTLE` | beats `D0C`'s fitted router but not its ceiling |
| **> 3.597** | **`PARTITION-IS-NOT-THE-LEVER`** | **one label family was as good as any; the T4 ask stands, strengthened** |

**The 1.005 bar is not invented.** E22 measured its `QO512+V52` fp32 arm at **1.005039 BPB** and
that same arm scored **126/160** teacher-forced — a measured anchor in this programme for "still
predicts", not a threshold chosen to be reachable.

## 6. Predictions — fixed here, before the run

1. **All three gates fire, and `G-E41C` is EXACTLY zero**, not merely small.
2. **`PARTITION-IS-NOT-THE-LEVER` is what I expect to be wrong about, so I am registering against
   myself**: I predict the best partition lands in **`PARTITION-HELPS`**, specifically
   **2.90 BPB** at `k = 16`. Better than `D0C`'s fitted 3.597 and better than its oracle 3.449,
   but nowhere near 1.005.
3. **`CONC` wins at `k = 3` and does NOT win at `k = 16`.** Concentrating mass pays most when you
   may keep almost nothing; by 6.25% the heavy neurons are already inside the selection whatever
   the grouping.
4. **`COACT` beats `D0C` at both `k`.** Grouping by co-activation is the structure a linear router
   can actually predict, and `D0C`'s family was never designed for that.
5. **`STRIPE` is the worst attainable partition at every `k`**, worse than `RAND` — deliberately
   equalising mass across groups is exactly wrong when groups must be dropped, and it should be
   worse than grouping at random.

## 7. What E41 will NOT be able to claim

- **Nothing about ternary.** fp32, like E38 and for the same reason: E37's window was 0.5941 BPB
  wide and could not resolve effects this size.
- **Nothing about training.** No healing, no straight-through, no gradient ever taken. If a
  partition helps, whether *training* into it helps more is still H0/H1's question.
- **Nothing about 10 B.** One donor at 1.5 B. The activation *fraction* is what transfers, not
  the BPB.
- **Nothing about criteria beyond these five.** E38 §7 item 1 (a criterion sweep for the
  *selector*) stays owed; this probe moves the *partition*, which is the other half.
- **One `E`.** 256 groups throughout. Group count is a third axis and is not touched.
