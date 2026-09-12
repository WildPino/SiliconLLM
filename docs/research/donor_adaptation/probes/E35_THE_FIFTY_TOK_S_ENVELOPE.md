# E35 — the 50 tok/s envelope: measure the shape that fits, do not derive it

**Verdict: `DEPTH-IS-HALVED`. `L* = 19.33`** under the model as registered — the depth at which a
T10-width model crosses 50 tok/s with its FFN at the floor. The brief predicted **19**.

**The number that matters more than the verdict**: at 50 tok/s this box affords
**0.9451 G active charged weights per token**, measured by interpolation between two arms that
bracket the goal — **56.16 tok/s at `L=16` and 40.35 at `L=24`.** For the first time in this
programme the goal is bracketed by measurements on both sides instead of approached from below.

**Brief**: `briefs/BRIEF_E35_THE_FIFTY_TOK_S_ENVELOPE.md`, pushed at `17e7fc8` before the runner
existed. **Runner**: `engine/e35_envelope.py`. **Result**: `engine/results/e35_envelope.json`,
75 s of timing, five arms, five interleaved reps. Idle box: witness **5.2% mean / 9% peak**
before, and **2.5 / 0.5 / 0.0 / 0.0 / 0.0%** between reps.

---

## 1. The gates

| gate | what it demanded | reading | |
|---|---|---|---|
| **`G-E35A`** the planted control | `T10-L48` within ±10% of E34's `T10-FLOOR = 20.03` (same box, same artifact, same flags) | **19.88**, **−0.8%** | **FIRES** |
| **`G-E35B`** | for all five arms, the exporter's `active_weights_per_token` equals the runner's independently computed `attention·L + head + carve residue`, **zero tolerance** | exact on all five | **FIRES** |
| **`G-E35C`** | every invented shape passes `GATE V3` | 5/5 (`579 / 387 / 291 / 195 / 147` tensors) | **FIRES** |

`G-E35A` at −0.8% is the strongest session-to-session agreement in the branch, and it is on
E34's own file, so this probe and E34 are the same measurement continued.

## 2. The measurement

One artifact per depth, `--carve 256 --carve-k 1` baked in, so no arm depends on a flag another
arm lacks — the defect that voided E34 run 1. Reps outermost.

| arm | `L` | charged/token | measured tok/s | registered model | deviation | spread |
|---|---|---|---|---|---|---|
| `T10-L48` | 48 | 2.2308 G | **19.88** | 20.95 | **−5.1%** | 13.0% |
| `T10-L32` | 32 | 1.5320 G | **30.24** | 30.51 | −0.9% | 10.4% |
| `T10-L24` | 24 | 1.1825 G | **40.35** | 39.52 | +2.1% | 16.2% |
| `T10-L16` | 16 | 0.8331 G | **56.16** | 56.10 | +0.1% | 8.6% |
| `T10-L12` | 12 | 0.6584 G | **71.35** | 70.99 | +0.5% | 15.9% |

**`T10-L16` reads 56.16 tok/s — above the goal — and `T10-L12` reads 71.35.** Those are the first
arms in this programme at a 4096-wide shape to clear 50, and §5 says exactly how much of a model
they are.

## 3. Predictions — scored as registered. 3 HIT / 2 MISS, and both misses are defects in my model

| # | registered | outcome | |
|---|---|---|---|
| 1 | all three gates fire | −0.8% / exact / 5 of 5 | **HIT** |
| 2 | `L*` in `[17, 21]` | **19.33** | **HIT** |
| 3 | **`L*` < 19** (I predicted my own model was biased high) | **19.33** | **MISS** — see §3.1 |
| 4 | the curve is linear in `1/charged` to within **5%** | worst **5.1%** | **MISS**, by 0.1 of a point — see §3.2 |
| 5 | this cannot produce a 10 B at 50 tok/s and will not claim to | holds; §5 | **HIT** |
| 6 | the FFN budget left at each depth is reported | §4 | reported |

### 3.1 Prediction 3 fails, and the interesting part is that the *unit* decides it

The brief's §2 model is labelled **"attention·L + head, FFN at carve k=1"** but its per-layer
coefficient is `41,943,040` — the **pure floor**, with the `k=1` residue left out. The arms
actually measured carry that residue: their measured per-layer charge is **43,679,744**
(`(2.2308 G − 0.6584 G) / 36`), 4.14% larger.

Converting the same interpolated crossing back into a depth:

| convention | per-layer coefficient | `L*` | prediction 3 (`L* < 19`) |
|---|---|---|---|
| **as registered** (pure floor) | 41,943,040 | **19.33** | **MISS** |
| residue-inclusive (what was on the box) | 43,679,744 | **18.56** | would HIT |

**Scored as registered: MISS.** I am not taking the reading that flatters the prediction — the
brief fixed the model and the brief's model gives 19.33. But the defect is real and is the same
class as E33's `G-E33C`: **a model labelled as one object whose arithmetic describes another.**
Both numbers are reported because they answer different questions — 19.33 is the depth of a *true
zero-FFN* model, 18.56 is the depth of *an arm like the ones measured*, and the second is the one
a builder would use.

### 3.2 Prediction 4 fails on one arm, and the failure has a shape

Four of five arms sit within **2.1%** of the registered model; only `L=48` misses, at **−5.1%**,
and it misses **slow**. E34's floor arm did the same thing in the same direction. A charged-weight
model cannot see a *per-layer fixed cost*, and a fixed cost is exactly what shows up as "deep arms
read slower than their weight count says". E26 measured the carve machinery at **−6.36%** before a
single group falls, which is the right order of magnitude.

**So the model is excellent where the goal lives (`L ≤ 32`: worst 2.1%) and pessimistic-to-wrong
at depth.** That is the better half to be right about, but prediction 4 was registered at 5%
across *all* arms and it broke.

## 4. The envelope, which is the deliverable

**At 50 tok/s this box affords `0.9451 G` active charged weights per token.** Interpolated between
the two bracketing arms, both measured this session: the implied numerator is **47.254 G-w/s**,
against E34's independently measured dense **46.74** (+1.1%).

Room left for an activated FFN, after attention and head are paid:

| `L` | charged | **room to the 50 tok/s line** |
|---|---|---|
| 48 | 2.2308 G | **−1.2858 G** (over budget by 2.4×) |
| 32 | 1.5320 G | −0.5869 G |
| 24 | 1.1825 G | −0.2375 G |
| **16** | 0.8331 G | **+0.1120 G** |
| **12** | 0.6584 G | **+0.2867 G** |

At `L=16` that room is **111,980,220 weights per token**, or **7.0 M per layer**. A carve group at
`E=256` is 56 neurons and costs `3·D·56 = 688,128` weights, so the room is **≈10 groups per layer
— 560 of 14,336 neurons, 3.9% activation.**

**And this is gather-inclusive, not an idealisation.** Every arm here is a `quant==4` carved
artifact reading one group per layer, so the measured numerator already pays E31's penalty at
`GSZ=56` granularity (`down` runs of `56 × 64 = 3,584 B`, the steep part of E31's curve). The
envelope is priced with the gather in it.

## 5. What this is NOT: `L=16` at 4096 wide is 0.8 B of attention, not a 10 B

Registered as prediction 5 before the run, and it holds: **`T10-L16` at 56 tok/s is not a 10 B
model.** It is 16 layers of attention (0.67 G), a head (0.13 G), and essentially no FFN.

What E35 actually licenses is a **spec**, and the spec is an MoE statement:

> **a 10 B at 50 tok/s on this box means ~16 layers at 4096 wide, with ~560 of its FFN neurons
> activated per layer per token, read in carve groups — which puts the *total* FFN pool at
> whatever size the parameter count wants, because only the activated slice is charged.**

Sizing the pool: 10 B total at `L=16, D=4096, V=32768` needs about **46,000 FFN neurons per
layer** (`3·4096·F·16 ≈ 9.06 G` after attention, head and embedding), so the activation rate is
**≈1.2%** — 560 of ~46,000. That is desk arithmetic on measured inputs and it is **not** measured
here; it is the shape of the next probe, not a claim of this one.

Two constraints already measured that any such design must respect:

* **E31's granularity**: those 560 neurons must arrive in contiguous runs. At row granularity the
  read delivers `0.579` of the dense rate and the envelope shrinks by a third; at ≥32 KB blocks it
  delivers `0.944`. The carve groups used here sit between the two.
* **E27/E29's depth result**: 16 of 48 layers is a *deep* cut, and E27 measured that depth cuts
  cheaply only **if the right layers go** (`L21-MINRES` 113/160 vs `L21-LAST` 57/160 at identical
  cost). Nothing here says a 16-layer model is any good — that is the training axis, untouched.

## 6. What E35 cannot claim

- **Nothing about quality.** Synthetic noise weights, no BPB. Whether a 16-layer 4096-wide model
  is worth running is E24/E27/E29's axis and this probe does not touch it.
- **Nothing about width, rank, or KV heads.** Depth only, one axis at a time. Their costs are
  **not** additive without measurement.
- **Nothing about the KV cache.** 40-token benches, weight traffic only. KV grows with context
  and with depth, so `L*` here is an **upper** bound on affordable depth (E30 §8 item 2, still
  owed and now more load-bearing).
- **Nothing about `--lutblk`** (revoked by E32; all arms packed).

## 7. Owed after E35

1. **Build the spec and measure it** — `L=16`, `D=4096`, a large FFN pool, ~10 carve groups
   activated, total parameters ≈10 B, and see whether it reads 50 tok/s **and** counts as 10 B.
   That is the direct demonstration the goal asks for, on the speed axis, and E35 has now made it
   a concrete and falsifiable target rather than an aspiration.
2. **The per-layer fixed cost** (§3.2) is unmeasured and is what makes deep arms read slow. It
   bounds how much of E30's remaining `1.57×` kernel headroom is actually reachable.
3. **KV traffic at long context** (E30 §8 item 2) — now the largest un-priced term in the
   envelope.
4. **Time `--lut --lut-group 32`** (E32 §8 item 1) — unchanged, still the cheapest numerator.
