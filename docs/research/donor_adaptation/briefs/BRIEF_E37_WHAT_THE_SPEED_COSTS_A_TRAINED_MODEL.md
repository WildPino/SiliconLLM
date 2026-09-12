
# BRIEF E37 — what E36's speed costs a model that was actually trained

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. The gap this closes, stated as a structural fact rather than a complaint

E36 ran a ten-billion-parameter artifact at 49.96 tok/s with **1.17% of its FFN neurons active**.
Its weights were noise, and the probe said so in its own first paragraph.

Look at what the branch actually contains:

| probes | weights | quality | speed |
|---|---|---|---|
| E21, E22, E23, E24, E27, E29 | **real, trained** | measured | **explicitly "no timing"** |
| E25, E26B, E28, E30–E36 | **synthetic noise** | none | measured |

**The two halves of the goal have never been measured on the same object.** Every "inside
budget" claim on the quality side (E19, E22 §, E23 §7, E24, E27) is a *weight-count* claim
priced with a numerator that E32 revoked and that E36 has since split into two different rates.
No probe in this programme has ever put a stopwatch and a BPB on one file.

**And there is a coincidence that makes the question answerable today rather than at 10 B.**
S15 (Qwen2.5-1.5B, real, local, anchored by E1) has `F = 8960`, and the label file this
programme already owns cuts it into `E = 256` groups of exactly 35 neurons. So **`k = 3` at S15
is `3/256 = 1.17%` activation — the same rate A10B needed at 10 B.** E36's requirement can be
put to a genuinely trained model, on the real engine, this session.

## 1. The question

> **When a model that was actually trained is forced to the activation rate E36's speed
> requires, how much quality is left — and what is the exchange rate between BPB and tok/s
> when both are read off the SAME artifact?**

## 2. The object

Donor **Qwen2.5-1.5B** (`S15`: `D=1536, F=8960, L=28, V=151936`), real weights, exported by
`engine/qwen_export.py` under the **same rule and head treatment E1/E16 used for the published
anchor**, so the dense arm is comparable to a number already in the record.

* labels: `density/results/d0c_labels/labels_E256.npz` — 256 groups, 35 neurons each, uniform.
  Permuting `F` is **exact**: it permutes rows of `gate`/`up` and columns of `down` and changes
  nothing the model computes.
* routers: **fitted**, by importing `ternary/e23_router.py`'s `fit_routers_multi` rather than
  reimplementing it. E26's carved real-donor export writes a **synthetic** router
  (`--carve-seed`), which is fine for pricing and useless for quality — a random router selects
  random neurons. **A trained router is the whole point here**, and the exporter cannot
  currently write one. Adding that path (`--carve-router`) is part of this probe.

## 3. Arms

One artifact per container, and the `k` sweep is done with the engine's `--carve-k` override so
**no cell can differ by file, permutation, router or export** (E26's design; E34's void run 1 is
why every arm passes the flag explicitly even when it equals the file's).

| arm | what it is |
|---|---|
| `S15-DENSE` | the real donor, uncarved — **the published anchor**, and the speed reference |
| `S15-K256` | the carved file with **every** group kept — **the planted control** |
| `S15-K64`, `K32`, `K16`, `K8`, `K4`, **`K3`**, `K2`, `K1` | the sweep, 25% down to 0.39% activation |

Both axes on every arm: **BPB** by E1's protocol (`--seqlen 512`, 51,870 scored bytes, 12,264
predictions, chance line **4.069819**) — deterministic, so it may be measured under load — and
**tok/s** by E36's protocol — idle box, operator idle, reps outermost.

## 4. The gates

**`G-E37A` — the planted control, and it must FIRE before any null here counts.**
`S15-K256` keeps every group, so the carved artifact computes *exactly* what the dense one
computes. Its BPB must equal `S15-DENSE`'s to **< 1e-6**. This is what proves that the labels,
the permutation, the fitted router, the `quant==4` container and the `--carve-k` path are
**inert when nothing is dropped** — and therefore that any damage at lower `k` is the sparsity
and not my plumbing. A probe that cannot pass this cannot report a quality number.

**`G-E37B` — the anchor.** `S15-DENSE` through the engine must reproduce E1/E16's published
**3.475706372** to within `0.001`. Otherwise this session is not measuring the object the record
describes.

**`G-E37C` — charged accounting, zero tolerance.** At every `k`, the exporter's
`active_weights_per_token` must equal the runner's independently computed
`attention·L + head + E·D·L + 3·D·(F/E)·k·L`.

**`G-E37D` — the router is real.** The fitted routers must beat a **seed-matched random router**
on held-out group-mass prediction, reported per layer. A router that does not beat random is a
synthetic router wearing a better name, and its quality numbers would be E26's numbers.

## 5. The verdict cell, named before the run

**`S15-K3` BPB** — 1.17% activation, the rate E36 requires — with `G-E37A` satisfied.

| band | name | what it would mean |
|---|---|---|
| **≤ 3.70** | `SPARSITY-SURVIVES` | a trained dense donor tolerates E36's activation rate within `+0.22` BPB, and adaptation alone could reach the goal |
| **3.70 – 4.069819** | `SPARSITY-DEGRADES` | it still beats the chance line, so the shape is learnable but the donor's weights are in the wrong place for it |
| **> 4.069819** | `SPARSITY-DESTROYS` | **worse than chance**: E36's speed spec is unreachable by transforming a pretrained dense model, and training inside the format is not an optimisation but a precondition |

## 6. Predictions — fixed here, before the run

1. **All four gates fire.**
2. **`SPARSITY-DESTROYS`. I expect the verdict cell to come in worse than the chance line**, and
   I am registering that in advance so the result cannot be dressed up afterwards as a surprise.
   E19 measured every carved arm collapsing, E27 measured every in-budget configuration broken,
   and 1.17% is an order of magnitude past anything either tested. **If this is right, the
   headline of E37 is not a failure of the engine — it is the measured proof that the remaining
   work is TRAINING and not conversion.**
3. **The curve has a knee, and locating it is the constructive half.** I predict BPB is within
   `+0.10` of dense down to **`k ≥ 64`** (25% activation) and crosses the chance line somewhere
   in **`k ∈ [4, 16]`**. The two numbers that matter are *the largest `k` that still costs
   nothing* and *the smallest `k` that still beats chance*.
4. **Speed will NOT be the binding constraint at S15**, and that is deliberate: at `D=1536,
   L=28` the dense donor already clears 50 tok/s. E37 is not a speed probe. Its speed column
   exists so that, for the first time, an exchange rate `ΔBPB per Δtok/s` is measured on one
   object rather than assembled from two.
5. **The exchange rate will be worse than the charged-weight model predicts**, for the reason
   E36 measured: an FFN weight costs `1.25×` an attention weight, so buying speed by dropping
   FFN groups buys less rate per unit of quality than a weight count says.
6. **Registered limitation.** This is **one donor at one scale with one label set and one router
   family.** A `SPARSITY-DESTROYS` verdict does not prove no sparse 10 B exists — it proves this
   *conversion* does not produce one, which is a statement about adaptation, not about the
   architecture. Saying otherwise would be the scale-law error this programme has made before.

## 7. What E37 will NOT be able to claim

- **Nothing about a trained-sparse model.** It measures what happens when a **dense-trained**
  donor is forced sparse. Whether training *inside* the format works is H0's and rung-1's axis
  and is untouched here.
- **Nothing about 10 B.** S15 is 1.5 B. The activation *rate* matches A10B's; the shape does
  not. E15 already measured that donor results do not automatically transfer up.
- **Nothing about healing.** No fine-tuning, no distillation, no recovery step of any kind.
  If the verdict is `SPARSITY-DESTROYS`, the obvious next question — how much of it healing
  buys back — is precisely the question this probe is designed to make worth asking.
- **Nothing about other routers.** One ridge family, one target. E23 priced a real router end
  to end and its cost is additive to every rate here.
