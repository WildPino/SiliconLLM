
# BRIEF E36 — ten billion parameters at fifty tokens a second, or the number that says why not

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. Why this probe: everything since E18 has been pricing. This one builds the thing.

The goal is **"far girare un modello grande (es 10B) a 50 token/s"**. Thirty-five probes have
priced every term in that sentence. As of E35 the price list is complete enough to *design* the
artifact rather than approach it:

* **E30**: the machine's ceiling is **36.30 GB/s**, measured, planted control at 12.5×.
* **E32**: the operative kernel is packed — `--lutblk` costs `+0.049 BPB` and lost its licence.
* **E34**: T10's attention floor **alone** reads 20.03 tok/s, so the literal 10 B shape is dead.
* **E35**: the envelope is **0.9451 G active charged weights per token** at 50 tok/s,
  gather-inclusive, and `L=16` at 4096 wide reads **56.16 tok/s** with the FFN at the floor,
  leaving **+0.1120 G** for an activated FFN.

**Nobody has ever built an artifact that is actually 10 B and asked the engine to run it inside
that envelope.** Every 10 B-shaped file this programme has benched (T10) is *dense-active*: 10.6 G
charged per token. This probe builds a file with **~10 B parameters on disk** whose **active**
slice fits E35's envelope, and measures it.

## 1. The question

> **Does an artifact with ~10 billion parameters read 50 tok/s on this box, when its activated
> slice is sized to the envelope E35 measured?**

## 2. The shape, fixed here, before it is built

`A10B`: `D = 4096, F = 46080, L = 16, NH = 32, NKV = 8, HD = 128, V = 32768`, untied head.
`F = 46080 = 256 × 180`, so a carve at `E = 256` gives `GSZ = 180` neurons per group.

| term | weights |
|---|---|
| attention, per layer | 41,943,040 |
| FFN, per layer (`3·D·F`) | 566,231,040 |
| × 16 layers | 9,730,785,280 |
| embedding + head | 268,435,456 |
| **total on disk** | **9,999,220,736 — 10.0 B** |

Charged per token at `E=256` with `k` groups active:

```
charged(k) = 822,083,584 + 35,389,440 · k
```

At E35's measured numerator of **47.254 G-w/s**:

| `k` | neurons active / 46,080 | charged | predicted tok/s |
|---|---|---|---|
| 1 | 180 (0.39%) | 0.8575 G | **55.1** |
| 2 | 360 (0.78%) | 0.8929 G | **52.9** |
| **3** | **540 (1.17%)** | **0.9283 G** | **50.9** |
| 4 | 720 (1.56%) | 0.9636 G | **49.0** |
| 6 | 1080 (2.34%) | 1.0344 G | **45.7** |

**The crossing is predicted between `k=3` and `k=4`.**

## 3. Arms

| arm | what it is |
|---|---|
| `T10-L16` | **the planted control** — E35's own artifact, re-benched, must reproduce 56.16 |
| `A10B-K1` … `A10B-K6` | the 10 B file through the engine's `--carve-k`, `k ∈ {1, 2, 3, 4, 6}` |

One 10 B file for every `k`, E26's design, so no cell can differ by file, permutation or export.
Five interleaved reps, reps outermost, idle box, operator idle.

## 4. The gates

**`G-E36A` — the planted control.** `T10-L16` within ±10% of E35's **56.16**, or this session is
not comparable to the envelope it is testing and E36 is VOID.

**`G-E36B` — the artifact must actually BE ten billion.** The exporter's own parameter count,
read from the file's JSON, must be **≥ 9.9 G**, and the file must pass `GATE V3`. **This is the
gate that stops the probe from quietly demonstrating something smaller than the goal.**

**`G-E36C` — charged accounting, zero tolerance.** For every `k`, the exporter's
`active_weights_per_token` must equal `822,083,584 + 35,389,440·k`. If it does not, §2's table is
wrong and the runner wins.

## 5. The verdict cell, named before the run

**`A10B-K3` tok/s**, with `G-E36B` satisfied.

| band | name | what it would mean |
|---|---|---|
| **≥ 50** | `TEN-B-AT-FIFTY` | **the goal's speed half is demonstrated on this box**: a ten-billion-parameter artifact, on the engine, at the target rate |
| **45 – 50** | `TEN-B-NEAR-FIFTY` | the envelope is right to within a few percent and the shape needs a small trim |
| **< 45** | `TEN-B-SHORT` | something outside the charged model costs real time at this shape, and E35's envelope does not transfer to a 10 B file |

## 6. Predictions — fixed here, before the run

1. **All three gates fire.**
2. **`A10B-K3` lands in `[49, 54]`** — the charged model says 50.9.
3. **The measurement comes in FASTER than the charged model, not slower.** E35's numerator was
   calibrated on arms with `GSZ = 56`, whose `down` runs are `3,584 B` — the steep part of E31's
   curve. `A10B` has `GSZ = 180`, so its `down` runs are `11,520 B`, where E31 measured ≈0.85
   instead of ≈0.685. **The FFN half of this file should read more efficiently per charged weight
   than the arms the model was fitted on.** If instead it comes in slow, the per-layer fixed cost
   of E35 §3.2 is bigger than the gather gain, and that is the finding.
4. **`k=6` misses 50** (predicted 45.7) and `k=1` clears it comfortably (55.1), so the crossing is
   bracketed by measured arms and not extrapolated.
5. **Registered as the honest limit, before any number exists.** **A pass here is the SPEED half
   only.** The weights are synthetic noise: this probe demonstrates that *the engine can move a
   ten-billion-parameter artifact at the target rate*, and says **nothing** about whether a
   trained model of this shape exists, is reachable by adaptation from a donor, or is any good.
   **The goal's sentence — "far funzionare un LLM già addestrato" — is not satisfied by this
   probe alone**, and the result document must say so in its first paragraph.
6. **Registered in advance**: `1.17%` activation at `L=16` is a far more aggressive MoE than
   anything this programme has validated for *quality*. Probe-4's retro-audit already demoted
   its own MoE result, and E27 measured that in-budget configurations were broken at 0.998 G.
   **This probe does not rehabilitate any of that.**

## 7. What E36 will NOT be able to claim

- **Nothing about quality**, at any activation rate. No BPB, no donor, no training.
- **Nothing about the router.** `k` is fixed and oracle-free; E23 priced a real router end to end
  and that cost is **additive** to everything here.
- **Nothing about long context.** 40-token benches, weight traffic only. KV grows with context
  and is the largest un-priced term in the envelope (E30 §8 item 2, E35 §7 item 3).
- **Nothing about other boxes.** One Zen 2, six threads, 36.30 GB/s. The *method* transfers; the
  numbers are this machine's.
