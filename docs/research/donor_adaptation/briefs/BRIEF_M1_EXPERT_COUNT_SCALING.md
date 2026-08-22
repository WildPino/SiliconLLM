# Brief M1 — does MoE quality survive as the expert POOL grows at fixed expert size and fixed k?

**Author:** the Adapter / Principal · **Date:** 2026-08-22
**Status:** DESIGN REQUEST to the Builder. **This brief does not yet specify an apparatus** — it states a
question and a confound, and asks for costed designs. I will pre-register the chosen one before it runs.

---

## 1. Where this came from

`ADAPTER_MEMO_01` §2.2e (the extrapolation ledger) put every quantity the ">20 tok/s" claim depends on
next to its validated value. **Row 3 is the one that turned out to be misread, by me.**

Probe-4 validated a **granularity**: expert size `h=128`, `top-k=8`. Its FFN was `d_ffn = 4096`, so that
meant `E = 32` experts and `8×128 / 4096 =` **25% active**. I had been carrying "25%" as a property of the
method. It is not — **it is a property of a 4096-wide FFN.**

A 100B-class donor has `d_ffn = 28672`. Hold the validated expert size and `k` and let `E` grow with width:

```
    E = 28672 / 128 = 224 experts,  top-8  ->  8×128 / 28672 = 3.57% active
```

**At 3.57% FFN active, 12.5% attention and 5-trit packing, the speed budget closes at 30.1 tok/s.** At
4 bits/weight the same configuration gives 12.1 and does not. So this row, together with D5, is what the
programme's headline rests on.

**The open question, stated so it can only have one answer:**

> Probe-4 measured quality at `top-8 of 32`. The donor configuration is `top-8 of 224`. The expert size and
> `k` are unchanged, but **the fraction of the FFN each token sees falls from 25% to 3.57%.**
> **Does the MoE-over-dense advantage probe-4 measured survive as `E` grows at fixed `h` and fixed `k`?**

---

## 2. The confound, which is why this brief asks rather than specifies

**Raising `E` at fixed `h` raises total parameters.** So a naive sweep `E ∈ {32, 64, 128, 224}` improves
quality for a reason that has nothing to do with routing, and the result would be uninterpretable.

Probe-4 already knew this and handled it — its verdict was stated **"at matched total params ~22.5M"**,
against *two* dense arms (`dense-1024` and `dense-4096`), which is why its result meant something. **Any
design here must carry the same discipline, and it is harder here because the thing being swept is the
one that moves the parameter count.**

Two things that look like the same sweep and are not, so neither may be substituted for the other:

| | token sees | active fraction | what it tests |
|---|---|---|---|
| `E=32, k=2` | **2** experts | 6.25% | fewer experts per token, small pool |
| `E=224, k=8` | **8** experts | 3.57% | **same experts per token, much larger pool** |

**Lowering `k` at fixed `E` is NOT a cheap proxy for raising `E` at fixed `k`.** The donor case is the
second row: the token still gets 8 experts, drawn from a wider pool. A design that sweeps `k` answers a
different question and must not be reported as answering this one.

---

## 3. What I want from you, before any training runs

**Do not build yet. Propose 2–3 designs and let me choose**, because the confound above is the whole
difficulty and I would rather spend a round on the design than a GPU budget on an uninterpretable result.

For each design, give me:

1. **How it isolates `E` from total parameter count.** Matched-total-params dense controls (probe-4's own
   approach), or a fixed-parameter reparameterisation, or something better. **Name what stays fixed and
   what moves.**
2. **The arms**, concretely: `E`, `h`, `k`, `d_model`, `d_ffn`, total params, active params, active
   fraction — as a table, one row per arm, computed not asserted.
3. **The cost**, in steps and in GPU-hours on a **T4** (this is going to the Owner's Kaggle operator, who
   has T4s and a 30 GPU-h/week/account ceiling — see the constraints in §5).
4. **What could make the result uninterpretable**, in your own judgement. You have built this apparatus and
   I have not.
5. Whether probe-4's existing code path (`moe_gran.pt` and whatever produced it) can be reused, or whether
   this needs new training code. **Reuse strongly preferred** — a shared apparatus makes the comparison to
   probe-4's own numbers direct rather than analogical.

**Flag it immediately if you think a small-model A/B cannot answer this question at all.** If expert-pool
scaling only shows up at a width we cannot afford to train, that is a real finding and I would rather have
it now than after a run.

---

## 4. Pre-registration, thresholds and controls — for the chosen design, before it runs

I will write these once a design is picked, but so you can size the work:

- **The comparison is MoE-vs-dense-at-matched-total-params, at each `E`** — the quantity of interest is
  whether probe-4's advantage *persists*, not the absolute BPB.
- **σ_seed = 0.005 BPB** is this project's banked seed noise. Any effect smaller than that is nothing, and
  **arms must be replicated across seeds** — replicates are load-bearing here, not optional.
- **A planted positive control is mandatory.** The instrument must be shown to detect a known-real quality
  difference before any null it reports is worth anything. Propose one.
- **Report ACHIEVED, not REQUESTED, configuration** — printed from the constructed model objects. D1 lost
  two points to a requested 90% sparsity that the code silently quantised to 100%.
- **Router health must be reported** at every `E`: dead experts, max/mean load. Probe-4 reported `0 dead,
  max/mean ≤1.47×` under Switch aux loss. **At `E=224` with the same `k`, load balancing is a far harder
  problem than at `E=32`, and a collapsed router would produce a quality result that says nothing about
  granularity.** If the router degrades with `E`, that is itself a headline finding.

---

## 5. Constraints that are sealed and not mine to move

- **Kaggle free tier, T4s**, ~30 GPU-h per week per account, three accounts. The Owner's operator runs the
  training; I do not launch it and neither do you.
- **No `--compile`** on the Windows dev box; viable on Linux cloud.
- **Never two trainers at once** on the local machine.
- This is a **from-scratch A/B**, like probe-4 — it is not donor conversion and does not touch the donor
  pipeline. It exists to license one row of the ledger.

---

## 6. Why this is worth a GPU budget at all

Because of what it gates. If `top-8 of 224` holds quality, the FFN half of the speed budget stands on our
own measured recipe carried to donor width, and the only remaining unvalidated lever is packing (D5). **If
it does not hold, the ledger's ✅ rows collapse and the programme needs a different FFN story** — and it is
much cheaper to learn that from a small A/B than from a 1.5B conversion that has already consumed its
budget.
