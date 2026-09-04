# BRIEF T1 — What does the engine's OWN ternarization cost the donor?

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-04.**
**Depends on: `benchmarks/phase60/e1_export.py` (the rule), `engine.c` (the consumer),
`density/common.py` (the eval), `SPEED_LEDGER.md` (why it matters now).**

---

## 1. The gap this closes, and it is the most basic one in the programme

`engine.c` is a **ternary** engine. Every weight it multiplies is in {−1, 0, +1} with an fp32 scale.
That is not an optimisation on top of the architecture — it **is** the architecture, and it is the
reason the whole bandwidth argument works.

The donor-adaptation programme has measured, on Qwen2.5-1.5B: co-activation carving (D0, D0c),
magnitude and structured pruning (D1), basis rotation (D2), low-rank (D3), Hessian reconstruction
(D4), sparsity-vs-BPB (S1). **Every one of them was measured on fp32 weights.**

> **Nobody has ever measured what happens when the donor's weights are ternarized.**

There is no probe, no brief and no result. `grep -rl ternar benchmarks/donor_adaptation/` returns the
inventory scripts and the speed budget, and nothing that evaluates BPB. The programme has spent weeks
pricing *structural* damage on a model that would have to survive a *numerical* conversion first, and
the numerical conversion is the one the engine actually requires.

The project's own precedents point both ways and neither is about a donor:
- Phase 61: ternarizing the **SSM projections** of *our* model cost **+0.018–0.022 BPB** and they
  were left fp32 — "precision-hungry organs, measured".
- Probe-1: our **ternary MLP** cost **+0.028 BPB**.

Those are our model, trained *with* quantization in the loop (QAT). **A donor is not.** It was trained
in bf16 with no knowledge that its weights would be rounded to three values. Post-training
ternarization is a categorically harder problem than the two numbers above, and quoting them as
reassurance would be exactly the transplant this programme has been caught making twice.

> **The question this brief pre-registers:**
> **Applying the engine's exact ternarization rule to a pretrained donor, organ by organ, what does
> it cost in BPB — and does a finer scale granularity buy it back?**

## 2. The rule under test — the engine's, not a generic one

From `benchmarks/phase60/e1_export.py:34-38`, verbatim, BitLinear158:

```python
scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)   # per OUTPUT row, mean over INPUT features
wq    = (w / scale).round().clamp(-1, 1)                    # {-1, 0, +1}
deq   = wq * scale
```

Weights are consumed by `matvec_lut_full` as ternary codes with **one fp32 scale per output row**
(`gate_sc[l]` is `MLP_HID` long, `down_sc[l]` is `D` long — `engine.c:196, 245-247`). Activations are
separately quantized int8 at `AQ = 63` (`engine.c:63, 185-186`).

**This probe measures weight ternarization only.** Activation int8 is a second, separable conversion
and is registered here as **NOT tested** (§6), so that a bad result cannot be blamed on it and a good
one cannot be claimed to include it.

## 3. Arms — and the granularity sweep is in from the start

Donor Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`. Eval slice `heldout`,
24 × 512, seed 1234, `ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
12,264 predicted tokens, 51,870 scored bytes. Per-sequence nats stored so every SE is **paired**.

| arm | what is ternarized | scale granularity | role |
|---|---|---|---|
| **base** | nothing | — | **replication gate: must reproduce 0.7675949641196624** |
| **F** | FFN: gate, up, down | per-row (engine's rule) | primary |
| **A** | attention: q, k, v, o | per-row | primary |
| **FA** | FFN + attention | per-row | **primary — this is the engine's actual conversion** |
| **FA+H** | FFN + attention + lm_head | per-row | ⚠ **the head is TIED to the embedding** on this donor (`tie_word_embeddings: true`), so this arm ternarizes the embedding table too. Reported separately and never folded into FA |
| **F/g256** | FFN | scale per 256 input features | granularity sweep |
| **F/g64** | FFN | scale per 64 input features | granularity sweep |
| **Z** | FFN, **random sign** at the same per-row scale | per-row | **planted control — see §3.1** |

### 3.1 ⚠ The planted control, non-negotiable

**Arm Z replaces each ternary code with a random sign, keeping the same per-row scale and the same
zero-fraction.** It preserves every norm and every shape and destroys only the *information*.

This programme's standing law is that it fails on the **plausible artefact**, not the wrong number,
and that an instrument must be shown to **fire on a known positive before its nulls mean anything**.
If arm Z does not come back catastrophically worse than arm F, then the harness is not really
substituting weights and **no other number in this probe may be reported**. Z is cheap and it is what
makes F trustworthy.

### 3.2 What is held fixed

Same `d_baseline` code path, same batch size, same dtype (fp32 master, dequantized to fp32 — the
engine computes `wq * scale`, so the arithmetic under test is exactly `deq`), same
`torch.set_num_threads`, same seed. **Only the weight tensors change.** No activation quantization,
no sparsity, no carve, no router.

## 4. Pre-registered decision rule — fixed before any result

`Δ(arm)` = BPB(arm) − BPB(base). Baseline **0.7675950**, σ_seed **0.005**. The label is read off
**Δ(FA)**, the engine's actual conversion.

| outcome | condition on Δ(FA) | what it means |
|---|---|---|
| **CONVERSION-CHEAP** | ≤ **0.05** (10 σ_seed) | the donor survives the engine's numeric format nearly intact. The route is wide open and every structural probe should be re-read as *additional* to a near-free base |
| **CONVERSION-COSTLY** | **0.05 – 0.30** | it costs real quality but is in the range where healing (QAT, calibration, layer-wise correction) plausibly recovers it. Names the next brief |
| **CONVERSION-FAILS** | **> 0.30** (60 σ_seed) | naive post-training ternarization does not work on this donor. **Then no amount of speed work matters until it is fixed**, and the programme's whole ordering must change: healing becomes the critical path, ahead of every structural probe |
| **VOID** | arm Z is not ≫ arm F | the instrument is not substituting weights. Report nothing else |

**Secondary, pre-registered separately so it cannot be used to rescue the primary:** granularity is
**GRANULARITY-RECOVERABLE** if `Δ(F/g64) ≤ Δ(F) − 0.10`, i.e. a finer scale buys back at least
0.10 BPB. D0c is on record that a free parameter measured at one extreme is not a result; the
engine's per-row rule is one extreme (the coarsest scale it could use), so the sweep is here from the
start rather than after a negative.

**Cost of finer granularity, to be reported alongside its benefit and not separately:** a per-row
scale costs 4 bytes per output row; a scale per `g` input features costs `4 · in/g` bytes per row.
For `down_proj` at `in = 8960`: per-row is 4 B, `g=64` is 560 B per row against 4480 B of codes —
**+12.5% on the weight bytes**, which `SPEED_LEDGER.md` prices directly. A granularity win that costs
more bytes than it saves is not a win and must be reported as such.

## 5. Cost

Eight arms × one BPB pass. D0c measured 131–159 s per pass at 6 threads on this machine, so
**≈ 20 minutes**, CPU only, no GPU, no training, no download. It is the cheapest unanswered question
in the programme and it sits underneath every answered one.

## 6. What this brief does NOT claim and does NOT test

- It does **not** test activation quantization (int8, `AQ=63`). Separable, registered as untested.
- It does **not** test the engine's *packing* (base-3 g=2 codes). Packing is lossless given the codes;
  this is about the codes.
- It does **not** test any healing method — no QAT, no calibration, no layer-wise correction. If the
  outcome is CONVERSION-COSTLY or CONVERSION-FAILS, healing is the **next** brief, not this one.
- It does **not** test another donor or another size. Every arm is Qwen2.5-1.5B, and the same
  scale-transplant warning that forced S1's Amendment 1 applies here.
- It does **not** measure speed. Ternarization's speed benefit is already priced in
  `SPEED_LEDGER.md`; this is its quality bill.

## 7. Reporting

`probes/T1_DONOR_TERNARIZATION.md`. Report: Δ and paired SE for every arm, the arm-Z control, the
replication of the baseline, the granularity sweep with its byte cost, the §4 label verbatim, and —
if CONVERSION-FAILS — an explicit statement that the programme's ordering must change, because every
structural result on record was measured on weights the engine cannot consume.
