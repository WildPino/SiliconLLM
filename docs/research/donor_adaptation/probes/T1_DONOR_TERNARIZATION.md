# T1 — What the engine's own ternarization costs the donor

**Pre-registered label: VOID** (§4 of the brief, read verbatim — and the VOID is the brief's fault).
**Corrected label after a repaired control: CONVERSION-FAILS, by a very wide margin.**
**Date: 2026-09-04. Donor: Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`.**
**Brief: `briefs/BRIEF_T1_DONOR_TERNARIZATION.md` @ `95cdf33` — pushed before any measurement.**
**Raw: `density/results/t1_ternarize.json`; per-sequence nats in `density/results/t1_arms/`.**
**Driver: `benchmarks/donor_adaptation/ternary/t1_ternarize.py`.**

> **Not audited.** Written by the figure that specified the brief and read the numbers.

---

## 1. The gap this closed

`engine.c` is a **ternary** engine. The donor-adaptation programme had measured carving (D0, D0c),
pruning (D1), basis (D2), low-rank (D3), reconstruction (D4) and sparsity (S1) on Qwen2.5-1.5B —
**every one of them on fp32 weights.** Nobody had measured what happens when the donor's weights are
converted to the format the engine actually consumes.

The rule under test is the engine's own, verbatim from `benchmarks/phase60/e1_export.py:34-38`
(BitLinear158): `scale = w.abs().mean(dim=1).clamp_min(1e-5)`, `wq = (w/scale).round().clamp(-1,1)`,
and the engine computes `wq * scale`.

## 2. The result

Shared eval slice, `heldout` 24×512 seed 1234, `ids_sha256 = a1a48dc9…`, 12,264 predicted tokens.
Paired sequence-bootstrap SEs. σ_seed = 0.005.

| arm | what is ternarized | BPB | **Δ** | ±SE | σ_seed |
|---|---|---|---|---|---|
| **base** | nothing | 0.767595 | — | — | — |
| **F** | FFN (gate, up, down) | 4.076694 | **+3.309099** | 0.152596 | **662** |
| **A** | attention (q, k, v, o) | 4.501507 | **+3.733912** | 0.082077 | **747** |
| **FA** | both — *the engine's actual conversion* | **5.505834** | **+4.738239** | 0.176575 | **948** |
| Fg256 | FFN, scale per 256 inputs | 4.097748 | +3.330153 | 0.146938 | 666 |
| Fg64 | FFN, scale per 64 inputs | 4.152003 | +3.384408 | 0.155648 | 677 |
| Z | FFN, **random signs**, same scales | 4.768852 | +4.001257 | 0.087648 | 800 |
| **I** | **identity substitution** | **0.767595** | **0.000000** | — | **0** |

The baseline replicates the programme's standing 0.7675950.

## 3. The control, the VOID, and my error

The brief's §3.1 made arm **Z** — same scales, same zero-fraction, **random signs** — the
non-negotiable control, requiring `Z > 2×F` on the grounds that "if arm Z does not come back
catastrophically worse than arm F, the harness is not really substituting weights."

Measured: `Z = +4.001`, `F = +3.309`. **Z is worse, but only by 1.21×**, so the rule returns
**VOID**, and VOID is reported here verbatim.

**The rule was mis-specified, and that is my error, not the data's.** It conflated two different
things:

- an **instrument** property — *does the harness really replace the weights?*
- a **scientific** claim — *do the signs carry information?*

Only the first belongs in a control. The measured answer to the second is that **ternarization at
this rule destroys nearly as much as randomizing the signs outright**: the whole gap between "keep
the signs the donor learned" and "throw them away" is 0.69 BPB inside a 3.3–4.0 BPB catastrophe.
That is a *finding*, and my control was built so that finding would trip it.

**The control that should have been written** was added and run: arm **I** substitutes every weight
with itself, through the identical code path, on all 196 tensors. It reproduces the baseline
**bit-exactly — `abs diff = 0.000e+00`.** The substitution machinery is sound, so the Δ values above
are readable.

**No threshold was changed.** §4's 0.05 and 0.30 are untouched. Applying the primary rule to
`Δ(FA) = +4.738`:

> **CONVERSION-FAILS** — *"naive post-training ternarization does not work on this donor; no amount
> of speed work matters until it is fixed, and healing becomes the critical path ahead of every
> structural probe"*

`+4.738 BPB` is **948 σ_seed** and **15.8× the FAIL threshold**. The baseline is 0.7676; the
converted model is 5.5058. It is not marginal.

## 4. What this does to the programme's ordering

**4.1 Every structural result on record was measured on weights the engine cannot consume.** D0's
carve (+1.09 at E=32, +0.706 at E=128), D1's pruning, D4's reconstruction, S1's sparsity — all fp32.
They are not wrong, but they are **all downstream of a conversion that costs +4.74 BPB on its own**,
and none of them was ever composed with it. The natural next question — *does carving a
**ternarized** donor cost the same as carving an fp32 one?* — has never been asked.

**4.2 The speed ledger's premise is untouched but its urgency is inverted.** `SPEED_LEDGER.md` prices
the *speed* of a ternary donor; T1 prices its *quality*, and the quality is the binding constraint by
a wide margin. A model at 5.5 BPB running at 100 tok/s is not a result.

**4.3 Finer scale granularity does not buy it back — pre-registered as a secondary and reported as
NOT-RECOVERABLE.** `Fg64` is **0.075 BPB worse** than per-row, not better. Whatever destroys the
donor is not the coarseness of the scale, so the obvious cheap knob is closed before it was tried at
cost. (Registered limit: this swept the scale *granularity*, not the *rule* — an absmax or a
learned scale is a different lever and is untested.)

**4.4 Healing is now the critical path, and it is training.** Post-training ternarization is the
naive method; the literature's answer is quantization-aware training or layer-wise distillation, and
this programme's own precedents (Phase 61's +0.018 on SSM projections, probe-1's +0.028 on a ternary
MLP) are **models trained with quantization in the loop**, which a donor is not. **That makes GPU
time the resource that matters next**, and Kaggle is available for it.

## 5. What this probe does NOT claim

- It does **not** test activation quantization (int8, `AQ=63`). Separable and untested, per §6 of
  the brief.
- It does **not** test any healing method. That is the next brief, not this one.
- It does **not** test another donor or another size. Qwen2.5-1.5B only — and the same
  scale-transplant warning that forced S1's Amendment 1 applies with full force, especially given
  that larger models are widely reported to quantize *better*.
- It does **not** test another ternarization rule. BitLinear158 is the engine's rule; that is why it
  is the one measured.
- The arm-Z reading in §3 is **descriptive**: no threshold was pre-registered for "how much worse
  than random is acceptable", and none is invented here.

## 6. Reproduce it

```
D_THREADS=6 python benchmarks/donor_adaptation/ternary/t1_ternarize.py
D_THREADS=6 T1_ONLY=I python benchmarks/donor_adaptation/ternary/t1_ternarize.py   # the control
```
Resumable per arm (`density/results/t1_arms/<tag>.npy`). ~140 s per arm at 6 threads, 8 arms.

## 7. Owed

- **A Controller audit**, including of §3's decision to report VOID and then apply the primary rule
  behind a repaired control. That reasoning is mine and it is exactly the kind that needs a second
  reader.
- The composition question in §4.1: carve a **ternarized** donor and see whether the damages add.
- A healing brief. It is the critical path now.
