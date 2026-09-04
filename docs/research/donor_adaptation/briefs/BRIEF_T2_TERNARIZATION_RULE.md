# BRIEF T2 — Was +4.74 BPB the ternary FORMAT, or one naive RULE for reaching it?

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-04.**
**Depends on: `probes/T1_DONOR_TERNARIZATION.md`, `probes/D4_RECONSTRUCTION.md`.**

---

## 1. The question

T1 measured the engine's ternarization at **+4.738 BPB = 948 σ_seed** on Qwen2.5-1.5B and returned
**CONVERSION-FAILS**. That result now blocks the whole programme: the runtime built in
`R1_DONOR_RUNTIME.md` executes a donor at 40–46 tok/s, and the model it executes is one this
conversion has destroyed.

**But T1 tested exactly one rule, and said so.** Its §5 registers, in advance:

> *"It does **not** test another ternarization rule. BitLinear158 is the engine's rule; that is why
> it is the one measured."*

That rule is `scale = mean|w|` per output row, then round-to-nearest of `w/scale`. It is the
**simplest possible** post-training ternary quantizer: no error feedback, no threshold search, no
use of the activations. Round-to-nearest is known to be the weakest option in the quantization
literature, and this programme has been caught **twice** measuring at one extreme of a knob that was
never swept — D0c and D4b are both on record.

> **The question this brief pre-registers:**
> **Is +4.738 BPB a property of the ternary FORMAT, or of one naive RULE for getting into it?**
> If a better rule recovers most of it, the engine's format survives and healing may not be needed.
> If no rule helps, the format itself is the wall and training is unavoidable.

**Why this before GPU healing:** healing is quantization-aware training — expensive, slow, and it
would be started without knowing whether the damage is even reducible by cheaper means. Every arm
here is CPU-only, needs no gradients and no labels, and takes minutes.

## 2. Arms — all reaching the SAME format the engine consumes

Every arm produces weights in `{-1, 0, +1}` with one fp32 scale per output row, so **every arm is
executable by `donor_engine.c` unchanged**. Only the choice of codes and scale differs.
Fixed: donor Qwen2.5-1.5B rev `8faed761…`, shared eval slice (`heldout` 24×512 seed 1234,
`ids_sha256 = a1a48dc9…`), FFN organs only (`gate/up/down`) unless stated, paired SEs.

| arm | rule | costs |
|---|---|---|
| **base** | none | replication gate: must reproduce 0.7675950 |
| **R0** | **BitLinear158**, `α = mean|w|`, round-to-nearest — **T1's rule** | replication: must reproduce **+3.309099** |
| **R1** | **TWN**: threshold `Δ = 0.7·mean|w|`, then `α = mean(|w| : |w| > Δ)` | free |
| **R2** | **α-search**: per row, scan `Δ/α` on a grid and keep the pair minimising `‖w − αq‖²` | seconds |
| **R3** | **activation-weighted α-search**: same search, but minimising `‖(w − αq)·diag(d)‖²` with `d` the per-input activation RMS from the calibration slice | one calibration pass |
| **R4** | **GPTQ / OBQ error compensation**: quantize column by column, and after each column push its error onto the not-yet-quantized columns through the inverse Hessian of the layer inputs | the expensive one; still no training |
| **Z** | R0 codes with **random signs** | reference for "how bad is destroyed" (T1: +4.001) |
| **I** | identity substitution | **instrument control — must reproduce base bit-exactly** |

### 2.1 Why R4 is the arm this brief exists for

R0–R3 all quantize each weight in isolation. **R4 is the only arm that uses the fact that a layer's
inputs are correlated**: having rounded column *j* the wrong way, it adjusts the remaining columns to
cancel the error that rounding introduced. It is the standard answer in the literature for
post-training quantization and it is **not training** — one calibration pass, one Cholesky per organ,
no gradients.

**This programme already owns the machinery.** D4 built layer-wise Hessian accumulation
(`d4_reconstruction.py`) and its `calib_eval_disjointness` discipline; R4 reuses it rather than
re-deriving it, and inherits D4's warning that the solve is only as good as its calibration budget
(`BRIEF_D4B_CALIBRATION_BUDGET.md`, still unrun).

### 2.2 The control that must not be skipped

**Arm I** — identity substitution through the same code path — must reproduce the baseline
**exactly**. T1's pre-registered control (arm Z) was mis-specified: it required random signs to be
≫ the treatment, which conflated an instrument property with a scientific claim, and it returned
VOID on a probe whose numbers were in fact sound. **Arm I is the instrument control here; arm Z is
kept only as a descriptive reference point and gates nothing.** That correction is recorded in
`T1_DONOR_TERNARIZATION.md` §3 and is not repeated as a mistake.

## 3. Calibration and disjointness

R3 and R4 need activations. They use the **calibration** split, never the eval slice, and the
disjointness assertion is re-run and recorded, exactly as D4 does. `calib_tokens = 16384` to match
D4's registered operating point — **and that is a knob D4b was written to sweep and never did, so
the R4 number here is a floor, not a ceiling, and must be reported as such.**

## 4. Pre-registered decision rule — fixed before any result

Let `Δ(arm)` = BPB(arm) − BPB(base) on the FFN organs. Reference: `Δ(R0) = +3.309099` from T1.
Let `best` = the smallest Δ among R1–R4.

| outcome | condition | what it means |
|---|---|---|
| **RULE-BOUND** | `best ≤ 0.50` | the format is fine and T1's number was the rule. **The conversion is essentially solved without training**; re-run T1's FA arm with the winning rule and re-price everything |
| **RULE-HELPS** | `0.50 < best ≤ Δ(R0) − 1.00` | a better rule recovers a large part but not enough. Healing is still needed but starts from a much better place; report the recovered fraction |
| **RULE-MARGINAL** | `Δ(R0) − 1.00 < best ≤ Δ(R0) − 0.20` | rules move it a little. **The format is the wall**, and QAT/distillation is the only route |
| **FORMAT-BOUND** | `best > Δ(R0) − 0.20` | no rule helps at all. T1's negative **hardens** into a statement about ternary itself at this width, and the next question is width, not rule |
| **VOID** | arm I does not reproduce base exactly | instrument broken; report nothing else |

**0.50 BPB** is the RULE-BOUND bar because it is 100 σ_seed and still far worse than anything this
project would ship — it is deliberately not a "success" threshold, only the point at which the
diagnosis flips from *format* to *rule*. **These thresholds are fixed here, before the run.**

**No extrapolation.** Whatever wins on the FFN may not be assumed to transfer to attention or to the
head without measuring those separately, and may not be assumed to transfer to another donor size —
the same warning that forced S1's Amendment 1.

## 5. What this brief does NOT claim and does NOT test

- It does **not** test QAT, distillation or any training. If the outcome is RULE-MARGINAL or
  FORMAT-BOUND, that is the next brief.
- It does **not** test activation quantization (int8, `AQ=63`), still untested since T1.
- It does **not** sweep R4's calibration budget. D4b is the brief for that and it is still unrun;
  R4's result here is therefore a **floor**.
- It does **not** change the engine's format. Every arm lands in `{-1,0,+1}` + per-row fp32 scale,
  and `donor_engine.c` runs any of them unmodified. **A rule that needed a different format would be
  out of scope, because the format is the architecture.**
- It does **not** test another donor or size.

## 6. Cost

R0–R3 and I are one BPB pass each (~140 s at 6 threads, measured in T1). R4 adds one calibration
pass plus a Cholesky and a column sweep per organ. **CPU only, no GPU, no gradients, well under an
hour.** Against a healing run measured in GPU-hours, this is the cheap thing that must be done first.

## 7. Reporting

`probes/T2_TERNARIZATION_RULE.md`. Report Δ and paired SE for every arm, the arm-I control, the
replication of `Δ(R0)`, the §4 label verbatim, and — if RULE-BOUND or RULE-HELPS — an explicit
restatement of T1's verdict in its light, including which of T1's conclusions no longer stand.
