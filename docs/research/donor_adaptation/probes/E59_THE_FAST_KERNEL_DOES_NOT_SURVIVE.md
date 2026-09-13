# E59 — the fast kernel is real, and it eats the model. But the gate that would have missed that is the finding.

**Verdict: `FAST-AND-LOSSY`.** Scores `briefs/BRIEF_E59_DOES_THE_FAST_KERNEL_SURVIVE_A_TRAINED_MODEL.md`,
pre-registered and pushed (`c8aa28a`) before the runner existed. Runner
`benchmarks/donor_adaptation/engine/e59_lut.py` (`e4e5c87`), results `results/e59_lut.json`,
log `D:/_ktmp/e59_run.log`. Closes E13 §8 item 2, open since 2026-09-07.

**`--lutblk` is worth ×1.253 at 0.5 B and ×1.333 at 1.5 B on trained weights, intervals separated
— and it changes 42–74% of the tokens the packed path emits from the same bytes, depending on
the activation grouping.** The obvious
fidelity test, scoring each arm against HuggingFace, would have **passed it**: 3/160 → 4/160 and
10/160 → 10/160. That column is on its floor and cannot show damage, and refusing it the verdict
before the run is the reason E59 has an answer at all.

---

## 1. `G-E59a` — the planted control, both sides

| artifact | (i) `--lut` vs `--lutblk` | (ii) `PACKED` vs `--lutblk` | `G-E59a` |
|---|---|---|---|
| `05b_tqh` | **160/160 identical** | **41/160 equal** — differs | **FIRES** |
| `15b_tqh` | **160/160 identical** | **55/160 equal** — differs | **FIRES** |

Side (i) reproduces E13's `G-M0` sha256 identity on **trained** weights, six days and one engine
build later, through a different instrument (token sequences, not a 311 MB logit dump). Side (ii)
is the half that matters: an instrument that reported the two kernels identical would have been
measuring nothing, and this one discriminates by 119 and 105 tokens.

## 2. `G-E59c` — speed, interleaved in one sweep

`donor_engine_e53.exe`, `--threads 6 --bench 160`, k = 9 reps, a discarded warm-up per arm, arms
interleaved (E13 §6: absolutes carry ±5% between sweeps, ratios do not).

**`qwen25-05b_tqh`**

| arm | median | p25 | bootstrap 95% | IQR | foreign | ÷ PACKED |
|---|---|---|---|---|---|---|
| `PACKED` | 85.32 | 84.43 | [84.43, 86.37] | 1.35% | 3.08% | 1.000 |
| **`LUTBLK`** | **106.88** | 106.18 | [105.70, 107.71] | 0.95% | 3.91% | **1.253** |
| `LUTBLK32` | 105.44 | 104.97 | [104.55, 106.12] | 0.98% | 3.56% | 1.236 |

**`qwen25-15b_tqh`**

| arm | median | p25 | bootstrap 95% | IQR | foreign | ÷ PACKED |
|---|---|---|---|---|---|---|
| `PACKED` | 28.87 | 28.82 | [28.63, 29.41] | 1.14% | 3.60% | 1.000 |
| **`LUTBLK`** | **38.48** | 38.26 | [34.56, 39.20] | 2.21% | **7.14%** | **1.333** |
| `LUTBLK32` | 38.38 | 37.87 | [36.72, 38.69] | 2.11% | 5.42% | 1.329 |

Both ratios **SEPARATED** — the bootstrap intervals do not overlap at either scale.

**Two contamination notes, reported rather than cleaned.** The `15b_tqh` `LUTBLK` cells carry
**7.14%** foreign occupancy against `OCC_BAR = 4.39` (E52), and `LUTBLK32` **5.42%**; the packed
control sat at 3.60%. **The contamination is on the treatment, not the control, so it biases the
ratio DOWN**: at E52's `k = 0.262%` per point, the 3.54-point excess is worth ~0.93%, putting the
corrected ratio near **×1.345**. The wide `LUTBLK` interval `[34.56, 39.20]` comes from the same
cells and is why the *separation*, not the width, is what the gate reads.

**This reproduces E13 §5 closely, on different weights:** E13 measured ×1.217 at 0.5 B
(`--bench 300`) and ×1.358 on Coder-7B (`--bench 100`); E59 measures **×1.253 at 0.5 B and ×1.333
at 1.5 B** on trained artifacts at `--bench 160`, monotone in size the same way. **E58 addendum A's
correction is confirmed by measurement**: the packed kernel's ×1.11–1.17 was never "the engine".

## 3. `G-E59b` — fidelity, and the two columns tell opposite stories

| artifact | arm | **vs `PACKED`, same bytes** | vs HuggingFace | first divergence vs packed |
|---|---|---|---|---|
| `05b_tqh` | `PACKED` | 160/160 | 3/160 *(E57 says 3 — REPRODUCED)* | — |
| | `LUT` | **41/160 — 25.6%** | 3/160 | [0, 2, 0, 0, 0] |
| | **`LUTBLK`** | **41/160 — 25.6%** | 3/160 | [0, 2, 0, 0, 0] |
| | **`LUTBLK32`** | **67/160 — 41.9%** | 4/160 | [1, 5, −1, 0, 0] |
| `15b_tqh` | `PACKED` | 160/160 | 10/160 *(E57 says 10 — REPRODUCED)* | — |
| | `LUT` | **55/160 — 34.4%** | 5/160 | [0, 0, 0, 3, 0] |
| | **`LUTBLK`** | **55/160 — 34.4%** | 5/160 | [0, 0, 0, 3, 0] |
| | **`LUTBLK32`** | **92/160 — 57.5%** | 10/160 | [0, −1, 1, 1, 1] |

**The kernel changes 74.4% and 65.6% of tokens at whole-vector scaling, and 58.1% and 42.5% at
group-32.** `--lutblk` diverges at **token 0** on **8 of its 10 prompts**. This is not drift that
accumulates; it is a different model from the first step.

**`--lut-group 32` is a Pareto point nobody had measured.** It costs **1.4% of speed at 0.5 B and
0.3% at 1.5 B** — inside or beside the dispersion — and buys **+16.3 and +23.1 points of
agreement** (25.6→41.9%, 34.4→57.5%). E11 had the activation-error half of this (1.40e-01 rel-L2
whole-vector against 3.10e-02 at G=32, a 4.5× reduction) and E14 had the BPB half; **neither had
the token-level consequence with the speed beside it.** If this kernel is ever used, it is used at
group-32, and that is now a measured statement rather than a preference.

## 4. The finding that outranks both columns: **a gate on its FLOOR cannot show damage**

`G-E59b` was registered to report the HuggingFace column and **forbid it the verdict**, because
E57 had already measured packed at 3/160 and 10/160. Run the counterfactual on the numbers above:

| if the verdict had been read from `vs HF` | reading |
|---|---|
| `05b_tqh`: 3/160 → `LUTBLK` 3/160, `LUTBLK32` **4/160** | *"the kernel costs nothing, and group-32 is an improvement"* |
| `15b_tqh`: 10/160 → `LUTBLK` 5/160, `LUTBLK32` **10/160** | *"group-32 is free"* |

**Both readings are false.** The same arms change **74% and 66%** of the tokens the engine
actually emits, and group-32 — the arm both readings call free or better — changes **58% and
43%**. The HF counter cannot move because the artifact is already wrong almost everywhere: there is
no room below 3/160 for a kernel to push it.

This is `feedback_gate_is_not_a_progress_meter` **inverted, and the inverse had never been
written down**. That memory says a counter near its *ceiling* cannot show progress — the trap that
closed H0 early on `tf` 115/160. E59 is the mirror: **a counter near its floor cannot show
damage**, and it is the more dangerous half, because the failing direction is the one a gate exists
to catch. The fix is the same shape in both cases and is registered below.

## 5. Predictions, scored — 4.5 of 6, and the miss is in the costly direction

| # | registered prediction | outcome |
|---|---|---|
| 1 | `G-E59a`(i) `lut ≡ lutblk`, 160/160 both artifacts | **HELD**, exactly |
| 2 | `G-E59a`(ii) differs, within the first 4 tokens | **HELD** — first divergence 0, 2, 0, 0, 0 / 0, 0, 0, 3, 0 |
| 3 | speed ratio `05b_tqh` ×1.20–1.40 | **HELD** — ×1.253 |
| 4 | speed ratio `15b_tqh` ×1.25–1.50 | **HELD** — ×1.333 |
| 5 | fidelity whole-vector **35–60%** | **WRONG** — 25.6% and 34.4%, **both below the band** |
| 6 | fidelity group-32 **55–80%** | **HALF** — 57.5% at 1.5 B (inside), **41.9% at 0.5 B (below)** |

I anchored 5 and 6 on E14's measured 45.6% / 64.4% and the kernel came in **worse on every cell**.
E14's numbers were agreement against *fp32 activations on the same LUT path*; E59's are against
*the packed path*, which differs in the accumulate tree as well — a narrower comparison would have
predicted a smaller gap, and I predicted a smaller gap for the wrong reason and still overshot.

**The claim I registered as the decision point holds and is not close.** I wrote: *"if group-32
agreement comes back above 90% I am wrong, the kernel is close to free."* It came back at
**41.9% and 57.5%.**

## 6. What this means for the goal

**For a broken artifact the ×1.33 is free, and that is worth nothing.** `15b_tqh` is already
10/160; making it 5/160 at 38.48 tok/s is a faster wrong answer.

**For a faithful artifact the lever is currently unusable**, so **E58's packed ceiling stands for
the case that matters** — reached by a different road than E58 took, and for a reason E58 did not
know: not because there is no faster kernel, but because the faster kernel is not serving the same
model.

**And that leaves one specific, cheap, unexplored move, which is the real output of E59.** Every
number above measures int8 activations applied **post hoc** to a model that never saw them. The
programme's own repeated finding is that **post-hoc conversion breaks what training absorbs**:
E1/E17/E57 on ternary weights (3/160 post-hoc), H0 on the same organs trained (97.9% of the damage
removed in half a schedule), E18 §8 *"the model must be trained into the format rather than
converted into it."* **The activation quantiser is a format, and nothing has ever been trained into
it.**

> **`H2T` should heal against the ACTIVATION quantiser as well as the weight quantiser.**
> `s1/h0_qat.py` already carries STE through the shipped weight quantizer; adding a fake-quant of
> the layer inputs matching `quant_i8` at `--lut-group 32` (AQ=63, per-group amax) is a small
> change to an existing trainer and **costs no extra GPU hours** — it changes what the same
> sessions optimise, not how many there are.

If that works, a healed 1.5 B is served at the **measured 38.48 tok/s — 77% of the good bar** —
instead of 28.87. If it does not, the honest engine-side number for a faithful arm is E58's, and
the 50 tok/s bar is entirely a weights-streamed problem. **Either way the arithmetic that decides
the goal is unchanged**: at 1.5 B, 50 tok/s wants 33.7% fewer streamed weights, and no kernel
supplies that.

## 7. `G-E59d` — nothing is promoted

No flag became a default; `donor_engine.c` is untouched; `packed` remains the shipped path, which
is what E13 §6 and `donor_engine.c:190` already required. Per E13 §7's law, **neither ×1.253 nor
×1.333 may be quoted without the token count beside it**, and both appear together everywhere they
appear in this repository.

## 8. The rule this adds

> **A fidelity gate must be measured against a reference the treatment can still move.** Before
> registering one, state the control arm's current reading and ask *how far can this counter travel
> in the failing direction?* If the answer is "a few counts", the gate is on its floor and will
> return `PASS` for a treatment that destroys the model. Pair it with, or replace it by, a
> **paired** comparison against the untreated arm on the same inputs, whose range is the full 0–100%.

Filed against `feedback_gate_is_not_a_progress_meter.md` as its second face, not as a new memory:
ceiling and floor are the same defect and belong in one place.

## 9. Checked, not assumed

* **`probes/E13_BLOCKED_TILE_MAJOR.md`** — opened; §5's engine table (0.5 B 75.03→91.31 = ×1.217;
  Coder-7B 6.78→9.21 = ×1.358), `G-M0`'s sha256 identity between `--lut` and `--lutblk`, §6's
  explicit refusal to ship on the strength of it, §7's "not the same kind of number" law, §8 item 2
  naming exactly the measurement E59 takes.
* **`probes/E14_ACTIVATION_INT8_COST.md`** — opened; `CHEAP-BUT-NOT-NEUTRAL`, dBPB −0.016961 /
  −0.005991 *better*, greedy agreement **45.6% / 64.4%** against fp32 activations, and §0's
  statement that its own registered instrument set could not tell band from model. E59's
  predictions 5 and 6 were anchored on those two percentages and both overshot.
* **`probes/E11_LUT_KERNEL_CEILING.md`** — opened; `NO-LIFT` on the shipped layout (0.391 at the
  512 MB verdict cell), activation error **1.40e-01** whole-vector and **3.10e-02** at group-32.
* **`probes/E58_…` + addendum A** — the packed rates E59's control column reproduces, and the
  correction E59 now confirms by measurement.
* **E57 / E17's stored counts** — 3/160 and 10/160, reproduced by E59's own packed arm
  (printed as a control on every row), which is the third independent reproduction of those two
  numbers.
* **`donor_engine.c:1552`** — `--lut` requires a packed model, so the fp32 arms are structurally
  excluded from this kernel at any price.
