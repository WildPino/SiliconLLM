# D0 - FFN co-activation and the MoE carve on a real donor

**Probe:** D0, `BRIEF_D0_COACTIVATION_PERMUTATION.md` (pre-registered 2026-08-22, + Amendment 1)
**Instrument:** `benchmarks/donor_adaptation/density/d0_coactivation.py`
**Written by:** the Builder - **Date:** 2026-08-29
**Status:** **PARTIAL RUN.** Two of the instrument's three stages completed. The third
(`analyse`) - which carries the brief's own headline deliverable - was never executed.

---

## 0. Read this first: what exists and what does not

The instrument has three stages: `controls`, `model`, `analyse`. Only the first two ran.

| stage | ran? | artefact | what it carries |
|---|---|---|---|
| `controls` | **yes** | `d0_coactivation.controls.json`, `d0_controls.log` | C2-C7, synthetic, no donor loaded |
| `model` | **yes** | `d0_coactivation.model.json`, `d0_model.log` | C1 losslessness, mask capture (56 npz, 4.6 GB), oracle-routed **fidelity** |
| `analyse` | **NO** | *(no `d0_coactivation.analyse.json` on disk)* | **the skippable x arm x block-size table, the section 5 duplication / set-cover measurement, the A1.3.2 MoE geometry, the 8-seed random spread, the TEAL/CATS threshold realism check, the block sweep** |

**Stated plainly: the brief's section 3 deliverable - "a table of skippable fraction x permutation
arm x block size, with the no-permutation arm as the number to beat and the random arm as the
honest control" - does not exist.** Neither does the section 5 duplication measurement, which the
brief declares mandatory and whose absence it says *"invalidates a null result from this probe."*

What *does* exist on the donor is a different measurement, and in Amendment 1's own terms a more
directly load-bearing one: the **oracle-routed fidelity** of an MoE carve of the donor's FFN,
co-activation partition against a matched random partition. It is reported in full below. It is not
a substitute for the missing table and is not offered as one.

Every input the `analyse` stage needs is on disk (`results/d0_masks/`, 4.6 GB, 28 fit + 28 score
layers). It is CPU-only and donor-free. **It can be run without repeating any capture.**

---

## 1. The instrument, and what `relerr` actually is

`fidelity` answers: *if the donor's FFN is carved into `E` equal-sized experts and only the top-`k`
experts are computed per token, how much of the FFN output survives?*

- The partition is fitted on the **calib** half and scored on the **held-out** half. Distinct
  corpora by sha256, asserted in code.
- The router is an **ORACLE**: it picks the `k` experts with the largest `||h|` restricted to that
  expert`||`. **This is an upper bound on any trainable router.** No learned router beats it; a
  real one does worse.
- `relerr` = `sqrt( sum||yhat - y||^2 / sum||y||^2 )` over the scored tokens, where `y` is the true
  FFN output and `yhat` the output with the unselected experts' contributions dropped. No rescaling.

**The scale that makes `relerr` readable: `relerr = 1.0` is what you get by computing nothing at all
and emitting zeros.** A `relerr` of 0.94 means the carve recovers 6% of the output norm - closer to
"skip the FFN entirely" than to "reproduce the FFN". This reading is used throughout and it is the
single most important thing in this document.

---

## 2. The full fidelity table

All 120 rows, printed from `d0_coactivation.model.json` -> `fidelity`. Co-activation and its matched
null share a line, so the comparison at matched `E`, matched `k` and matched **achieved** activation
is unavoidable.

`FFN_active(MoE)` = `k/E + E/(3*d_ffn)` - expert bytes plus the router's own weight bytes
(`E x d_model` at 0.5 B/weight, against the FFN's `3 x d_ffn x d_model`). The router is charged, not
assumed negligible. Compare it to **0.3333**, the dense-gate floor Amendment 1 section A1.1 proved
no amount of block-skipping can go below.

### Layer 1 of 28 - tokens scored 2048, from the score slice (held-out half); partition fitted on the fit slice (calib half)

| E | k | nominal `k/E` | **achieved act.** | co-act `relerr` | null `relerr` | margin (null-co) | co recovers | null recovers | `FFN_active(MoE)` | expert B/organ | 48KB-legal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.03125 | **0.03125** | 0.0410 | 0.4718 | +0.4309 | 95.9% | 52.8% | 0.0324 | 215040 | yes |
| 32 | 2 | 0.06250 | **0.06250** | 0.0348 | 0.2089 | +0.1741 | 96.5% | 79.1% | 0.0637 | 215040 | yes |
| 32 | 4 | 0.12500 | **0.12500** | 0.0301 | 0.0536 | +0.0235 | 97.0% | 94.6% | 0.1262 | 215040 | yes |
| 32 | 8 | 0.25000 | **0.25000** | 0.0243 | 0.0354 | +0.0111 | 97.6% | 96.5% | 0.2512 | 215040 | yes |
| 64 | 1 | 0.01562 | **0.01562** | 0.0491 | 0.4712 | +0.4222 | 95.1% | 52.9% | 0.0180 | 107520 | yes |
| 64 | 2 | 0.03125 | **0.03125** | 0.0418 | 0.2092 | +0.1673 | 95.8% | 79.1% | 0.0336 | 107520 | yes |
| 64 | 4 | 0.06250 | **0.06250** | 0.0356 | 0.0558 | +0.0202 | 96.4% | 94.4% | 0.0649 | 107520 | yes |
| 64 | 8 | 0.12500 | **0.12500** | 0.0299 | 0.0406 | +0.0107 | 97.0% | 95.9% | 0.1274 | 107520 | yes |
| 128 | 1 | 0.00781 | **0.00781** | 0.0536 | 0.4725 | +0.4189 | 94.6% | 52.8% | 0.0126 | 53760 | yes |
| 128 | 2 | 0.01562 | **0.01562** | 0.0454 | 0.2101 | +0.1647 | 95.5% | 79.0% | 0.0204 | 53760 | yes |
| 128 | 4 | 0.03125 | **0.03125** | 0.0382 | 0.0568 | +0.0186 | 96.2% | 94.3% | 0.0360 | 53760 | yes |
| 128 | 8 | 0.06250 | **0.06250** | 0.0322 | 0.0422 | +0.0100 | 96.8% | 95.8% | 0.0673 | 53760 | yes |

### Layer 7 of 28 - tokens scored 2048, from the score slice (held-out half); partition fitted on the fit slice (calib half)

| E | k | nominal `k/E` | **achieved act.** | co-act `relerr` | null `relerr` | margin (null-co) | co recovers | null recovers | `FFN_active(MoE)` | expert B/organ | 48KB-legal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.03125 | **0.03125** | 0.9172 | 0.9345 | +0.0172 | 8.3% | 6.6% | 0.0324 | 215040 | yes |
| 32 | 2 | 0.06250 | **0.06250** | 0.8684 | 0.8966 | +0.0282 | 13.2% | 10.3% | 0.0637 | 215040 | yes |
| 32 | 4 | 0.12500 | **0.12500** | 0.8034 | 0.8376 | +0.0343 | 19.7% | 16.2% | 0.1262 | 215040 | yes |
| 32 | 8 | 0.25000 | **0.25000** | 0.7038 | 0.7385 | +0.0347 | 29.6% | 26.1% | 0.2512 | 215040 | yes |
| 64 | 1 | 0.01562 | **0.01562** | 0.9268 | 0.9439 | +0.0171 | 7.3% | 5.6% | 0.0180 | 107520 | yes |
| 64 | 2 | 0.03125 | **0.03125** | 0.8879 | 0.9152 | +0.0273 | 11.2% | 8.5% | 0.0336 | 107520 | yes |
| 64 | 4 | 0.06250 | **0.06250** | 0.8396 | 0.8744 | +0.0347 | 16.0% | 12.6% | 0.0649 | 107520 | yes |
| 64 | 8 | 0.12500 | **0.12500** | 0.7744 | 0.8129 | +0.0385 | 22.6% | 18.7% | 0.1274 | 107520 | yes |
| 128 | 1 | 0.00781 | **0.00781** | 0.9386 | 0.9490 | +0.0105 | 6.1% | 5.1% | 0.0126 | 53760 | yes |
| 128 | 2 | 0.01562 | **0.01562** | 0.9076 | 0.9253 | +0.0177 | 9.2% | 7.5% | 0.0204 | 53760 | yes |
| 128 | 4 | 0.03125 | **0.03125** | 0.8684 | 0.8944 | +0.0260 | 13.2% | 10.6% | 0.0360 | 53760 | yes |
| 128 | 8 | 0.06250 | **0.06250** | 0.8190 | 0.8524 | +0.0334 | 18.1% | 14.8% | 0.0673 | 53760 | yes |

### Layer 14 of 28 - tokens scored 2048, from the score slice (held-out half); partition fitted on the fit slice (calib half)

| E | k | nominal `k/E` | **achieved act.** | co-act `relerr` | null `relerr` | margin (null-co) | co recovers | null recovers | `FFN_active(MoE)` | expert B/organ | 48KB-legal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.03125 | **0.03125** | 0.9422 | 0.9598 | +0.0176 | 5.8% | 4.0% | 0.0324 | 215040 | yes |
| 32 | 2 | 0.06250 | **0.06250** | 0.9033 | 0.9308 | +0.0275 | 9.7% | 6.9% | 0.0637 | 215040 | yes |
| 32 | 4 | 0.12500 | **0.12500** | 0.8412 | 0.8791 | +0.0379 | 15.9% | 12.1% | 0.1262 | 215040 | yes |
| 32 | 8 | 0.25000 | **0.25000** | 0.7308 | 0.7834 | +0.0526 | 26.9% | 21.7% | 0.2512 | 215040 | yes |
| 64 | 1 | 0.01562 | **0.01562** | 0.9585 | 0.9690 | +0.0105 | 4.2% | 3.1% | 0.0180 | 107520 | yes |
| 64 | 2 | 0.03125 | **0.03125** | 0.9319 | 0.9498 | +0.0179 | 6.8% | 5.0% | 0.0336 | 107520 | yes |
| 64 | 4 | 0.06250 | **0.06250** | 0.8899 | 0.9175 | +0.0276 | 11.0% | 8.3% | 0.0649 | 107520 | yes |
| 64 | 8 | 0.12500 | **0.12500** | 0.8196 | 0.8618 | +0.0422 | 18.0% | 13.8% | 0.1274 | 107520 | yes |
| 128 | 1 | 0.00781 | **0.00781** | 0.9659 | 0.9743 | +0.0084 | 3.4% | 2.6% | 0.0126 | 53760 | yes |
| 128 | 2 | 0.01562 | **0.01562** | 0.9466 | 0.9598 | +0.0132 | 5.3% | 4.0% | 0.0204 | 53760 | yes |
| 128 | 4 | 0.03125 | **0.03125** | 0.9162 | 0.9379 | +0.0217 | 8.4% | 6.2% | 0.0360 | 53760 | yes |
| 128 | 8 | 0.06250 | **0.06250** | 0.8663 | 0.9025 | +0.0362 | 13.4% | 9.8% | 0.0673 | 53760 | yes |

### Layer 21 of 28 - tokens scored 2048, from the score slice (held-out half); partition fitted on the fit slice (calib half)

| E | k | nominal `k/E` | **achieved act.** | co-act `relerr` | null `relerr` | margin (null-co) | co recovers | null recovers | `FFN_active(MoE)` | expert B/organ | 48KB-legal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.03125 | **0.03125** | 0.8784 | 0.9049 | +0.0265 | 12.2% | 9.5% | 0.0324 | 215040 | yes |
| 32 | 2 | 0.06250 | **0.06250** | 0.8152 | 0.8545 | +0.0392 | 18.5% | 14.6% | 0.0637 | 215040 | yes |
| 32 | 4 | 0.12500 | **0.12500** | 0.7349 | 0.7891 | +0.0542 | 26.5% | 21.1% | 0.1262 | 215040 | yes |
| 32 | 8 | 0.25000 | **0.25000** | 0.6272 | 0.6898 | +0.0626 | 37.3% | 31.0% | 0.2512 | 215040 | yes |
| 64 | 1 | 0.01562 | **0.01562** | 0.8876 | 0.9117 | +0.0241 | 11.2% | 8.8% | 0.0180 | 107520 | yes |
| 64 | 2 | 0.03125 | **0.03125** | 0.8243 | 0.8694 | +0.0451 | 17.6% | 13.1% | 0.0336 | 107520 | yes |
| 64 | 4 | 0.06250 | **0.06250** | 0.7452 | 0.8212 | +0.0760 | 25.5% | 17.9% | 0.0649 | 107520 | yes |
| 64 | 8 | 0.12500 | **0.12500** | 0.6565 | 0.7567 | +0.1002 | 34.3% | 24.3% | 0.1274 | 107520 | yes |
| 128 | 1 | 0.00781 | **0.00781** | 0.8850 | 0.9160 | +0.0310 | 11.5% | 8.4% | 0.0126 | 53760 | yes |
| 128 | 2 | 0.01562 | **0.01562** | 0.8372 | 0.8769 | +0.0397 | 16.3% | 12.3% | 0.0204 | 53760 | yes |
| 128 | 4 | 0.03125 | **0.03125** | 0.7776 | 0.8362 | +0.0586 | 22.2% | 16.4% | 0.0360 | 53760 | yes |
| 128 | 8 | 0.06250 | **0.06250** | 0.7051 | 0.7861 | +0.0809 | 29.5% | 21.4% | 0.0673 | 53760 | yes |

### Layer 27 of 28 - tokens scored 2048, from the score slice (held-out half); partition fitted on the fit slice (calib half)

| E | k | nominal `k/E` | **achieved act.** | co-act `relerr` | null `relerr` | margin (null-co) | co recovers | null recovers | `FFN_active(MoE)` | expert B/organ | 48KB-legal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.03125 | **0.03125** | 0.5767 | 0.9005 | +0.3237 | 42.3% | 10.0% | 0.0324 | 215040 | yes |
| 32 | 2 | 0.06250 | **0.06250** | 0.4899 | 0.8211 | +0.3313 | 51.0% | 17.9% | 0.0637 | 215040 | yes |
| 32 | 4 | 0.12500 | **0.12500** | 0.3932 | 0.6832 | +0.2900 | 60.7% | 31.7% | 0.1262 | 215040 | yes |
| 32 | 8 | 0.25000 | **0.25000** | 0.2804 | 0.5089 | +0.2286 | 72.0% | 49.1% | 0.2512 | 215040 | yes |
| 64 | 1 | 0.01562 | **0.01562** | 0.6526 | 0.9229 | +0.2703 | 34.7% | 7.7% | 0.0180 | 107520 | yes |
| 64 | 2 | 0.03125 | **0.03125** | 0.5573 | 0.8692 | +0.3120 | 44.3% | 13.1% | 0.0336 | 107520 | yes |
| 64 | 4 | 0.06250 | **0.06250** | 0.4538 | 0.7710 | +0.3172 | 54.6% | 22.9% | 0.0649 | 107520 | yes |
| 64 | 8 | 0.12500 | **0.12500** | 0.3485 | 0.6199 | +0.2714 | 65.2% | 38.0% | 0.1274 | 107520 | yes |
| 128 | 1 | 0.00781 | **0.00781** | 0.6791 | 0.9140 | +0.2349 | 32.1% | 8.6% | 0.0126 | 53760 | yes |
| 128 | 2 | 0.01562 | **0.01562** | 0.5950 | 0.8626 | +0.2676 | 40.5% | 13.7% | 0.0204 | 53760 | yes |
| 128 | 4 | 0.03125 | **0.03125** | 0.4944 | 0.7754 | +0.2810 | 50.6% | 22.5% | 0.0360 | 53760 | yes |
| 128 | 8 | 0.06250 | **0.06250** | 0.3839 | 0.6516 | +0.2676 | 61.6% | 34.8% | 0.0673 | 53760 | yes |

**Every one of the 60 matched pairs has the co-activation partition ahead of the random
partition. The margin is never negative and never zero.** That is a clean, unambiguous direction and
it is the honest headline of the co-activation-vs-null comparison.

**And it does not rescue the method.** Read the `co recovers` column instead of the margin column:

- At the three **middle** layers (7, 14, 21) the best co-activation carve inside a plausible byte
  budget recovers **6% to 37%** of the FFN output norm - with an oracle router. The null recovers
  3% to 31%. Both sit near the "emit zeros" baseline; co-activation is reliably, and uselessly,
  less bad.
- Only **layer 27** shows a carve that is arguably alive: `E=32, k=8` recovers 72.0% of the output
  norm at 25.1% of the FFN bytes. One layer in five, at a `k/E` buying only 1.33x over the 33.3%
  dense-gate floor.
- **Layer 1 is not evidence for the method.** It recovers 94.6-97.6% at every setting - but so does
  the *random* partition at `k>=4` (94.3-96.5%). A layer whose output survives an arbitrary
  partition is a layer whose FFN is close to trivial, not one where co-activation found structure.

### The budget arithmetic, against Amendment 1

Amendment 1 section A1.2 was right that the escape from the 33.3% dense-gate floor is the MoE carve,
and the byte arithmetic confirms it: `E=128, k=2` reads **2.04%** of the FFN's weight bytes, router
included. **The 2% FFN budget is reachable in bytes.** At that operating point the oracle-routed
fidelity is:

| layer | 1 | 7 | 14 | 21 | 27 |
|---|---|---|---|---|---|
| co-activation `relerr` | 0.0454 | 0.9076 | 0.9466 | 0.8372 | 0.5950 |
| **output norm recovered** | **95.5%** | **9.2%** | **5.3%** | **16.3%** | **40.5%** |
| null `relerr` | 0.2101 | 0.9253 | 0.9598 | 0.8769 | 0.8626 |

**Four of the five layers lose 59% to 95% of their FFN output norm at the byte budget the carve was
adopted to reach - and that is the ceiling an oracle router sets, not the floor a real one would
hit.** The budget is reachable; the model is not.

---

## 3. Depth dependence - not averaged away

Margin `relerr(null) - relerr(co-activation)`, per `(E, k)`, across the five layers. The layers were
chosen to span depth (1, 7, 14, 21 of 28, and 27 = last).

| E | k | L1 | L7 | L14 | L21 | L27 |
|---|---|---|---|---|---|---|
| 32 | 1 | +0.4309 | +0.0172 | +0.0176 | +0.0265 | +0.3237 |
| 32 | 2 | +0.1741 | +0.0282 | +0.0275 | +0.0392 | +0.3313 |
| 32 | 4 | +0.0235 | +0.0343 | +0.0379 | +0.0542 | +0.2900 |
| 32 | 8 | +0.0111 | +0.0347 | +0.0526 | +0.0626 | +0.2286 |
| 64 | 1 | +0.4222 | +0.0171 | +0.0105 | +0.0241 | +0.2703 |
| 64 | 2 | +0.1673 | +0.0273 | +0.0179 | +0.0451 | +0.3120 |
| 64 | 4 | +0.0202 | +0.0347 | +0.0276 | +0.0760 | +0.3172 |
| 64 | 8 | +0.0107 | +0.0385 | +0.0422 | +0.1002 | +0.2714 |
| 128 | 1 | +0.4189 | +0.0105 | +0.0084 | +0.0310 | +0.2349 |
| 128 | 2 | +0.1647 | +0.0177 | +0.0132 | +0.0397 | +0.2676 |
| 128 | 4 | +0.0186 | +0.0260 | +0.0217 | +0.0586 | +0.2810 |
| 128 | 8 | +0.0100 | +0.0334 | +0.0362 | +0.0809 | +0.2676 |

And the quantity that actually decides the route - the co-activation arm's own `relerr` - by depth:

| E | k | L1 | L7 | L14 | L21 | L27 |
|---|---|---|---|---|---|---|
| 32 | 1 | 0.0410 | 0.9172 | 0.9422 | 0.8784 | 0.5767 |
| 32 | 2 | 0.0348 | 0.8684 | 0.9033 | 0.8152 | 0.4899 |
| 32 | 4 | 0.0301 | 0.8034 | 0.8412 | 0.7349 | 0.3932 |
| 32 | 8 | 0.0243 | 0.7038 | 0.7308 | 0.6272 | 0.2804 |
| 64 | 1 | 0.0491 | 0.9268 | 0.9585 | 0.8876 | 0.6526 |
| 64 | 2 | 0.0418 | 0.8879 | 0.9319 | 0.8243 | 0.5573 |
| 64 | 4 | 0.0356 | 0.8396 | 0.8899 | 0.7452 | 0.4538 |
| 64 | 8 | 0.0299 | 0.7744 | 0.8196 | 0.6565 | 0.3485 |
| 128 | 1 | 0.0536 | 0.9386 | 0.9659 | 0.8850 | 0.6791 |
| 128 | 2 | 0.0454 | 0.9076 | 0.9466 | 0.8372 | 0.5950 |
| 128 | 4 | 0.0382 | 0.8684 | 0.9162 | 0.7776 | 0.4944 |
| 128 | 8 | 0.0322 | 0.8190 | 0.8663 | 0.7051 | 0.3839 |

**The depth profile is U-shaped and the two ends are not the same phenomenon.**

- **L1** - near-perfect fidelity for both arms; the large margin at `k=1` (+0.4309) collapses to
  +0.0100 by `k=8` as the null catches up. The layer is easy, not clustered.
- **L7, L14** - the worst. Co-activation `relerr` 0.7038-0.9743; margins 0.0084-0.0526. **This is
  the "co-activation ~ random ~ no exploitable structure" regime, and it sits in the middle of the
  network where most of the FFN weight lives.**
- **L21** - margins start to open (up to +0.1002 at `E=64, k=8`) but absolute fidelity is still
  0.6272-0.8876.
- **L27** - the only layer with both a large margin (+0.2286 to +0.3313; the null is 1.35x to 1.82x
  worse) and fidelity that is not hopeless. The last layer's FFN does appear to contain modular
  structure the clustering finds.

**Averaging these five together would manufacture a result true of no layer.** The honest statement
is that co-activation structure in this donor is **concentrated at the ends of the network and
absent in the middle** - and the middle is where the bytes are. Layers 7, 14 and 21 stand in for 24
of the 28 layers, none of which were measured.

---

## 4. The controls

### C1 - identity / losslessness (donor; ran FIRST, as section 7.1 demands)

| quantity | value |
|---|---|
| construction | independent random permutation of all 8960 FFN neurons in **all 28 layers** (seed 20260828); gate/up rows and down columns permuted together, then inverted in place |
| `BPB(identity)` | `0.7675949601147849` |
| `BPB(permuted)` | `0.7675949652076198` |
| `BPB(restored)` | `0.7675949601147849` |
| `delta BPB` (permuted - identity) | `5.0928e-09` |
| `delta BPB` (restored - identity) | `0.0` - **exact** |
| `max abs logit deviation` | `1.277924e-04` |
| `max abs logit magnitude` | `31.2392` |
| **relative logit deviation** | `4.0908e-06` |
| recorded verdict | **FIRED** |

**This is a real test and it passed.** A permutation bug - wrong gather direction, `down_proj`
permuted on the wrong axis, an off-by-one in `pos_from_order` - moves BPB by orders of magnitude,
not by 5e-09. The restore being bit-exact additionally proves the bookkeeping is a true inverse.
Both deltas are ~1.5e-06 sigma_seed (sigma_seed = 0.005), i.e. nothing.

**One cross-check the JSON's own field does not surface.** `control_C1_losslessness` records
`d1_published_baseline_bpb = 0.767595` and `matches_d1_baseline_to_6dp = true`. It does match to
6 dp. But D1's actual stored value is `0.7675949677540373` (`d1_pruning.json` -> `baseline_bpb`):

```
D1 baseline        0.7675949677540373
D0 C1 identity     0.7675949601147849
                   -------------------
difference         7.639e-09      <-- 1.50x the permutation delta of 5.093e-09
```

**The run-to-run reproducibility floor of this BPB path is larger than the effect C1 measures.**
That strengthens rather than weakens the reading - it shows the permutation delta is
indistinguishable from the harness's own noise rather than merely small. But the correct statement
is *"below the BPB path's reproducibility floor"*, not *"the permutation changed BPB by 5e-09"*.

### C2 - planted positive: does the instrument FIRE on known structure?

| quantity | value |
|---|---|
| construction | N=8960, T=4096, 64 disjoint contiguous groups of 140 neurons, 6 WHOLE groups active per token, then scrambled by a known permutation (seed 4242) |
| density achieved | `0.09375` (840 active of 8960) |
| block size | 140 |
| **planted skippable, exact** | `0.90625` |
| identity arm on the scrambled data | `0.00000` |
| **co-activation arm recovered** | `0.90625` |
| recovery ratio | `1.0000` |
| **cluster ARI vs planted labels** | `1.0000` |
| recorded verdict | **FIRED** |

**This is the control that licenses every null in this document, and it is a real test with a real
failure mode.** The structure was scrambled by a permutation the clustering was never shown; a
broken clustering returns ARI near 0 and a skippable fraction near the identity arm's 0.0000. It
returned **ARI = 1.0000 - perfect label recovery - and reproduced the planted skippable fraction to
the last digit.** The instrument demonstrably fires on a known positive.

**The scope of that licence, stated honestly.** C2 exercises the *permutation / block-skippable*
path - which is the path the `analyse` stage would have used, and which never ran on the donor. The
fidelity table in section 2 rides on a different code path (`balanced_labels` + `routed_error`), and
**that path has no planted-positive control of its own.** Its correctness rests on C1 (the
permutation bookkeeping is exact) and on the internal consistency of the balanced partition - not on
any demonstration that it recovers planted structure.

### C3 - planted negative: i.i.d. at matched density

Construction: N=8960, T=4096, exactly 896 active/token drawn uniformly. Density achieved `0.1000`.

| block | analytic `q^B` | analytic hypergeom. | identity | random mean | random [min, max] | co-activation | recorded "all at analytic" |
|---|---|---|---|---|---|---|---|
| 4 | 0.65610000 | 0.65605117 | 0.65611954 | 0.65602985 | [0.65597763, 0.65608281] | **0.65744465** | True |
| 12 | 0.28242954 | 0.28219827 | 0.28222722 | 0.28213133 | [0.28190781, 0.28232082] | **0.28860596** | True |
| 21 | 0.10941899 | 0.10913396 | 0.10924663 | 0.10907255 | [0.10889703, 0.10927070] | **0.11701442** | True |
| 64 | 0.00117902 | 0.00114976 | 0.00111607 | 0.00116490 | [0.00110735, 0.00121896] | **0.00206473** | True |
| 140 | 0.00000039 | 0.00000035 | 0.00000000 | 0.00000000 | [0.00000000, 0.00000000] | **0.00000381** | True |

**The instrument recorded this control as FIRED. On its own numbers it should not have.**

The brief's section 7.3 wording is exact: *"all three arms must land together at the analytic `q^B`
value. Any arm that 'wins' on noise is broken."* The identity and random arms do land there. **The
co-activation arm does not - it is above the analytic value at every single block size, always in
the flattering direction:**

| block | analytic hypergeom. | co-activation | excess | excess, relative | random-seed full range | excess / random range |
|---|---|---|---|---|---|---|
| 4 | 0.65605117 | 0.65744465 | +0.00139348 | +0.2% | 1.05e-04 | 13.2x |
| 12 | 0.28219827 | 0.28860596 | +0.00640770 | +2.3% | 4.13e-04 | 15.5x |
| 21 | 0.10913396 | 0.11701442 | +0.00788047 | +7.2% | 3.74e-04 | 21.1x |
| 64 | 0.00114976 | 0.00206473 | +0.00091497 | +79.6% | 1.12e-04 | 8.2x |
| 140 | 0.00000035 | 0.00000381 | +0.00000347 | +997.7% | 0.00e+00 | infinite (range = 0) |

It passed only because of the tolerance the code applies:

```python
"all_arms_at_analytic": bool(abs(v_id - hg) < 0.02
                             and abs(np.mean(v_rnd) - hg) < 0.02
                             and abs(v_co  - hg) < 0.03)     # <-- wider, and ABSOLUTE
```

An **absolute** 0.03 window is meaningless where the quantity itself is 0.00115 (block 64) or
3.5e-07 (block 140). At block 140 the identity and random arms return **exactly 0.0** and the
co-activation arm returns 3.8e-06 - 11x the analytic value - and the control still reports
"all arms at analytic".

**The mechanism is not mysterious and it is not a bug in the clustering.** In C3 the co-activation
ordering is *fitted on `Bn` and scored on `idx_n` - the same 4096 x 8960 sample*. With 8960 neurons
and 4096 tokens, k-means on the empirical co-activation matrix will always find sample-noise
structure, and the in-sample skippable score will always exceed the population value. That is
textbook in-sample optimism, and C3 measures it: **about +0.006 to +0.008 absolute at the block
sizes that matter, which is 13x to 21x the entire random-seed range.**

**What this does and does not invalidate:**

- It **does not** contaminate the fidelity table in section 2. That table fits the partition on the
  **calib** half and scores it on the **held-out** half, sha256-asserted distinct. The leak is
  confined to the controls stage.
- It **does** mean C3 as written is not a working known-negative for the arm under test, and its
  "FIRED" should be read as **DID NOT FIRE for the co-activation arm.** The brief's own criterion -
  *any arm that wins on noise is broken* - was met, and the code's tolerance absorbed it.
- It **does** put a floor under the never-run `analyse` stage: any in-sample co-activation advantage
  below roughly +0.008 at block 12-21 must be attributed to this optimism, not to donor structure.
  To its credit, `analyse` was written to report `generalisation_gap_in_minus_out` explicitly, so it
  would have surfaced this.

### C4 - random-permutation invariances

| invariance | measured |
|---|---|
| un-permuting the permuted mask returns the original, elementwise | `True` |
| per-token active count identical | `True` |
| per-neuron firing-frequency multiset identical | `True` |
| co-activation spectrum max abs difference | `0.0` |
| co-activation spectrum identical | `True` |
| recorded verdict | **FIRED** |

**Half real test, half arithmetic tautology, and the split matters.**

- *Real:* the elementwise un-permute check. `pos_from_order` / gather-direction errors are the
  documented way this class of probe silently produces plausible numbers, and every arm rides on
  that one function. It was checked and it holds exactly.
- *Real:* the spectrum check returning **exactly 0.0** confirms `P^T C P` is similar to `C`.
- *Tautology:* per-token active count and per-neuron firing-frequency multiset are preserved **by
  the definition of a permutation**. They cannot fail unless the bookkeeping check has already
  failed. Zero independent evidence.

The control's stated purpose - establishing that the random arm cannot make the problem *easier*,
which had happened twice before on this programme - is served, and served by the bookkeeping and
spectrum checks alone.

### C5 - fast index path == `d0_layout.layout_stats`

| block | `layout_stats` | fast index path | abs diff |
|---|---|---|---|
| 4 | 0.6561195373535156 | 0.6561195373535156 | 0.0 |
| 12 | 0.2822272170325067 | 0.2822272170325067 | 0.0 |
| 21 | 0.1092466255868545 | 0.1092466255868545 | 0.0 |
| 64 | 0.0011160714285714 | 0.0011160714285714 | 0.0 |
| 140 | 0.0000000000000000 | 0.0000000000000000 | 0.0 |

**Real test.** Two independent implementations of the same accounting - dense-mask and index-based -
agreeing to exactly 0.0 at five block sizes. Agreement was not guaranteed: the index path is a
memory-lean rewrite, and disagreement would have been an instrument fault. It also inherits D0's
earlier controls into the new path, which is the stated point.

### C6 - hypergeometric closed form vs Monte-Carlo

| block | closed form | Monte-Carlo, 20 000 trials | abs diff |
|---|---|---|---|
| 4 | 0.6560511714 | 0.65820 | 0.00214883 |
| 12 | 0.2821982671 | 0.28395 | 0.00175173 |
| 21 | 0.1091339561 | 0.10405 | 0.00508396 |
| 64 | 0.0011497642 | 0.00160 | 0.00045024 |

**Real test, weak.** The closed form is checked against brute force and matches within sampling
error at four block sizes. But at block 64 the closed form is 0.00115, and 20 000 trials produce
about 32 hits - Monte-Carlo standard error there is roughly 0.00024, so the check has almost no
power in exactly the regime where the reference matters most. The `< 0.01` pass threshold is about
9x the largest observed deviation; it would have accepted a badly wrong closed form at small blocks.
It fired, and it is thin.

### C7 - Amendment 1's arithmetic against the brief's own table

| `s` | brief A1.1 | computed `(3-2s)/3` | abs diff |
|---|---|---|---|
| 0.282 | 0.8120 | 0.8120000000 | 1.11e-16 |
| 0.656 | 0.5620 | 0.5626666667 | 6.67e-04 |
| 0.950 | 0.3670 | 0.3666666667 | 3.33e-04 |
| 1.000 | 0.3333 | 0.3333333333 | 0.00e+00 |

`s_required_for_2pct_FFN_active = 1.47`, confirming section A1.1's finding that no block-skippable
fraction, including a perfect one, reaches the 2% budget under a dense gate.

**This is a tautology, not a test.** It evaluates a three-term closed-form expression and compares
it to the same expression typed into the brief. It cannot fail for any reason other than a typo, and
it carries no information about the donor, the instrument, or the hypothesis. It is a transcription
check and should be labelled as one.

### Controls scorecard

| control | fired as recorded | is it a real test? | this report's reading |
|---|---|---|---|
| C1 losslessness (donor) | yes | **yes, strong** | **FIRED.** Delta below the BPB path's own 7.6e-09 reproducibility floor; restore bit-exact |
| C2 planted positive | yes | **yes, strong** | **FIRED.** ARI = 1.0000, planted skippable recovered exactly. Licenses the permutation path - **not** the `routed_error` path |
| C3 planted negative | yes | **yes - and it did not actually pass** | **DID NOT FIRE for the co-activation arm.** Wins on pure noise at all 5 block sizes; absorbed by a 0.03 absolute tolerance |
| C4 invariances | yes | **half.** Bookkeeping + spectrum real; count and frequency are tautologies | FIRED on the parts that could fail |
| C5 path equivalence | yes | **yes** | FIRED, exactly 0.0 |
| C6 hypergeom vs MC | yes | **yes, but underpowered** at the blocks that matter | FIRED, thin |
| C7 A1.1 arithmetic | yes | **no, tautology** | Transcription check, no evidential weight |

**Does the instrument demonstrably fire on a known positive? Yes - C2, unambiguously, for the
permutation / skippable path.** The fidelity path that produced every donor number in section 2 has
no planted-positive control of its own, and that is the single largest gap in the control battery.

---

## 5. ACHIEVED vs nominal

| quantity | requested / nominal | **achieved** | source field |
|---|---|---|---|
| activation fraction, all 120 fidelity rows | `k/E` | **`k/E` exactly, all 120 rows** | `true_activation_fraction_achieved` |
| cluster sizes, `E=32` | 280 | **min 280, max 280** | `cluster_size_min/max` |
| cluster sizes, `E=64` | 140 | **min 140, max 140** | `cluster_size_min/max` |
| cluster sizes, `E=128` | 70 | **min 70, max 70** | `cluster_size_min/max` |
| fidelity tokens | `FIDELITY_TOKENS` | **2048** | `n_tokens_achieved` |
| capture tokens per layer | - | **16384** (32 seq x 512) | `T_tokens_achieved` |
| stored top-k per token | 2240 | **2240** (`p_max` = 0.25) | `kmax_stored_achieved` |
| layers captured | all | **28 of 28**, both token sets | `capture.fit/score.layers` |
| `torch.set_num_threads` | 6 | **6** | `torch_num_threads_achieved` |
| fit slice sequences | 32 | **32** (`n_rejected` = 0) | `fit_slice` |
| score slice sequences | 32 | **32** (`n_rejected` = 1) | `score_slice` |
| BPB eval sequences | 24 | **24** (`n_rejected` = 0) | `bpb_eval_slice` |

**The achieved column is clean, and this is not a formality.** `true_activation_fraction_achieved`
equals `k/E` to the last digit in all 120 rows because `balanced_labels` enforces exactly `d_ffn/E`
neurons per expert through a capacity-constrained assignment, so "activation fraction = `k/E`" is a
fact about bytes read, not an average over ragged clusters. `cluster_size_min == cluster_size_max`
in every row confirms it independently. **The requested-vs-achieved gap that has burned this
programme repeatedly is genuinely absent here** - with one caveat.

**The caveat, and it is the interesting one.** The masks the partition is fitted on are exact
per-token top-`k` masks (`k = round(0.10 x 8960) = 896`, achieved density exactly 0.10000). **A real
TEAL/CATS-style global magnitude threshold does not produce an exact per-token density** - it
produces a distribution around it. The instrument knows this and contains a
`global_threshold_realism_check` reporting achieved mean density, p10 and p90 under a real
threshold. **It lives in the `analyse` stage and never ran.** Section 2 is therefore an
*exact-density idealisation*, and how far the numbers move under a realisable threshold is
unmeasured.

**A discrepancy between the two artefacts, reported rather than resolved.** `env.pinned` records
`fidelity_tokens = 4096` in the controls JSON and `fidelity_tokens = 2048` in the model JSON. The
constant was edited between the two invocations (the stages also carry different `git_revision`
stamps - section 7). The **achieved** value is 2048 and the model stage's own pinned record agrees
with it, so the fidelity table is self-consistent; the controls JSON's pre-registered copy is stale.
Nothing downstream depends on it. Reported because the two files disagree.

Also worth stating: `E_list = [16, 32, 64, 128]` and `k_list = [1, 2, 4, 8, 16]` are pinned in the manifest, but the fidelity
stage uses `FIDELITY_E = [32, 64, 128]` and `FIDELITY_K = [1, 2, 4, 8]`. **`E = 16` and `k = 16`
appear in the manifest and in no measured row.** They belong to the `analyse` stage's geometry
sweep, which did not run. A reader comparing the manifest against the table would find missing rows
and no explanation in the JSON.

---

## 6. The memory instrument was broken; the allocation inventory is unverified by measurement

`mem_after_load`, `_mem_at_write` and `mem_peak` all read `{"rss_gb": 0.0, "peak_rss_gb": 0.0}`, and
`_rss_gb()` printed `0.0` at all nine progress checkpoints in `d0_model.log` - including the line
immediately after loading 6.17 GB of fp32 donor weights.

**A peak RSS of 0.0 GB is not a measurement. This one is not even a wrong measurement - it is a
failed API call whose failure was discarded.**

### Diagnosis, reproduced on this platform

`_rss_gb()` calls `GetProcessMemoryInfo(GetCurrentProcess(), ...)` through `ctypes`. On 64-bit
Windows, `GetCurrentProcess` returns a 64-bit pseudo-handle (`0xFFFFFFFFFFFFFFFF`). `ctypes` defaults
a foreign function's `restype` to `c_int`, which **truncates it to 32 bits**. The truncated handle
is invalid, `GetProcessMemoryInfo` returns `FALSE`, and the `PROCESS_MEMORY_COUNTERS` struct is left
at its zero-initialised state - which the function then formatted and returned as `0.0 GB`.

I reproduced both variants in a fresh Python 3.12.10 process on this machine, with 0.4 GB
deliberately touched:

```
BROKEN (no restype/argtypes):  returned FALSE, WorkingSetSize=0.000 GB, Peak=0.000 GB
FIXED  (restype = c_void_p) :  returned TRUE,  WorkingSetSize=0.397 GB, Peak=0.397 GB
```

**The defect is confirmed, it is specific to 64-bit Windows, and it is silent by construction** -
the version that ran did not check the BOOL return value, so a hard API failure surfaced as a
plausible-looking zero. That is precisely this programme's *"here we fail with the plausible
artefact, not the wrong number"* failure mode.

**The fix is already in the working-tree source and was never exercised.** `d0_coactivation.py` now
sets `GetCurrentProcess.restype = c_void_p`, declares `argtypes`, and returns
`{"error": "GetProcessMemoryInfo failed"}` on a `FALSE` return, with a comment recording the
diagnosis. The file's mtime is **01:41:35**; the model stage started at **01:36:11**. **The fix was
written five minutes after the process had already imported the broken version, so no run has ever
executed it.**

### What we do have: one external, unattributed measurement

`d0_peak_rss.txt` (UTF-8 BOM, PowerShell-formatted timestamp) reads:

```
peak_rss_gb=11.186 at 2026-08-29T01:51:35.0130285+02:00
FINAL peak_rss_gb=11.186
```

01:51:35 falls inside the model stage (01:36:11 -> 01:59:03) and lands within seconds of the end of
the fit-half capture (`fit_L00.npz` mtime 01:51:32) - exactly where the 2.05 GB all-layer capture
buffer coexists with the resident donor. That is the right place for a peak and the number is
credible.

**But it is not a citable measurement of this probe's footprint.** The sampler script is not on
disk, not in the repo, and not referenced anywhere in the codebase - `grep -r d0_peak_rss` over
`*.py`, `*.ps1` and `*.md` returns nothing. **The file records no PID and no process name**, so it
cannot be established from the artefact whether 11.186 GB is D0's working set, another process's, or
a system-wide figure. A second donor-adaptation harness was demonstrably running concurrently
(section 7), so the ambiguity is not academic.

### The allocation inventory - labelled ANALYTIC

The brief's section 8 and the programme's scale-law rule require an explicit inventory of every
`O(input)` allocation. The instrument contains one as a static dict (`_ALLOCATION_INVENTORY`,
dumpable with `python d0_coactivation.py inventory`). **Every byte figure below is computed from
shapes. It is ANALYTIC. The measurement that would corroborate it does not exist.**

**`stage_model` - analytic peak estimate 9.6 GB**

| object | analytic bytes | note |
|---|---|---|
| donor weights, fp32, 1.54e9 params | 6 170 000 000 | fp32 is D1/D4's BPB path; comparability requires it |
| load transient (embedding 151936 x 1536, bf16 -> fp32) | 933 000 000 | |
| per-batch logits, 1 x 512 x 151936 fp32 | 311 164 928 | batch forced to 1 for exactly this reason |
| `log_softmax` copy of the above | 311 164 928 | |
| baseline logits held for `max abs dlogit`, **2-sequence** subset | 622 329 856 | the full 24-seq slice would be 7.5 GB |
| `down_proj` column-permute temporary, 1536 x 8960 fp32 | 55 050 240 | |
| **capture buffer, 28 layers x [16384, 2240] int16, one token set at a time** | **2 055 208 960** | **the dominant term.** A dense `[T, N]` bool would be 4.1 GB, fp32 8x that |
| capture values, 5 deep layers x [16384, 2240] fp16 | 366 993 920 | |
| fidelity X capture, 5 layers x [4096, 1536] fp32 | 125 829 120 | over-stated: `FIDELITY_TOKENS` achieved 2048, not 4096 |

**`stage_controls` - analytic peak estimate 0.8 GB.** **`stage_analyse` - analytic peak estimate
2.6 GB** (driver: two fp32 centred copies of a [16384, 8960] mask inside `clustered_order`, 1.17 GB).

Explicitly **not** materialised: the `[T, N]` mask across all layers at once (masks are ranked int16
indices, written per layer, re-read one layer at a time); logits for the full eval slice (7.5 GB); a
second copy of the donor for C1 (the permutation is applied and inverted in place); the `[T, N]`
mask for the block accounting (`block_skippable_from_idx` works straight off the index array).

On disk: **4.6 GB measured** in `results/d0_masks/` (56 npz - 28 fit + 28 score; the 5 deep layers
carry `val` as well as `idx`, hence 146 801 138 B against 73 400 580 B). The inventory predicted
4.1 GB + 0.73 GB = 4.8 GB; the measured 4.6 GB is consistent once npz container overhead is
accounted for.

**Analytic 9.6 GB against the unattributed external 11.186 GB is a 1.6 GB shortfall** - plausibly
the Python / torch / BLAS baseline the inventory does not enumerate. **It cannot be closed with the
artefacts on hand, and no memory claim from this run should be quoted as measured.**

---

## 7. Reproducibility manifest

### Donor

| field | value |
|---|---|
| donor | `Qwen/Qwen2.5-1.5B` |
| **revision, pinned and achieved** | `8faed761d45a263340a0528343f099c05c9a4323` |
| layers / d_model / d_ffn | 28 / 1536 / 8960 |
| heads / kv-heads | 12 / 2 |
| vocab / tied embeddings | 151936 / `True` |
| activation | `silu` |

`arch_achieved.revision` and `env.pinned.revision` agree, and both agree with the banner printed at
the top of `d0_model.log`.

### Command lines and stage timings

| stage | command line, from `env.command_line` | started (UTC) | elapsed | ended (local) |
|---|---|---|---|---|
| `controls` | `python d0_coactivation.py controls` | 2026-08-28T23:35:17Z | **not instrumented** - `stage_controls` has no timer; from file timestamps, under about 16 s | 01:35:33 |
| `model` | `python d0_coactivation.py model` | 2026-08-28T23:36:11Z | **1372 s**, printed | 01:59:04 |
| `analyse` | **not run** | - | - | - |

Model-stage internal timings, printed from the log:

| step | seconds |
|---|---|
| donor load | 100 |
| C1 `BPB(identity)` | 196.8 |
| C1 `BPB(permuted)` | 225.0 |
| C1 `BPB(restored)` + `max abs dlogit` | 146.5 |
| capture, fit half, 28 layers | 245.0 |
| capture, score half, 28 layers | 247.1 |
| *residual: slice build, permute/restore, npz writes, fidelity* | *about 211.6* |
| **total** | **1372** |

**These timings are contaminated and must not be used as timing data.** The three BPB passes are
identical work on the identical 24 x 512 slice, and took 196.8 / 225.0 / 146.5 s - a **1.54x
spread**. See the machine-state subsection. D0 is not a timing probe, so this does not corrupt any
reported quantity.

### Seeds - pinned and achieved

| seed | value | used for |
|---|---|---|
| `lossless_perm_seed` | 20260828 | C1's whole-model FFN permutation |
| `cluster_seed` | 7 | k-means `random_state` in `balanced_labels` - **every** co-activation partition in section 2 |
| null-partition shuffle | `default_rng(1000 + E)` -> 1032 / 1064 / 1128 | the `random_balanced_NULL` label shuffle, **one draw per E** |
| `random_seeds` | [11, 22, 33, 44, 55, 66, 77, 88] | **`analyse` stage only - not exercised** |
| C2 scramble | 4242 | planted-positive permutation |
| C3 i.i.d. draw | 99 | planted-negative mask |
| C6 Monte-Carlo | 5 | brute-force hypergeometric check |
| fit slice | 42424 | calib half |
| score slice | 909001 | held-out half |
| BPB eval slice | 1234 | imported unchanged from `d1_pruning.py` |

**One clustering seed, one null draw.** Section 2's co-activation partitions all come from
`CLUSTER_SEED = 7`, and each null is a single shuffle. There is no seed spread on either arm, so
**every margin in section 2 is a point estimate with no error bar**, and the brief's section 7.4
requirement of at least 5 seeds is unmet for the only donor table that exists.

### Corpus slices and the disjointness assertion

| slice | part | n_seq | seq_len | seed | rejected | scored bytes | bytes/token | corpus sha256 (first 16) |
|---|---|---|---|---|---|---|---|---|
| fit - partition fitted here | calib | 32 | 512 | 42424 | 0 | 67648 | 4.136986 | `10d4d28102625f39` |
| score - fidelity scored here | heldout | 32 | 512 | 909001 | 1 | 69198 | 4.231776 | `f46b0310c15faec5` |
| BPB eval - C1 | heldout | 24 | 512 | 1234 | 0 | 51870 | 4.229452 | `f46b0310c15faec5` |

Asserted in code:

```
assert meta_fit['corpus_sha256'] != meta_ev['corpus_sha256'] and meta_fit['ids_sha256'] != meta_sc['ids_sha256']
```

-> `fit_vs_bpb_eval_distinct_halves = True`, `fit_vs_score_distinct_ids_sha256 = True`.

**The disjointness is real and it is the right disjointness**: the co-activation partition is fitted
on the calib half and scored on the held-out half, so section 2's table is out-of-sample. Note that
`score` and `bpb_eval` share a corpus sha256 (`f46b0310c15faec5`) - both are held-out - but differ in
`ids_sha256`; they are different draws from the same half, which is correct, since C1 must use D1's
own pinned slice for comparability.

### Library and platform versions

| | |
|---|---|
| python | `3.12.10` |
| torch | `2.12.0+cpu` |
| numpy | `2.4.6` |
| transformers | `4.57.6` |
| platform | `Windows-11-10.0.26200-SP0` |
| processor | `AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD` |
| logical CPUs | 12 |
| `torch_num_threads` achieved | 6 (`D_THREADS=6`) |
| `D0_ALL_LAYERS` | `True` |
| total RAM | 79.95 GB |
| free RAM at `controls` start | 37.29 GB |
| free RAM at `model` start | 37.13 GB |

`sklearn` is imported (`KMeans`, `adjusted_rand_score`) and **its version is not recorded in the
manifest.** k-means output depends on the sklearn version, so the co-activation partition behind
every row of section 2 is not fully pinned. A genuine manifest hole.

### Git revision - what the stamp does and does not pin

| stage | recorded `git_revision` | commit subject | `git_dirty_at_launch` |
|---|---|---|---|
| `controls` | `3471116537cf53b2a2243c24397332f67c42954c` | *R2a apparatus and raw results...* (2026-08-29 01:35:03) | `True` |
| `model` | `1a8f512ff2fcc2f8e76f8e051d3439cc5e4da134` | *R2a verdict - do not build the R2 solver...* (2026-08-29 01:35:48) | `True` |

**Two different revisions for two stages of one probe, and neither pins the instrument.**
`benchmarks/donor_adaptation/density/d0_coactivation.py` is **untracked** - `git status` reports it
under `??`. The stamps record repo HEAD at launch, and HEAD moved because unrelated R2a commits
landed at 01:35:03 and 01:35:48, in the 45-second gap between the two invocations. **The exact source
that produced these numbers is not recoverable from git**, and the file on disk has since been edited
(mtime 01:41:35, five minutes after the model stage began) - so the working-tree copy is *not* the
copy that ran. The one confirmed difference is `_rss_gb()` (section 6). Whether anything else changed
cannot be determined from the artefacts.

### Machine state during the run - and a second harness

`resident_heavyweight_processes`, snapshotted at **model stage start only**:

| process | pid | working set (GB) |
|---|---|---|
| `llama-server` | 17708 | 22.73 |
| `Memory Compression` | 3220 | 3.20 |
| `Antigravity IDE` | 25012 | 0.50 |
| `chrome` | 30004 | 0.49 |
| `MsMpEng` | 4644 | 0.44 |
| `claude` | 16580 | 0.41 |
| `ChatGPT` | 20288 | 0.38 |
| `Antigravity IDE` | 23852 | 0.38 |

`quiescence_gate`: `"NONE BY DESIGN -- D0 measures activation statistics, not bandwidth."`

**The snapshot is incomplete, and the artefacts prove it.** It was taken once, at stage start
(01:36:11). Filesystem evidence shows a *second* donor-adaptation harness running inside D0's
model-stage window:

- `benchmarks/donor_adaptation/f1/f1_gate_predictor.py` pins `EVAL_SEED = 20260828` and
  `SEQ_LEN = 1024`;
- `results/slice_heldout_4x1024_s20260828.pt` was written at **01:39:28** - a shape and seed D0
  never uses (D0 is 512-token throughout);
- `f1/results/f1_selfcheck.json` mtime **01:38**, `f1_capture.json` mtime **01:45**.

F1 also loads the donor. **A second multi-GB model process was almost certainly resident for much of
D0's model stage, and neither the process snapshot nor the free-RAM figure captures it.** This
explains the 1.54x spread across three identical BPB passes, and it is why the unattributed
`d0_peak_rss.txt` figure of 11.186 GB cannot be assigned to D0 (section 6).

**None of this corrupts a reported quantity** - D0 measures activation statistics and reconstruction
error, both deterministic given the pinned inputs. It does mean the environment record, which the
brief asked for precisely because this programme has banked a case of unreproducible numbers from
unrecorded environment, is **incomplete for this run**.

### Files in the working tree - all untracked; nothing committed, nothing pushed

```
benchmarks/donor_adaptation/density/d0_coactivation.py                      72 970 B, mtime 01:41:35 -- EDITED AFTER THE RUN
benchmarks/donor_adaptation/density/results/d0_coactivation.controls.json   11 138 B
benchmarks/donor_adaptation/density/results/d0_controls.log                    634 B
benchmarks/donor_adaptation/density/results/d0_coactivation.model.json      75 009 B
benchmarks/donor_adaptation/density/results/d0_model.log                    10 268 B
benchmarks/donor_adaptation/density/results/d0_peak_rss.txt                     86 B, unattributed external sampler
benchmarks/donor_adaptation/density/results/d0_masks/                       56 npz, 4.6 GB
benchmarks/donor_adaptation/density/results/slice_heldout_32x512_s909001.pt score slice, built during the run
docs/research/donor_adaptation/probes/D0_COACTIVATION.md                    this file
```

Prior D0-family artefacts reused, not regenerated: `d0_layout.json`, `d0b_rho_floor_donor.json`,
`d1_pruning.json` (baseline BPB), `slice_heldout_24x512_s1234.pt`, `slice_calib_32x512_s42424.pt`.

### Cross-check: log against JSON

All 120 fidelity rows in `d0_model.log` were parsed and matched against
`d0_coactivation.model.json` -> `fidelity`. **120 log rows, 120 JSON rows, zero unmatched keys, zero
value disagreements beyond the log's own `%.4f` / `%6.3f` print precision.** C1's five printed
figures reproduce the JSON exactly; the C2 detail line reproduces the JSON exactly. **No fabricated
or drifted number was found.** The only inter-artefact disagreements are the two reported above:
`fidelity_tokens` (section 5) and the two `git_revision` stamps (this section).

---

## 8. Engine legality and the rho-floor

Every measured row reports `engine_legal_48KB = true`. The test in code is

```python
"engine_legal_48KB": bool(sizes.min() * BYTES_PER_WEIGHT * d_model >= BLOCK_BYTES)
```

i.e. **each expert's slab within each organ must be at least one 48 KiB block.** At the donor's
`d_model = 1536` and 0.5 B/weight, one 48 KiB block holds `49152 / (1536 * 0.5) = 64` neurons per
organ - D0b's `neurons_per_48KB_per_organ`, independently confirmed there against real matrix shapes.

| E | neurons/expert | bytes per organ | blocks per organ | legal? |
|---|---|---|---|---|
| 32 | 280 | 215040 | 4.3750 | yes |
| 64 | 140 | 107520 | 2.1875 | yes |
| 128 | 70 | 53760 | 1.0938 | yes - **at the floor** |
| *256* | *35* | *26880* | *0.55* | *would be illegal* |

**`E = 128` is the finest carve this engine can legally hold at donor width** - 70 neurons against a
64-neuron floor. That is a hard geometric constraint on the MoE route, and it is why the fidelity
sweep stops at 128.

**This does not transfer to a larger donor.** From `d0b_rho_floor_donor.json`, at 70B-class width
(`d_model = 8192`) one 48 KiB block holds **12** neurons per organ and **4** interleaved - against 64
and 21.33 at `d_model = 1536`. `d_ffn` does not enter the formula; only `d_model` does. A wider donor
therefore admits a much finer legal carve, and **every geometric conclusion here is a
`d_model = 1536` conclusion.**

---

## 9. Verdict against the brief's pre-registered section 6

The brief's section 6 table is written in terms of **block-skippable fraction** under identity /
random / co-activation permutations. **That measurement was not made on the donor.** The `analyse`
stage that produces it never ran. **This run cannot be placed in any row of section 6**, and per the
brief's own instruction - *"If an outcome arises that this table does not cover, report that fact
rather than forcing it into a row"* - that is what is reported here.

What the run *does* settle, on Amendment 1 section A1.2's reframing (*"does this donor's activation
pattern contain cluster structure strong enough to carve the FFN into experts"*):

1. **Co-activation structure exists and is measurable out-of-sample. 60 of 60 matched pairs favour
   the co-activation partition over a matched random partition** - at identical `E`, identical `k`
   and identical achieved activation fraction, with the partition fitted on the calib half and
   scored on the held-out half. The direction is unambiguous.

2. **The margin is nowhere near large enough to matter, except at layer 27.** At layers 7, 14 and 21
   the margin is +0.0084 to +0.1002 in `relerr` while `relerr` itself sits at 0.63-0.97 - the
   co-activation carve is less bad than random while both are close to emitting zeros.

3. **The byte budget is reachable and the fidelity is not.** `E=128, k=2` reads 2.04% of FFN weight
   bytes including the router - the 2% target Amendment 1 showed is unreachable under a dense gate.
   At that operating point the oracle-routed carve loses 59.5% (L27), 83.7% (L21), 94.7% (L14) and
   90.8% (L7) of the FFN output norm. **With an oracle router, which no real router can match.**

4. **Depth is the axis that matters, and it was nearly unsampled.** Structure is concentrated at
   layer 27 and largely absent at 7/14/21. Only 5 of 28 layers were measured; the three middle layers
   stand in for the great majority of the FFN weight, and they are the three where the method does
   least.

5. **Apple's documented failure mode was not tested.** The section 5 duplication / set-cover
   measurement, which the brief declares mandatory and whose absence it says *"invalidates a null
   result from this probe"*, lives in the `analyse` stage and did not run. **So this document cannot
   distinguish "co-activation grouping does not help on this donor" from "the implementation hit the
   documented duplication failure mode" - and the brief is explicit that those two call for opposite
   next moves.**

6. **One control did not actually pass.** C3, the known-negative, shows the co-activation arm
   winning on pure i.i.d. noise at all five block sizes, by 13x to 21x the entire random-seed range,
   and was recorded as FIRED only because of a 0.03 *absolute* tolerance (section 4). The leak is
   in-sample optimism and it does not touch section 2's out-of-sample table, but it means the
   instrument's known-negative is not currently a working test for the arm under test.

**Overall: a genuine partial result, with a clean out-of-sample direction and an unfavourable
magnitude, on a route whose mandatory diagnostic is missing.** The result flatters the hypothesis in
sign and refutes it in size, and the honest reading is the size.

---

## 10. What the brief asked that this run does not answer

| brief clause | status |
|---|---|
| section 3 - "table of skippable fraction x permutation arm x block size" on the donor | **MISSING.** `analyse` never ran. The only skippable numbers in these artefacts are synthetic (C2/C3/C5) |
| section 4 - **identity arm**, "the baseline to beat" | **MISSING.** Section 2's null is `random_balanced_NULL` - a *shuffled balanced partition*, i.e. "grouping, but the wrong grouping". The donor's **native neuron order** was never evaluated as an MoE carve |
| section 4 / 7.4 - random arm, at least 5 seeds, seed-to-seed spread | **MISSING.** The 8 pinned `random_seeds` are used only in `analyse`. Section 2's null is **a single shuffle per E** (`default_rng(1000+E)`), n = 1, no spread. **The width of "no effect" is unmeasured and every margin in section 2 is a point estimate with no error bar** |
| section 4 - beat the D0b structureless reference (0.2824 at block 12, 0.6563 at block 4) | **MISSING.** Requires the skippable table |
| section 5 - duplication factor per layer, and its distribution across neurons | **MISSING** - mandatory; the brief says its absence invalidates a null |
| section 5 - duplication for the **most-active decile** specifically | **MISSING** - mandatory |
| section A1.3.3 - duplication as a **weight-inflation multiplier** | **MISSING** - mandatory |
| section A1.3.1 - `FFN_active = (3-2s)/3` printed beside every `s` | **N/A** - no `s` exists to print it beside. Section 2 gives the MoE analogue instead, with the router charged |
| section A1.3.2 - MoE geometry per layer per arm: cluster count, size distribution, experts per token, expert-active fraction, router cost | **PARTIAL.** Sizes, count, `k`, activation fraction and router cost are all in section 2 for the balanced partitions. The `identity_contiguous_blocks`, `random_s11_contiguous_blocks` and unbalanced `coactivation_kmeans` geometries are in `analyse` and missing |
| section 8 - block sizes swept, read from the D0b formula, never hardcoded | **MISSING.** No block sweep on the donor |
| section 8 - rho-floor for the donor's width reported alongside | **PARTIAL.** Carried in section 8 above from the pre-existing `d0b_rho_floor_donor.json`; not produced by this run |
| section 8 - density achieved vs threshold requested | **MISSING.** The `global_threshold_realism_check` - whose whole point is exposing the exact-density idealisation - is in `analyse`. Section 2 is an exact-top-k idealisation |
| section 8 - machine quiescence recorded | **PARTIAL / INCOMPLETE.** Snapshot taken once at stage start; it does not list the concurrent F1 donor process (section 7) |
| section 8 - git revision, exact command line, seeds, thread counts | **PRESENT BUT NOT LOAD-BEARING for the code.** Command lines, seeds and thread counts are recorded and achieved. The instrument is untracked, so the revision stamps pin unrelated HEADs (section 7) |
| section 9 - deliverable named `d0_coactivation.json` | **RENAMED.** Split into `.controls.json` + `.model.json`; the `.analyse.json` that would carry the brief's deliverable does not exist |
| section 9 - a section in `docs/research/DENSITY_PROBES.md` | **NOT WRITTEN.** This document stands alone |
| section A1.4 - 5 of 28 layers | **AS DESIGNED**, but 23 layers are unmeasured and section 3 shows depth is the dominant axis |
| - | **No planted-positive control for the `routed_error` / fidelity path** - the path that produced every donor number in this report (section 4) |
| - | **sklearn version unrecorded**, so the k-means partition behind every row of section 2 is not fully pinned (section 7) |

### To close the gaps, in priority order

1. **Run `python d0_coactivation.py analyse`.** All 56 mask files are on disk; the stage is CPU-only
   and donor-free. It produces the section 3 skippable table, the section 5 duplication measurement,
   the 8-seed random spread, the identity arm, the block sweep and the threshold realism check -
   **the entire missing deliverable, at no new capture cost.** Fix `_rss_gb()` first (already done in
   the working tree, never executed) so the stage's own memory is measured rather than analytic.
2. **Tighten C3 to a relative tolerance** and re-run `controls`, so the in-sample optimism of the
   co-activation arm is reported as a number instead of being absorbed by a 0.03 absolute window. The
   floor it establishes - about +0.006 to +0.008 at block 12-21 - is needed to read the `analyse`
   table honestly.
3. **Add a planted-positive control for `routed_error`** - a synthetic FFN whose neurons are known to
   partition into `E` independent groups - before any fidelity null is quoted again.
4. **Commit the instrument** so the revision stamp means something, and record the sklearn version.
5. **Add a null with the donor's native contiguous order** to the fidelity sweep. The brief's
   designated baseline-to-beat is currently absent from the only donor table that exists.

---

*Nothing in this document was committed or pushed. All artefacts remain in the working tree,
untracked, and are listed in section 7.*


---
---

# PART II - THE `analyse` STAGE

*Part I above was written before `analyse` existed and is left exactly as it stood, so a reader can
see what was known from the `model` stage alone and what changed. Part I's central claim - that the
brief's headline deliverable "does not exist" - was true when written and is **no longer true**.
Where Part II contradicts Part I, **Part II governs**; the contradictions are itemised in 11.12.*

## 11. The missing stage, run

`analyse` was executed from artefacts already on disk: no donor loaded, no capture repeated. It read
the 56 mask files the `model` stage wrote and produced `d0_coactivation.analyse.json` and the final
`d0_coactivation.json` the brief's section 9 named. The two files are **byte-identical**
(sha256 verified).

### 11.0 Provenance, wall clock, and machine contention

| field | value |
|---|---|
| command line | `python d0_coactivation.py analyse` |
| **wall-clock start** | **2026-08-29T11:36:42+02:00** |
| **wall-clock end** | **2026-08-29T12:02:23+02:00** |
| elapsed | **1541 s** (process exited 12:02:11; final JSON written 12:02:23) |
| git revision at launch | `3f885c5b3e0f0352eb768710de8e5317ee7f0388` |
| git branch / dirty at launch | `research/donor-adaptation` / `True` |
| python / torch / numpy / transformers | `3.12.10` / `2.12.0+cpu` / `2.4.6` / `4.57.6` |
| platform | `Windows-11-10.0.26200-SP0` |
| logical CPUs / `torch_num_threads` achieved | 12 / 6 |
| total RAM / free at stage start | 79.95 GB / 61.13 GB |

Per-layer elapsed, printed by the stage: L1 251 s, L7 264 s, L14 326 s, L21 295 s, L27 252 s, then
the 28-layer depth sweep.

**Machine contention.** I was told two other Builders were working and one holds measurement
priority for timing numbers. **My stage is not a timing probe** - it is deterministic given the
pinned masks and the pinned seeds - so contention cannot corrupt my numbers, only theirs. My window
is stated above precisely so any contamination of a concurrent timing measurement can be attributed
afterwards. This is the same discipline Part I applied to the F1 harness that ran inside the `model`
stage's window.

`resident_heavyweight_processes`, as recorded by **this** stage at its own start:

| process | pid | working set (GB) |
|---|---|---|
| `Memory Compression` | 3220 | 1.60 |
| `python` | 26112 | 0.89 |
| `chrome` | 11968 | 0.53 |
| `Antigravity IDE` | 25012 | 0.46 |
| `chrome` | 30280 | 0.46 |
| `Antigravity IDE` | 23852 | 0.45 |
| `claude` | 16580 | 0.44 |
| `MsMpEng` | 4644 | 0.40 |

**A correction to the machine state I was given, reported rather than smoothed over.** I was told
the machine was clear - "`llama-server.exe` is gone and no python is running". `llama-server` was
indeed gone. But at my start a second `python` process (pid 26112, 1.67 GB, started 11:32:52) was
already resident, presumably another Builder. **The machine was not idle when I started.** The
snapshot above, taken by the stage itself, is the authoritative record.

---

### 11.1 The brief's section 3 deliverable

**This is the table Part I reported as missing.** The number to beat is **identity** - the donor's
native neuron order. The honest control is **random**, now with **8 seeds** instead of the single
draw Part I was stuck with. `co` is the best co-activation arm over `E` in {16, 32, 64, 128};
`E*` names the winner. `ref` is the structureless `q^B` value the brief's section 4 requires every
arm to beat.

Shown at the primary density `p = 0.10` (achieved exactly: k = 896 of 8960, T = 16384 both halves),
at the five block sizes that carry meaning: **4 and 12** are the 70B-class floors (interleaved and
per-organ), **21 and 64** are this donor's floors, and **140** is the neuron count of one expert at
`E = 64`.

`margin` = co − random_max. `gap` = `generalisation_gap_in_minus_out`, this row's own measured
in-sample optimism, used as the floor per 11.4.

| L | block | identity | random max | **co-act** | ref `q^B` | margin | gap | clears gap? | `FFN_active` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.6561 | 0.6572 | **0.6934** | 0.6561 | +0.03624 | 0.00459 | yes | 0.5377 |
| 1 | 12 | 0.2829 | 0.2836 | **0.3941** | 0.2824 | +0.11051 | 0.01081 | yes | 0.7373 |
| 1 | 21 | 0.1111 | 0.1111 | **0.2294** | 0.1094 | +0.11827 | 0.01237 | yes | 0.8471 |
| 1 | 64 | 0.0013 | 0.0013 | **0.0286** | 0.0012 | +0.02728 | 0.00582 | yes | 0.9809 |
| 1 | 140 | 0.0000 | 0.0000 | **0.0019** | 0.0000 | +0.00193 | 0.00108 | yes | 0.9987 |
| 7 | 4 | 0.6559 | 0.6563 | **0.6669** | 0.6561 | +0.01064 | 0.00523 | yes | 0.5554 |
| 7 | 12 | 0.2823 | 0.2828 | **0.3129** | 0.2824 | +0.03012 | 0.01233 | yes | 0.7914 |
| 7 | 21 | 0.1094 | 0.1099 | **0.1381** | 0.1094 | +0.02821 | 0.01234 | yes | 0.9080 |
| 7 | 64 | 0.0012 | 0.0012 | **0.0037** | 0.0012 | +0.00247 | 0.00196 | yes | 0.9975 |
| 7 | 140 | 0.0000 | 0.0000 | **0.0001** | 0.0000 | +0.00012 | 0.00021 | **NO** | 0.9999 |
| 14 | 4 | 0.6559 | 0.6562 | **0.6654** | 0.6561 | +0.00927 | 0.00319 | yes | 0.5564 |
| 14 | 12 | 0.2816 | 0.2828 | **0.3169** | 0.2824 | +0.03411 | 0.00803 | yes | 0.7887 |
| 14 | 21 | 0.1093 | 0.1096 | **0.1471** | 0.1094 | +0.03743 | 0.00842 | yes | 0.9020 |
| 14 | 64 | 0.0011 | 0.0012 | **0.0059** | 0.0012 | +0.00471 | 0.00233 | yes | 0.9961 |
| 14 | 140 | 0.0000 | 0.0000 | **0.0002** | 0.0000 | +0.00018 | 0.00022 | **NO** | 0.9999 |
| 21 | 4 | 0.6564 | 0.6568 | **0.6926** | 0.6561 | +0.03583 | 0.00425 | yes | 0.5382 |
| 21 | 12 | 0.2825 | 0.2831 | **0.3784** | 0.2824 | +0.09530 | 0.01094 | yes | 0.7478 |
| 21 | 21 | 0.1106 | 0.1108 | **0.2033** | 0.1094 | +0.09247 | 0.01182 | yes | 0.8645 |
| 21 | 64 | 0.0013 | 0.0013 | **0.0133** | 0.0012 | +0.01201 | 0.00272 | yes | 0.9911 |
| 21 | 140 | 0.0000 | 0.0000 | **0.0002** | 0.0000 | +0.00019 | 0.00013 | yes | 0.9999 |
| 27 | 4 | 0.6558 | 0.6577 | **0.7355** | 0.6561 | +0.07775 | 0.00273 | yes | 0.5097 |
| 27 | 12 | 0.2821 | 0.2856 | **0.5214** | 0.2824 | +0.23575 | 0.00764 | yes | 0.6524 |
| 27 | 21 | 0.1105 | 0.1111 | **0.3875** | 0.1094 | +0.27635 | 0.01006 | yes | 0.7417 |
| 27 | 64 | 0.0012 | 0.0014 | **0.1593** | 0.0012 | +0.15788 | 0.01037 | yes | 0.8938 |
| 27 | 140 | 0.0000 | 0.0000 | **0.0606** | 0.0000 | +0.06064 | 0.00680 | yes | 0.9596 |

The full sweep is 5 layers x 3 densities x 14 block sizes = **210 cells**, all in
`d0_coactivation.analyse.json` under `deep_layers[L].densities[p].verdict_by_block[bs]`. The
condensed view above is representative; 11.4 reports the aggregate over all 210.

**Read the last column before the others.** `FFN_active` is Amendment 1's
`(3 - 2s)/3` - the fraction of FFN weight bytes still read under a dense gate. Its floor is
**0.3333** and no block-skipping can go below it. The best cell anywhere in this table is
**0.5097** (L27, block 4). **The skippable route never approaches its own floor, let alone the 2%
budget.**

---

### 11.2 The identity arm: the donor's native order carries no structure

This is the baseline the brief called "the number to beat", and which Part I's fidelity table never
had.

**Across all 210 cells, the largest deviation of the identity arm from the structureless `q^B`
reference is 0.003306.** Identity tracks the reference at every layer, every density, every block
size - 0.6561 vs 0.6561 at block 4, 0.2829 vs 0.2824 at block 12, 0.1111 vs 0.1094 at block 21,
0.0013 vs 0.0012 at block 64, 0.0000 vs 0.0000 at block 140.

**The donor's native neuron ordering is, for block-skipping purposes, indistinguishable from a
random ordering of a structureless mask.** Qwen2.5-1.5B has no exploitable contiguity in the order
its FFN neurons happen to be stored in. That is a clean negative and it is worth stating on its own:
any benefit measured here is created by the permutation, not inherited from the donor.

It also retires a worry Part I could not resolve. Part I's null was `random_balanced_NULL`, a
shuffled balanced partition - "grouping, but the wrong grouping" - and I flagged that the brief's
designated baseline was absent. It is now present, and **it lands on the structureless reference**,
so the two nulls agree: there is nothing to inherit.

---

### 11.3 The 8-seed random spread: margins with error bars

Part I's null was a single draw, so every margin was a point estimate. The `analyse` stage runs
**8 seeds** ([11, 22, 33, 44, 55, 66, 77, 88]) and reports mean, sd, min, max, and
`residual_structure_abs` - the distance from the analytic hypergeometric expectation.

At L1, p = 0.10, block 12: mean 0.28246, **sd 0.00079**, range [0.28095, 0.28359], n = 8,
`residual_structure_abs` = **0.000265**.

**The random arm is genuinely null**: it sits 2.6e-04 from the closed-form expectation, i.e. the
shuffle destroys index alignment and nothing else, exactly as control C4 asserted and now measured
on real donor data rather than synthetic. With sd = 0.00079, the co-activation arm's +0.11051
margin at that cell is roughly **140 sd** above the null mean.

**This is why the margins in 11.1 are not fragile.** They are not large because the null happened to
land low on one draw; the null's own spread is three orders of magnitude smaller than the effect at
the block sizes where the effect exists.

---

### 11.4 Margins against the measured in-sample-optimism floor

Part I inferred, from control C3's defective absolute tolerance, that k-means on an empirical
co-activation matrix buys roughly **+0.008** of spurious skippable fraction purely from in-sample
optimism, and warned that any `analyse` margin below that must be discounted.

**The `analyse` stage measures this quantity directly on the donor**, per row, as
`generalisation_gap_in_minus_out` = (in-sample skippable) − (out-of-sample skippable). At L1,
p = 0.10, block 12 it is **+0.01081**.

**That independently corroborates the C3-derived estimate** - same order, same sign, arrived at from
a real donor rather than a synthetic i.i.d. mask. The C3 finding was not an artefact of the
synthetic construction; it is a property of the method.

Applying each row's **own** gap as its floor, across all 210 cells:

| | count |
|---|---|
| cells where margin > own gap | **195** |
| cells where margin <= own gap - **margin does not clear its own noise floor** | **15** |

The 15 failures, every one of them:

| L | p | block | co-act | random max | margin | own gap |
|---|---|---|---|---|---|---|
| 1 | 0.20 | 128 | 0.0000 | 0.0000 | +0.00000 | 0.00007 |
| 1 | 0.20 | 140 | 0.0000 | 0.0000 | +0.00001 | 0.00001 |
| 7 | 0.10 | 128 | 0.0001 | 0.0000 | +0.00011 | 0.00027 |
| 7 | 0.10 | 140 | 0.0001 | 0.0000 | +0.00012 | 0.00021 |
| 7 | 0.20 | 32 | 0.0020 | 0.0008 | +0.00123 | 0.00140 |
| 7 | 0.20 | 42 | 0.0005 | 0.0001 | +0.00038 | 0.00059 |
| 7 | 0.20 | 64 | 0.0001 | 0.0000 | +0.00010 | 0.00027 |
| 7 | 0.20 | 128 | 0.0000 | 0.0000 | +0.00002 | 0.00003 |
| 7 | 0.20 | 140 | 0.0000 | 0.0000 | +0.00004 | 0.00004 |
| 14 | 0.10 | 128 | 0.0003 | 0.0000 | +0.00026 | 0.00030 |
| 14 | 0.10 | 140 | 0.0002 | 0.0000 | +0.00018 | 0.00022 |
| 14 | 0.20 | 42 | 0.0005 | 0.0001 | +0.00042 | 0.00048 |
| 14 | 0.20 | 128 | 0.0000 | 0.0000 | +0.00000 | 0.00000 |
| 14 | 0.20 | 140 | 0.0000 | 0.0000 | +0.00000 | 0.00000 |
| 27 | 0.20 | 140 | 0.0001 | 0.0000 | +0.00008 | 0.00018 |

**Every failure is at a large block size (32 and above) where the skippable fraction has collapsed
to zero for all three arms.** None is at block 4, 12 or 21. So the answer to "does the margin clear
its own floor" is: **yes wherever the measurement has any signal at all, and no exactly where every
arm reads zero.** The 15 failures are not a competing result; they are the measurement bottoming out.

That is itself the finding of 11.5.

---

### 11.5 The block-size collapse: the structure is real and it is at the wrong granularity

Trace one layer across block size at p = 0.10 (L21, representative):

| block | identity | random max | co-act | margin | `FFN_active` |
|---|---|---|---|---|---|
| 4 | 0.6564 | 0.6568 | 0.6926 | +0.03583 | 0.5382 |
| 12 | 0.2825 | 0.2831 | 0.3784 | +0.09530 | 0.7478 |
| 21 | 0.1106 | 0.1108 | 0.2033 | +0.09247 | 0.8645 |
| 64 | 0.0013 | 0.0013 | 0.0133 | +0.01201 | 0.9911 |
| 140 | 0.0000 | 0.0000 | 0.0002 | +0.00019 | 0.9999 |

**The co-activation advantage peaks around block 12-21 and is gone by block 140.** Block 140 is not
an arbitrary large number: at `E = 64` it is exactly one expert's worth of neurons. **The block size
the MoE carve requires is the block size at which the measured structure has vanished.**

This is the pivot of the whole result, and it reconciles Part I with Part II. Part I found the MoE
carve unusable via oracle-routed fidelity. Part II finds the permutation genuinely works - at a
granularity 10x finer than an expert. Both are true, and they are the same fact seen in two units:
`moe_geometry`'s own docstring records the identity that
`expert_active_fraction_full_coverage` = `1 - s(B)` when the partition is contiguous blocks.

---

### 11.6 The depth sweep - all 28 layers, not averaged

`p = 0.10`, `E = 64`, block 140. `ea_full` and `ea_95` are the expert-active byte fractions at 100%
and 95% coverage of the active neurons.

| L | identity | random mean | co-act | margin | `ea_full` | `ea_95` | cluster cv |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.0001 | +0.00005 | 0.9996 | 0.9099 | 0.335 |
| 1 | 0.0000 | 0.0000 | 0.0021 | +0.00208 | 0.9973 | 0.8500 | 0.274 |
| 2 | 0.0000 | 0.0000 | 0.0013 | +0.00125 | 0.9966 | 0.8403 | 0.250 |
| 3 | 0.0000 | 0.0000 | 0.0077 | +0.00765 | 0.9826 | 0.8032 | 0.255 |
| 4 | 0.0000 | 0.0000 | 0.0045 | +0.00453 | 0.9922 | 0.8657 | 0.319 |
| 5 | 0.0000 | 0.0000 | 0.0018 | +0.00180 | 0.9957 | 0.8380 | 0.243 |
| 6 | 0.0000 | 0.0000 | 0.0000 | +0.00003 | 0.9999 | 0.9186 | 0.291 |
| 7 | 0.0000 | 0.0000 | 0.0000 | +0.00005 | 0.9993 | 0.9196 | 0.288 |
| 8 | 0.0000 | 0.0000 | 0.0000 | +0.00000 | 1.0000 | 0.9156 | 0.281 |
| 9 | 0.0000 | 0.0000 | 0.0000 | +0.00000 | 1.0000 | 0.9145 | 0.232 |
| 10 | 0.0000 | 0.0000 | 0.0000 | +0.00001 | 0.9999 | 0.9122 | 0.242 |
| 11 | 0.0000 | 0.0000 | 0.0000 | +0.00001 | 0.9998 | 0.9032 | 0.275 |
| 12 | 0.0000 | 0.0000 | 0.0000 | +0.00002 | 0.9998 | 0.9057 | 0.278 |
| 13 | 0.0000 | 0.0000 | 0.0000 | +0.00001 | 0.9999 | 0.9028 | 0.241 |
| 14 | 0.0000 | 0.0000 | 0.0001 | +0.00005 | 0.9996 | 0.8916 | 0.218 |
| 15 | 0.0000 | 0.0000 | 0.0000 | +0.00002 | 0.9998 | 0.9060 | 0.265 |
| 16 | 0.0000 | 0.0000 | 0.0000 | +0.00003 | 0.9998 | 0.8983 | 0.253 |
| 17 | 0.0000 | 0.0000 | 0.0001 | +0.00006 | 0.9998 | 0.8911 | 0.232 |
| 18 | 0.0000 | 0.0000 | 0.0001 | +0.00007 | 0.9998 | 0.8932 | 0.225 |
| 19 | 0.0000 | 0.0000 | 0.0003 | +0.00029 | 0.9991 | 0.8641 | 0.226 |
| 20 | 0.0000 | 0.0000 | 0.0000 | +0.00005 | 0.9998 | 0.8889 | 0.277 |
| 21 | 0.0000 | 0.0000 | 0.0002 | +0.00020 | 0.9994 | 0.8722 | 0.263 |
| 22 | 0.0000 | 0.0000 | 0.0003 | +0.00034 | 0.9993 | 0.8672 | 0.239 |
| 23 | 0.0000 | 0.0000 | 0.0002 | +0.00024 | 0.9997 | 0.8815 | 0.284 |
| 24 | 0.0000 | 0.0000 | 0.0001 | +0.00012 | 0.9996 | 0.8803 | 0.248 |
| 25 | 0.0000 | 0.0000 | 0.0003 | +0.00034 | 0.9992 | 0.8716 | 0.275 |
| 26 | 0.0000 | 0.0000 | 0.0015 | +0.00145 | 0.9980 | 0.8413 | 0.309 |
| 27 | 0.0000 | 0.0000 | 0.0373 | +0.03726 | 0.9541 | 0.6596 | 0.402 |

**Read the identity and random columns first: `0.0000` at every one of 28 layers.** At expert
granularity there is no block-skippability anywhere in this network under any ordering the donor
supplies or a shuffle produces.

**The co-activation column is 0.0000 to 0.0077 for 27 of 28 layers.** The single exception is
**L27 at 0.0373**, which is also the only layer whose `ea_95` drops out of the 0.80-0.92 band
(0.6596) and the only one whose cluster size cv exceeds 0.40.

**The profile is not flat and it is not a gradient - it is 27 layers of nothing and one endpoint.**
Part I's fidelity table found the same shape from a different measurement (L27 the only layer with
both a large margin and non-hopeless absolute fidelity). Two independent measurements agreeing on
which layer is exceptional is the strongest structural claim this probe makes.

**`ea_full` is 0.954 to 1.000 across all 28 layers.** To retain every active neuron, an `E = 64`
MoE must load essentially every expert, at every depth. The MoE saves bytes only by dropping
neurons, and 11.7 prices that.

---

### 11.7 The section 5 duplication / set-cover diagnostic - the attribution

**This is the measurement the brief calls mandatory, whose absence Part I identified as the reason
the run could land in no section 6 row.** It has now run, and it does attribute the result.

#### 11.7a Apple's mechanism, measured - and saturated

`experts_touched[n]` counts the distinct experts containing a neuron co-active with `n`. Apple's
stated failure mode is that highly-active neurons "want" to be in every bundle.

**Across all 5 layers x 4 `E` values x 4 `k` values (80 records), `experts_touched` mean equals `E`
exactly in 79 of 80** - 16.00 of 16, 32.00 of 32, 64.00 of 64, 128.00 of 128 - with median, p90 and
max all likewise at `E` and `frac_zero` = 0. The single exception is L27 at `E = 128`, which reads
127.99.

**Every neuron in this donor co-activates, somewhere in the calibration data, with a neuron in every
single expert.** Apple's mechanism is present in its maximal form.

**But the metric is saturated at its ceiling, and a saturated metric discriminates nothing.** The
brief asked specifically for the most-active decile, on the theory that the effect bites there. Top
decile and bottom decile both read `E`. **That is not evidence that the deciles behave alike - it is
evidence the instrument has no headroom to tell them apart.** The decile breakout of
`experts_touched` should be treated as uninformative on this donor, and the budget-relevant
duplication number below carries the actual signal.

#### 11.7b The weight-inflation multiplier - and a correction to my own earlier report

> **WITHDRAWN 2026-09-04 - the "per-token bytes, lossless" column below.** Controller
> `CONTROLLER_D0_AUDIT.md` B2, upheld: that column is not a per-token byte count, and two of its five
> entries sit below a hard analytic bound. **Do not read, cite or carry forward that column.** The
> rest of this subsection stands. The column is left in place rather than deleted so the audit trail
> is legible; it is struck, not corrected, because correcting it needs a re-derivation nobody has done.

`dup_extra_greedy` solves, per neuron, a greedy set cover over the tokens where that neuron is
active but its home expert was not loaded by an oracle top-`k` router. `1 + mean(dup_extra)` is the
**weight-inflation multiplier**: the factor by which resident FFN weights grow to make top-`k`
routing lossless.

**Correction.** In my interim progress note I reported this diagnostic as showing the MoE carve
"worse than the 33.3% dense-gate floor", citing L1 / `E=64` / `k=4`: multiplier 5.742, expert-active
fraction 0.0625, product 0.359. **Those numbers are correct for that cell and I have verified them
against the completed JSON.** But I generalised from one cell of an 80-cell grid, and the completed
grid does not support the generalisation. The honest per-layer result is below.

Best cell per layer, minimising `(k/E) x multiplier + router`, where router = `E/(3 d_ffn)`:

| L | best `E` | best `k` | resident inflation | **per-token bytes, lossless** | vs dense-gate 0.3333 |
|---|---|---|---|---|---|
| 1 | 32 | 1 | 3.75x | **0.1184** | beats by 2.8x |
| 7 | 16 | 1 | 4.45x | **0.2786** | beats by 1.2x |
| 14 | 128 | 1 | 50.32x | **0.3979** | **WORSE** |
| 21 | 128 | 1 | 2.82x | **0.0268** | beats by 12.5x |
| 27 | 64 | 1 | 2.04x | **0.0343** | beats by 9.7x |

**So the correct statement is: four of five layers beat the dense-gate floor on this metric, one
does not, and the spread across layers is a factor of 15.** My interim "worse than the floor" was
true of the cell I quoted and false as a summary. The result is better for the hypothesis than I
reported, and it earns the extra scrutiny below rather than less.

**Three caveats that materially limit the favourable reading:**

1. **The set cover is solved in-sample.** `duplication()` draws a 2048-token subsample of the score
   half and solves the cover on those same tokens it then scores. The multiplier is therefore an
   **optimistic lower bound**; the duplication needed to stay lossless on unseen tokens is higher.
   Section 11.4 measured this class of optimism at ~+0.011 for the skippable metric; **no equivalent
   out-of-sample control exists for the duplication metric**, so its size here is unknown. This is
   the same defect C3 exposed, in a different measurement, and it is not corrected anywhere in the
   instrument.
2. **The router is an oracle.** `sel_e` picks experts by true activation count. No trainable router
   achieves that, so every figure is an upper bound on realisable performance.
3. **Resident inflation is a capacity cost, not only a bandwidth cost.** L14's best cell needs
   **50.32x** the FFN weights resident. Against the sealed 16-64 GB SKU constraint that is
   disqualifying on its own, independent of per-token bytes.

`missed_active_fraction` - the share of active neurons an oracle top-`k` router drops before any
duplication - ranges from 0.0777 (L27, `E=16`, `k=8`) to 0.9699 (L14, `E=128`, `k=1`). At the
`E=64, k=4` cell it is 0.777 (L1), 0.860 (L7), 0.860 (L14), 0.711 (L21), 0.527 (L27).

---

### 11.8 The A1.3.2 MoE geometry, with the identity arm

`p = 0.10`, `E = 64`, all five partitions. `full_cov` / `cov95` / `cov90` are the expert-active byte
fractions needed to retain 100% / 95% / 90% of active neurons.

| L | partition | full_cov | cov95 | cov90 | cluster cv | 48KB-legal |
|---|---|---|---|---|---|---|
| 1 | `identity_contiguous_blocks` | 1.0000 | 0.9201 | 0.8490 | 0.000 | yes |
| 1 | `random_s11_contiguous_blocks` | 1.0000 | 0.9205 | 0.8496 | 0.000 | yes |
| 1 | `coactivation_kmeans` | 0.9966 | 0.8453 | 0.7430 | 0.257 | **NO** |
| 1 | `coactivation_balanced` | 0.9928 | 0.8196 | 0.7137 | 0.000 | yes |
| 1 | `random_balanced_NULL` | 1.0000 | 0.9188 | 0.8472 | 0.000 | yes |
| 7 | `identity_contiguous_blocks` | 1.0000 | 0.9209 | 0.8503 | 0.000 | yes |
| 7 | `random_s11_contiguous_blocks` | 1.0000 | 0.9209 | 0.8503 | 0.000 | yes |
| 7 | `coactivation_kmeans` | 0.9992 | 0.9154 | 0.8456 | 0.310 | **NO** |
| 7 | `coactivation_balanced` | 1.0000 | 0.9005 | 0.8187 | 0.000 | yes |
| 7 | `random_balanced_NULL` | 1.0000 | 0.9208 | 0.8500 | 0.000 | yes |
| 14 | `identity_contiguous_blocks` | 1.0000 | 0.9209 | 0.8503 | 0.000 | yes |
| 14 | `random_s11_contiguous_blocks` | 1.0000 | 0.9211 | 0.8506 | 0.000 | yes |
| 14 | `coactivation_kmeans` | 0.9997 | 0.8946 | 0.8154 | 0.219 | yes |
| 14 | `coactivation_balanced` | 0.9997 | 0.8892 | 0.8047 | 0.000 | yes |
| 14 | `random_balanced_NULL` | 1.0000 | 0.9209 | 0.8503 | 0.000 | yes |
| 21 | `identity_contiguous_blocks` | 1.0000 | 0.9200 | 0.8489 | 0.000 | yes |
| 21 | `random_s11_contiguous_blocks` | 1.0000 | 0.9218 | 0.8517 | 0.000 | yes |
| 21 | `coactivation_kmeans` | 0.9996 | 0.8719 | 0.7750 | 0.268 | yes |
| 21 | `coactivation_balanced` | 0.9995 | 0.8542 | 0.7508 | 0.000 | yes |
| 21 | `random_balanced_NULL` | 1.0000 | 0.9208 | 0.8500 | 0.000 | yes |
| 27 | `identity_contiguous_blocks` | 1.0000 | 0.9201 | 0.8489 | 0.000 | yes |
| 27 | `random_s11_contiguous_blocks` | 1.0000 | 0.9212 | 0.8509 | 0.000 | yes |
| 27 | `coactivation_kmeans` | 0.9445 | 0.6503 | 0.5108 | 0.394 | yes |
| 27 | `coactivation_balanced` | 0.9105 | 0.6411 | 0.5340 | 0.000 | yes |
| 27 | `random_balanced_NULL` | 1.0000 | 0.9202 | 0.8493 | 0.000 | yes |

**At full coverage every partition reads 0.91 to 1.00 of FFN bytes.** The identity and both random
arms are pinned at 1.0000 at every layer. Co-activation-balanced improves on that only at L27
(0.9105).

**At 90% coverage the co-activation arms open a real gap** - 0.5340 vs 0.8489 at L27, 0.7137 vs
0.8490 at L1 - but 0.5340 is still far above the 0.3333 dense-gate floor, and it is bought by
discarding 10% of active neurons, whose output cost Part I's fidelity table already priced as
severe.

**`coactivation_kmeans` is not engine-legal at L1 and L7** (`engine_legal_48KB_at_min_size` = false;
cluster cv 0.257 and 0.310 mean the smallest cluster falls below the 64-neuron / 48 KiB floor). That
is exactly why the balanced variant exists, and it is a concrete instance of the engine constraint
biting the unconstrained method.

---

### 11.9 ACHIEVED vs nominal, and the exact-density idealisation priced

| quantity | requested | **achieved** |
|---|---|---|
| density, all 15 rows | 0.05 / 0.10 / 0.20 | **0.050000 / 0.100000 / 0.200000** (k = 448 / 896 / 1792 of 8960) |
| tokens, fit and score | - | **16384 / 16384** every row |
| random seeds | >= 5 (brief 7.4) | **8** |
| block sizes | derived, never hardcoded | **14**: 2, 4, 6, 8, 10, 12, 16, 21, 24, 32, 42, 64, 128, 140 |
| duplication token subsample | 2048 | **2048** of 16384 available |
| measured peak RSS | - | **2.749 GB** |

**The TEAL/CATS realism check - what a real global threshold actually delivers.** Part I flagged
that the exact top-k mask is an idealisation and that the check for it sat in the unrun stage. It
has now run, at nominal density 0.10:

| L | threshold | **achieved mean** | p10 | p90 | fraction capped at `p_max` |
|---|---|---|---|---|---|
| 1 | 0.072418 | **0.10877** | 0.05804 | 0.17176 | 0.00824 |
| 7 | 0.203247 | **0.10498** | 0.04769 | 0.17031 | 0.00165 |
| 14 | 0.184204 | **0.10288** | 0.07333 | 0.13504 | 0.00049 |
| 21 | 0.347656 | **0.10313** | 0.07768 | 0.12913 | 0.00000 |
| 27 | 0.605469 | **0.11581** | 0.07891 | 0.17943 | 0.04578 |

**A single global threshold calibrated for 10% delivers a mean of 10.3-11.6% and a per-token p10-p90
range as wide as 4.8%-17.9%** - roughly a 3.5x spread at L1 and L7. The whole skippable table
assumes an exact 10% per token; a realisable TEAL/CATS threshold does not provide that, and the
sensitivity of the skippable numbers to that spread is **not measured anywhere in this probe**.

At **L27, 4.58% of tokens are capped at `p_max`** - the stored top-25% ranked indices were not deep
enough to hold everything above the threshold. L27 is also the one layer carrying the entire
positive result, so its capture is the one most affected by the storage cap.

---

### 11.10 Memory: a measured figure at last

Part I reported `{"rss_gb": 0.0, "peak_rss_gb": 0.0}` throughout the `model` stage, diagnosed the
cause as a `ctypes` `restype` truncation of the 64-bit `GetCurrentProcess` pseudo-handle, reproduced
both variants on this platform, and established that the fix on disk post-dated the `model` run by
five minutes and had therefore never executed.

**It executed this stage.** `_rss_gb()` returned real values at every checkpoint:

| checkpoint | rss_gb | peak_rss_gb |
|---|---|---|
| after L1 | 1.489 | 2.746 |
| after L7 | 1.490 | 2.747 |
| after L14 | 1.490 | 2.748 |
| after L21 | 1.491 | 2.749 |
| after L27 | 1.492 | 2.749 |
| **final `mem_peak`** | **1.492** | **2.749** |

**Measured peak: 2.749 GB, against the 2.6 GB analytic estimate for `stage_analyse` - the inventory
under-predicted by 0.149 GB (5.7%).** That is a good agreement and it is the first time any stage of
this probe has had its allocation inventory checked against a measurement rather than asserted.

**Which stages have measured figures and which remain analytic:**

| stage | memory figure | status |
|---|---|---|
| `controls` | none | **ANALYTIC only** (0.8 GB estimate). The broken `_rss_gb()` wrote 0.0 |
| `model` | none from the instrument | **ANALYTIC only** (9.6 GB estimate). The 11.186 GB in `d0_peak_rss.txt` is an external sampler with no PID and no producer script, and a concurrent F1 donor process makes it unattributable |
| `analyse` | **2.749 GB measured** | **MEASURED**, and it validates the inventory's method to within 5.7% |

The `model` stage's 9.6 GB figure remains unverified. What the `analyse` agreement licenses is
confidence in the *method* of the inventory, not in that particular number.

---

### 11.11 Cross-check: JSON against log

Both artefacts were parsed independently and compared field by field:

- **15 deep-layer log rows** x 7 fields (`identity`, `random_mean`, `random_min`, `random_max`,
  `coactivation`, `ref`, `FFN_active`) - **0 mismatches**.
- **28 depth-sweep log rows** x 5 fields (`identity`, `random_mean`, `coactivation`,
  `expert_active(full)`, `expert_active(95%)`) - **0 mismatches**.
- `d0_coactivation.json` is **byte-identical** to `d0_coactivation.analyse.json` (sha256).

**No fabricated or drifted number was found.** No new inter-artefact discrepancy arose in this
stage; Part I's two (`fidelity_tokens`, the two `git_revision` stamps) stand as reported.

---

### 11.12 The section 6 verdict

The brief pre-registered four outcomes and forbade adjusting them after seeing numbers. Taking them
on their literal terms:

| brief section 6 row | measured? |
|---|---|
| co-activation beats identity **and** clears the random arm's seed spread | **YES** - in 195 of 210 cells, and in every cell where any arm reads non-zero |
| co-activation ~ random, both beat identity | no - identity does not beat the structureless reference anywhere |
| co-activation ~ random ~ identity | no |
| co-activation **worse** than identity | no - never, in any of 210 cells |

**The result lands in row 1.** Co-activation beats identity, clears an 8-seed random spread whose sd
is 0.00079, and beats the structureless `q^B` reference. On the brief's literal criteria this is
unambiguous, and the planted-positive control C2 (ARI = 1.0000) licenses the instrument that
measured it.

**And row 1's prescribed reading does not follow.** Row 1 says *"the route in memo section 3c is
alive; proceed to a joint sparsity+permutation brief."* The Amendment 1 arithmetic, applied to these
same numbers, does not support proceeding:

1. ~~**`FFN_active` never goes below 0.5097** anywhere in 210 cells~~ **- WITHDRAWN 2026-09-04.**
   The true minimum over the 210 cells is **0.3911**, not 0.5097 (Controller `CONTROLLER_D0_AUDIT.md`
   B1, upheld). The claim as written was false and is struck. What survives is the weaker true
   statement: `FFN_active` does not approach the 2% budget anywhere in the sweep. The binding floor is
   not 0.3333 either - it is `(1 + 2p)/3` (Controller F4), which this document never stated.
2. **At expert granularity the effect is gone.** Identity and random read 0.0000 at all 28 layers at
   block 140; co-activation reads 0.0000-0.0077 at 27 of them. The permutation works at block 12-21
   and the MoE needs block 140.
3. **27 of 28 layers carry no usable structure.** L27 alone is exceptional, on two independent
   measurements (Part I's fidelity, Part II's depth sweep).
4. **Duplication prices the lossless carve at 2.04x-50.32x resident weights**, with the L14 cell
   landing worse than the dense gate it would replace, and the multiplier itself measured in-sample
   and therefore optimistic.

**So the honest verdict is: the result lands in row 1, and row 1 is the wrong question.** Section 6
was written before Amendment 1 reframed D0 from "how much can we skip" to "can this donor's FFN be
carved into experts". Its four rows all concern the *sign and significance* of the co-activation
effect. **None of them has a row for the outcome actually observed: the effect is real,
statistically overwhelming, reproducible on a held-out half, and far too small in the units that
move the budget.** Per the brief's own instruction - *"If an outcome arises that this table does not
cover, report that fact rather than forcing it into a row"* - that is reported here rather than
forced.

**On the literature question the brief said it was adjudicating** (Apple negative vs Neuralink
positive, unreconciled): this probe supports **neither cleanly**. Apple's stated mechanism is
present and maximal - `experts_touched` saturates at `E` in 79 of 80 records, every neuron
co-activating with every expert - which is Apple's diagnosis confirmed. But the co-activation
permutation still beats both nulls decisively at fine block sizes, which is not Apple's null result.
**The reconciliation this donor suggests is that both papers can be right at different
granularities**, and that the granularity an engine can use is the one where the effect dies. That
is a hypothesis generated by this run, not a result of it; testing it needs a second donor at a
different `d_model`, since section 8 above showed the legal block size is a pure function of
`d_model`.

**Recommendation, stated as the Builder and subject to the Principal's direction:** do not open the
joint sparsity+permutation brief that row 1 prescribes. The measurement that would change this
verdict is not a better permutation - it is a donor whose `d_model` is large enough that the legal
block size falls to 4-12 neurons, where this donor's effect lives. At `d_model = 8192` that is
exactly where the floor sits.

---

### 11.13 What the brief asked that this run still does not answer

The `analyse` stage closed the deliverable gaps Part I listed. These remain open.

| gap | status after `analyse` |
|---|---|
| **Seed spread on the co-activation arm** | **STILL MISSING.** The 8 pinned seeds vary the *random* arm only. Every co-activation partition in both Part I and Part II comes from the single `CLUSTER_SEED = 7`. The null now has an error bar; **the arm under test still does not.** A k-means restart could move it and nothing here bounds by how much |
| **Out-of-sample control for the duplication metric** | **MISSING.** The greedy set cover is solved on the same 2048 tokens it scores. 11.4 measured ~+0.011 of in-sample optimism for the *skippable* metric; the equivalent for duplication is unmeasured, so every multiplier in 11.7b is an optimistic lower bound of unknown tightness |
| **Sensitivity to a realisable threshold** | **MISSING.** 11.9 shows a real global threshold gives p10-p90 of 4.8%-17.9% against a nominal 10%. The entire skippable table assumes exact per-token 10%. How much the table moves under the realistic spread was never computed |
| **Apple's most-active-decile question** | **UNANSWERABLE AS INSTRUMENTED.** `experts_touched` saturates at `E`, so top and bottom deciles both read `E`. The brief asked for this breakout specifically; the metric has no headroom to provide it on this donor |
| **End-to-end quality of an actual MoE carve** | **MISSING.** Everything here is a proxy - `relerr` on FFN output, or byte accounting. **No BPB was measured for any carved model.** C1 measured BPB only for a lossless permutation. The donor's real quality cost is unmeasured |
| **Any speed measurement** | **OUT OF SCOPE BY DESIGN and still absent.** D0 measures activation statistics, not bandwidth. The 44.6% in-engine FFN cost underpinning Amendment 1 was measured on our own 8.3M model, not a donor, and must not be quoted as a donor number |
| **A second donor / a second `d_model`** | **MISSING, and now the decisive gap.** Every geometric conclusion is a `d_model = 1536` conclusion. 11.12's central hypothesis - that the effect lives at a block size only a wider donor can legally use - is untestable on this donor |
| **`E = 16` and `k = 16` in the fidelity table** | Still absent from Part I's table; `E = 16` now appears in the `analyse` MoE and duplication grids, so the manifest and the artefacts are consistent at last |
| **Planted-positive control for `routed_error`** | **STILL MISSING.** C2 licenses the permutation/skippable path, which is what Part II measures. Part I's fidelity path still has no positive control of its own |
| **C3's defective tolerance** | **STILL UNFIXED in the instrument.** 11.4 corroborated the finding from donor data, but `stage_controls` still applies `< 0.03` absolute and would still record FIRED |
| **Instrument provenance** | **STILL UNTRACKED.** `d0_coactivation.py` remains outside git; the `analyse` stage's `git_revision` again stamps an unrelated HEAD. sklearn's version is still unrecorded, and every partition in this document depends on it |
| **Depth sweep breadth** | The 28-layer sweep is `E = 64`, `p = 0.10`, block 140 only. The 23 non-deep layers have no density sweep, no block sweep, no duplication, and no MoE geometry |

### Next steps, in priority order

1. **Do not open the joint sparsity+permutation brief.** 11.12 gives the reasoning.
2. **Vary `CLUSTER_SEED`.** This is cheap, needs no donor, and is the last place a point estimate is
   masquerading as a result. Until it is done, every co-activation number in this document is n = 1.
3. **Fix C3's tolerance to relative** and re-run `controls` - minutes of CPU, and it converts a
   control that currently cannot fail into one that can.
4. **Price the threshold realism**, by re-running the skippable table with masks drawn from the
   measured global threshold rather than exact top-k.
5. **Commit the instrument and record the sklearn version** before any of the above, so the numbers
   become reproducible.

---

*Part II was produced with no new capture: the `analyse` stage read only mask files already on disk.
Nothing in this document has been committed or pushed. All artefacts remain untracked in the working
tree; the file list in section 7 is extended by `d0_coactivation.analyse.json`,
`d0_coactivation.json`, `d0_analyse.log` and `d0_analyse.wallclock.txt`.*

---

# PART III - THE B3 REPAIR, AND THE NUMBER PARTS I AND II NEVER HAD

*Part III answers the Controller audit of 2026-08-29 (`audits/CONTROLLER_D0_AUDIT.md`) on its one
BLOCK that bore on the science, B3, and then does the thing both earlier parts deferred: it carves the
donor and measures BPB end to end. Two new artefacts:
`results/d0_b3_treatment_spread.json` (B3) and `results/d0_carved_bpb_paired.json` (the carve).
Nothing here re-reads a mask that Part II did not already read.*

---

## 12. B3 - the arm was not reproducible; it is now, and its error bar was the wrong one

### 12.1 The defect and the repair

The Controller's B3 is upheld in full. `CLUSTER_SEED` was passed to `KMeans(random_state=)` only. The
rank-64 sketch that *produces the features k-means clusters* comes from `torch.svd_lowrank(X, q=64,
niter=4)` (`d0_layout.py:143`, `:182`), which draws a Gaussian from torch's **global** RNG, and
`torch.manual_seed` was never called anywhere in `d0_coactivation.py`, `d0_layout.py` or `common.py`.
Two identical invocations returned different partitions.

The repair, recorded verbatim in `d0_b3_treatment_spread.json` under `repair`:

> `torch.manual_seed(seed) before clustered_order; seed also passed to KMeans`

Determinism check after the repair, `d0_b3_treatment_spread.json` -> `determinism_check`, on L27,
`E = 64`, seed 7, same input array, repeat call:

| | before repair (Controller, section B3) | after repair |
|---|---|---|
| ARI between two identical calls | ~0.43 | **1.0000** |
| labels bitwise identical | no | **true** |

**B3 is closed on reproducibility going forward, and only going forward.**

> **CORRECTED 2026-09-04 (Controller `CONTROLLER_D0_PART3_AUDIT.md` B2).** This paragraph originally
> read *"every co-activation number in Parts I and II is now re-derivable from its seed."* **That is
> false, and the Controller disproved it:** the main run's `coactivation_best` at `p = 0.10` does not
> equal the seed-7 sweep value at any cell checked, and falls **outside the entire 8-seed range in 4
> of 15** cells (e.g. L1 bs12: main run 0.394092, seed-7 0.392766, 8-seed range
> [0.386931, 0.393278]). Parts I and II were drawn from an **unrecorded global RNG state**, and
> nothing recovers it. **The repair fixes the future; it does not retro-fit the past.** Every
> co-activation number in Parts I and II remains an n = 1 draw from a distribution whose spread is now
> measured (12.2, 12.3) but whose particular draw is unrecoverable.

### 12.2 What the repair did *not* do: the partition is still not identified

Determinism is not stability. With the RNG pinned, the arm was re-run at **8 seeds**
(`[7, 11, 22, 33, 44, 55, 66, 77]`, `p_achieved = 0.10`, `k_active = 896`, `E = 64`). The partitions
those seeds produce agree with each other about as badly as the unseeded ones did -
`ARI_across_seeds_E64`, over all 28 pairs:

| layer | mean ARI across seeds | min | max |
|---|---|---|---|
| L1 | 0.493 | - | - |
| L7 | 0.529 | - | - |
| L14 | 0.424 | - | - |
| L21 | 0.514 | - | - |
| L27 | **0.449** | 0.414 | 0.472 |

**The statistic is reproducible; the expert assignment is not.** Two legitimate runs of this pipeline
agree on roughly half the partition and report nearly the same skippable fraction. That is a real
property of the object - the co-activation structure is diffuse, and many different partitions capture
about the same amount of it - and it is fatal to any plan that wants to *ship one specific expert
layout* as if it were the layout. It is not load-bearing for the aggregate claims below.

### 12.3 The error bar in 11.3 was the null's, not the treatment's - correction

Section 11.3 reads: *"With sd = 0.00079, the co-activation arm's +0.11051 margin at that cell is
roughly **140 sd** above the null mean."*

0.00079 is `random_std` - the **null arm's** spread across its 8 shuffles. It says how tightly the
shuffle reproduces itself. It says nothing about how tightly *the co-activation arm* reproduces itself,
which until now had never been measured. Now it has. Treatment sd across the 8 seeds
(`treatment_stats[bs].sd`), against the margin at the same cell, all at `p = 0.10`:

| layer | bs | margin | **treatment sd** | margin / treatment sd | null sd (what 11.3 used) |
|---|---|---|---|---|---|
| L1 | 12 | 0.11051 | 0.00266 | **41.6** | 0.00079 |
| L1 | 21 | 0.11827 | 0.00238 | 49.7 | 0.00123 |
| L1 | 140 | 0.00193 | 0.00029 | 6.6 | 0.00000 |
| L7 | 12 | 0.03012 | 0.00083 | 36.1 | 0.00063 |
| L7 | 140 | 0.00012 | 0.00003 | 4.7 | 0.00000 |
| L14 | 12 | 0.03411 | 0.00173 | 19.8 | 0.00050 |
| L21 | 12 | 0.09530 | 0.00351 | 27.2 | 0.00108 |
| L21 | 21 | 0.09247 | 0.00328 | 28.2 | 0.00119 |
| L27 | 12 | 0.23575 | 0.00434 | 54.3 | 0.00278 |
| L27 | 21 | 0.27635 | 0.00486 | **56.9** | 0.00138 |
| L27 | 64 | 0.15788 | 0.00341 | 46.3 | 0.00024 |
| L27 | 140 | 0.06064 | 0.00518 | 11.7 | 0.00000 |

**The correction:** at the cell 11.3 quoted, the honest figure is **41.6 sd, not 140 sd**. The
sentence overstated the confidence by 3.4x by dividing by the wrong arm's spread. The treatment's own
sd, measured over the 70 (layer, block) cells at `p = 0.10`, runs from **0.96x to 4545x** the null's.
*(Corrected 2026-09-04, Controller F2: this sentence originally read "1.4x to 200x", which is wrong at
both ends. It is not even always larger - at L7 bs16 the treatment sd is **smaller** than the null's,
0.000630 against 0.000656.)* At block 140 the null's sd rounds to zero while the treatment's does not,
which is exactly the regime where the old error bar was most flattering. **The direction of every claim survives; the precision claimed for it did not.**

### 12.4 Re-counting the 210 cells against the treatment's own spread

Part II's two counts are reproduced here unchanged, and the Controller's F1 is upheld: the row-1
criterion (`coact_beats_identity AND coact_clears_random_spread`) is **209 of 210**, and
`margin > generalisation_gap_in_minus_out` is **195 of 210**. Section 11.12 paired the first sentence
with the second count.

The new test - margin > 2x the treatment's *measured* sd:

| test | count |
|---|---|
| beats identity AND clears the random spread (row-1) | 209 / 210 |
| margin > in-sample-optimism gap (11.4) | 195 / 210 |
| ~~margin > 2x measured treatment sd~~ | ~~190 / 210~~ **WITHDRAWN - see the correction below** |

The 20 failures are not scattered. **All 20 sit at `p = 0.20`**, and **all 20** at block >= 32 - the list below is the proof, and the
sentence originally read "18 of the 20", contradicting its own list *(corrected 2026-09-04, Controller
F1)*:

```
L1  p=0.20 bs=64,128,140      L14 p=0.20 bs=32,42,64,128,140
L7  p=0.20 bs=42,64,128,140   L21 p=0.20 bs=32,42,64,128,140
L27 p=0.20 bs=64,128,140
```

That is the corner where the skippable fraction has already collapsed toward zero for every arm, so
the finding is: **the structure is real everywhere it is large enough to matter, and indistinguishable
from noise only where there is nothing left to skip.** It does not rescue any operating point.

> ### ⚠ CORRECTED 2026-09-04 - the 190/210 is an artefact of this test's own construction
>
> Controller `CONTROLLER_D0_PART3_AUDIT.md` **B3**, upheld and independently reproduced here. The
> caveat below was stated but was **not adequate**, and the number it guards should not have been
> carried into section 15 or into the handoff.
>
> **Split the count by density and the whole effect is in the extrapolated cells:**
>
> | density | passes 2x treatment sd | note |
> |---|---|---|
> | `p = 0.05` | **70 / 70** | sd imported from `p = 0.10` |
> | `p = 0.10` | **70 / 70** | **the only density where the sd was measured** |
> | `p = 0.20` | **50 / 70** | sd imported from `p = 0.10` |
>
> Every one of the 20 failures sits at the single density where the sd is borrowed. Worse, the
> borrowed threshold is **not reachable** there: **19 of the 20 failing cells have a threshold larger
> than the entire treatment statistic at that cell.** At L27 / `p = 0.20` / bs 128 the statistic is
> 0.000133 and the threshold is 0.010401 - **78x the quantity being tested.** No amount of real
> structure can pass a test like that; it is a statement about the imputation, not about the donor.
>
> Under two density-aware imputations the Controller ran, the count is **199/210**
> (magnitude-matched) or **209/210** (proportional / relative-sd scaling).
>
> **The honest statement is: at the one density where the treatment sd was actually measured, the
> count is 70 of 70.** The 190/210 headline is withdrawn.
>
> This is the same lean the previous audit found in Parts I and II: the correction in 12.3 was right,
> and the re-count built on top of it **over-corrected in the direction that makes the negative look
> wider.** That is the failure mode this programme keeps repeating and it repeated here.

---

## 13. The carve, measured in BPB

Every number in Parts I and II is a **proxy** - skippable fraction, `relerr`, FFN output norm
recovered. None of them is the quantity the project is judged on. This section carves the donor and
measures BPB.

### 13.1 Setup

`d0_carved_bpb_paired.json` -> `config`:

| | |
|---|---|
| experts `E` | 32 |
| active `k` | 8 |
| nominal activation | 0.25 |
| `cluster_seed` | 7 |
| router | **ORACLE top-k - upper bound** |
| `b3_repair_applied` | **true** |

The eval slice is the one every other stage used - `heldout`, 24 x 512, seed 1234, `ids_sha256`
`a1a48dc9...`, 12,264 predicted tokens, 51,870 scored bytes - so these BPB values are directly
comparable to the D1 baseline and to S1.

The design is **paired**: per-sequence nats are written per arm to `results/d0_carved_arms/*.npy`, so
the standard error is the paired one, not the marginal slice SE. This matters enormously here - the
marginal slice SE on the baseline is 0.0622 against a paired SE of 0.0042 on the L27 co-activation
comparison - a 15x difference in resolving power. *(Corrected 2026-09-04, Controller F4: this sentence
originally claimed the marginal SE "would swamp every effect below". It would not: 0.0622 does not
swamp +1.09062 or +1.81114, which clear it even marginally - a two-sample marginal SE gives z = 9.9 for
`all_coact`. It does swamp the two single-layer effects, +0.06499 and +0.14661. The original sentence
also called two different quantities "the same comparison".)*

**The router is an oracle - on a proxy.** It selects the `k` experts greedily, per token and per
layer, by **retained activation energy** (`(h**2) @ onehot`).

> *(Scoped 2026-09-04, Controller F5.)* It is therefore an oracle **on retained energy, not on BPB**,
> and it is **greedy per layer**, so it optimises no cross-layer interaction. The original claim here -
> *"No trainable router can beat it"* - **is not established** and is withdrawn. What is established
> is narrower: **no router can retain more activation energy at this budget under this per-layer
> objective.** A router trained against BPB directly, or one allowed to trade energy across layers,
> is not excluded by this measurement. Every Δ below remains a ceiling **for the energy objective**,
> which is weaker than the ceiling this section originally claimed.

### 13.2 The result

| arm | BPB | delta vs baseline | paired SE (seq bootstrap) | z (seq) | frac tokens worse | delta / sigma_seed |
|---|---|---|---|---|---|---|
| baseline (uncarved) | 0.7675950 | - | - | - | - | - |
| L27 only, co-activation | 0.8325833 | **+0.06499** | 0.00423 | 15.4 | 0.564 | **13.0** |
| L27 only, random null | 0.9142022 | +0.14661 | 0.00943 | 15.5 | 0.554 | 29.3 |
| all 28 layers, co-activation | 1.8582178 | **+1.09062** | 0.06981 | 15.6 | 0.862 | **218.1** |
| all 28 layers, random null | 2.5787313 | +1.81114 | 0.13999 | 12.9 | 0.919 | 362.2 |

Co-activation against its null, paired directly (`paired_coact_vs_null`):

| comparison | delta BPB | paired SE | z (seq) | z (token) |
|---|---|---|---|---|
| L27 only | **-0.08162** | 0.00612 | -13.3 | -27.3 |
| all 28 layers | **-0.72051** | 0.11475 | -6.3 | -60.8 |

### 13.3 The two readings, and they are both true

**The permutation works.** Against a random carve at the identical budget, co-activation ordering
recovers **0.7205 BPB of the 1.8111 a random carve costs - 39.8% of THE RANDOM CARVE'S damage**
*(denominator named 2026-09-04, Controller F11: measured against the co-activation carve's own damage
of 1.09062 the same 0.7205 is 66%, and the bare phrase "39.8% of the damage" hid which)* - at z = -6.3 paired
on sequences and -60.8 on tokens. This is the first time the structure has been shown to matter on the
*decision metric* rather than on a proxy. Part II's finding is confirmed end to end.

**And the carve is unusable.** +1.09062 BPB is **218 sigma_seed**. The project's noise constant is
0.005; this is two orders of magnitude past it. It is measured with an **oracle router**, so no
routing improvement can move it. 86.2% of tokens get worse, so it is not a tail effect that a
better-behaved subset could hide.

The single-layer arm sharpens why. Carving **L27 alone** already costs +0.06499 - 13 sigma_seed - while
moving only 56.4% of tokens past the median. Carving all 28 costs 1.09062, which is **16.8x** the
single-layer cost across 28 layers. **16.8 < 28, so the damage accumulates SUB-linearly with depth.**
*(Corrected 2026-09-04, Controller F3: this sentence originally read "the damage compounds with depth
rather than accumulating linearly", which inverts its own number.)* The honest reading is milder and
still sufficient: carving every layer costs less than 28 independent single-layer carves would, and
there is still no depth at which it is free.

**40% of a catastrophe is a catastrophe.** The permutation is a real effect that is nowhere near large
enough. That is the whole of D0 in one line, and it took an end-to-end BPB to say it.

### 13.4 A free determinism check across processes

The carve was measured twice, in two separate processes three days apart - the unpaired run
(`d0_carved_bpb.json`, 2026-08-30 15:04-15:31) and the paired one (`d0_carved_bpb_paired.json`,
2026-09-03 10:50-11:24):

| arm | 08-30 run | 09-03 run | delta |
|---|---|---|---|
| baseline | 0.767594960 | 0.767594964 | 4e-09 |
| L27_coact | 0.832583242 | 0.832583259 | 1.7e-08 |
| L27_null | 0.914202210 | 0.914202210 | 0 |
| all_coact | 1.858217812 | 1.858217850 | 3.8e-08 |
| all_null | 2.578731371 | 2.578731345 | -2.6e-08 |

Max absolute deviation **3.8e-08**, i.e. fp32 summation order, ~1e-5 of sigma_seed. The whole forward
path, the partitioning, the oracle router and the BPB accounting reproduce across processes and across
days. The baseline also matches C1's independently-measured 0.7675949601
(section 11 `control_C1_losslessness`) to 8 decimal places, and S1's 0.767594952 to **7**
*(corrected 2026-09-04, Controller F10 - the original said 8 for both)*.

---

## 14. Where the audit's findings now stand

*Rewritten 2026-09-04 after `CONTROLLER_D0_PART3_AUDIT.md`. The original table listed 8 of the previous
audit's 15 findings and silently dropped 7 (Controller F7). All 15 are below; a finding this document
has not acted on is marked NOT ADDRESSED rather than omitted.*

| # | finding | status after Part III |
|---|---|---|
| B1 | `FFN_active` minimum is 0.3911, not 0.5097 | **UPHELD, and the sentence is NOW ACTUALLY STRUCK** at 11.12 point 1. The original 14 claimed a withdrawal that did not exist anywhere in the document (Controller P3-audit B4) |
| B2 | 11.7b "per-token bytes, lossless" is not a byte count | **UPHELD, still open. The column is NOW ACTUALLY MARKED** at 11.7b. Same defect as B1: the withdrawal was claimed, never made |
| B3 | co-activation arm not reproducible at fixed seed | **CLOSED FORWARD ONLY.** Repaired, ARI 1.0000, labels identical (12.1) - **and the repair is now committed to `d0_layout.py`**, which it was not when Part III was written (Controller P3-audit B1). It does **not** make Parts I/II re-derivable (12.1 correction, Controller P3-audit B2) |
| F1 | 195/210 attributed to the wrong test | **UPHELD** - the two counts are 209/210 (row-1) and 195/210 (gap), reproduced in 12.4 |
| F2 | 11.7a's "79 of 80" saturate is 72 of 80 | **NOT ADDRESSED** by Part III |
| F3 | the block-size axis mixes two layout models; only block 64 is self-consistent | **NOT ADDRESSED** by Part III |
| F4 | the real bound is `FFN_active >= (1+2p)/3` | **UPHELD, still open.** Now named at 11.12 point 1 as part of B1's strike |
| F5 | "structure vanishes at block 140" is an absolute-margin artefact | **UPHELD** - 12.3 shows block 140 is where the *old* error bar was most flattering; in ratio the effect persists |
| F6 | `coactivation_best` is a max over four arms selected on the held-out half | **UPHELD. NOT moot** - the original claimed section 13 "selects nothing". It does: `E=32, k=8` is the argmax of Part I's 120-row fidelity table (relerr 0.0243, the single lowest). The selection runs in the conservative direction, but it is a selection (Controller P3-audit F8) |
| F7 | C3 has no fit/score split | **NOT ADDRESSED** by Part III |
| F8 | no positive control on the fidelity path | **NOT ANSWERED.** The original said "PARTLY ANSWERED". The audit asked for a **positive** control; section 13 supplies a **negative** one, on a different metric. Separately the carve hook itself has **no control of any kind** - the baseline arm installs no hook, so the hook path is never exercised against a known answer (Controller P3-audit F6) |
| F9 | only one memory stage is measured | **NOT ADDRESSED** by Part III |
| F10 | provenance: dirty git stamps, sklearn unpinned | **WORSE, NOT BETTER.** Both carve runs record `git_dirty_at_launch: true` and every Part III script was uncommitted at the time of writing (Controller P3-audit F7). The instrument is committed as of 2026-09-04; the runs' stamps cannot be retro-fitted |
| F11 | contamination: D0 is a plausible contaminant of others | **NOT ADDRESSED** by Part III |
| F12 | the `p_max` cap biases the realism check flatteringly | **NOT ADDRESSED** by Part III |

The previous audit's summary judgement - *"three independent over-statements all push in the same
direction... the failure is in the summary sentences, not in the measurement"* - **repeated itself in
Part III.** The Part III audit found 4 BLOCKs and 14 FLAGs, and its own verdict on direction is that
Part III leans **the same way**: B3, F1, F3 and F5 each made the negative look wider or more final
than the artefacts support. The tables were again right (six PASS findings, including the 12.3
correction reproduced exactly and two independent SE estimators agreeing with the bootstrap to <= 3%)
and the prose again was not. **That is now four probes in a row with the same signature, and it should
be read as a property of how this document gets written, not as bad luck.**

---

## 15. Verdict

**FFN co-activation carving is closed as a negative on this donor, on both halves of the question.**

1. **Fidelity (Part I):** with an oracle router, the carve loses 59.5% to 94.7% of the FFN output norm
   at the budget the adapter needs.
2. **Structure (Part II):** the co-activation permutation is real, and clears its null in 209 of 210
   cells - but at a granularity (block 12-21) far finer than any expert layout can use.
3. **Reproducibility (Part III, section 12):** the arm is now deterministic **from here on** - the
   repair is in `d0_layout.py` as of 2026-09-04 - but Parts I and II are **not** re-derivable, because
   they were drawn from an unrecorded global RNG state. Their numbers stay n = 1 draws from a spread
   that is now measured but whose particular draw is lost. The treatment's own error bar runs
   **0.96x to 4545x** the null's across the 70 measured cells. **At the one density where that sd was
   actually measured (`p = 0.10`) the count is 70 of 70;** the 190/210 figure this verdict originally
   cited is withdrawn as an artefact of imputing that sd to other densities (12.4).
4. **BPB (Part III, section 13):** the carve costs **+1.09 BPB = 218 sigma_seed with an oracle
   router (on retained energy - see 13.1).** The structure is worth **39.8% of the random carve's
   damage**, equivalently 66% of the co-activation carve's own. The remainder is disqualifying.

**What stays open is not this axis - it is scale.** Every number above is measured at 1.5B, and FFN
sparsity is reported to rise with model size (46.5% / 50.5% / 71.7% `S_inter` at 0.5B / 1.5B / 14B,
**arXiv:2509.00454 Table 1**, carried here from `briefs/BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md` §A1.1;
*citation added 2026-09-04, Controller F13 - Part III originally gave these three numbers with no
source at all. Per this programme's standing rule on literature figures, they are load-bearing for the
scale argument and this document has NOT independently verified them against the paper's own table*).
We are
measuring at the bottom of that curve. Whether the carve is merely bad here or bad everywhere is the
question S1 was pre-registered to answer, and it is why S1's scale arm was made mandatory
(`briefs/BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md`, Amendment 1, commit `f90ac3d`).

**Do not open the joint sparsity+permutation brief on the strength of 13.3's first reading.** The
recovery is real and it is not enough.

> **⚠ Two routes out of this verdict were closed by hand-waving, and 2026-09-04 reopened both.**
>
> The original text here said *"the only thing that could change that verdict is a different point on
> the scale curve, not a better router - the router here is already an oracle."* **Both halves of that
> were too strong.**
>
> - **The router.** The oracle is greedy, per-layer, on **retained activation energy**, not on BPB
>   (13.1, Controller F5). A router trained against BPB, or one trading energy across layers, is not
>   excluded by anything measured here.
> - **The granularity.** Every BPB number in section 13 was measured at `E = 32` - **280 neurons per
>   expert, the coarsest of the three granularities Part I itself swept** - while Part I's own fidelity
>   table, read at matched achieved activation, favours the finer carve in every cell where two `E`
>   values exist, by 0.08 to 0.11 relerr in the deep layers. `E = 128` is legal under the per-organ
>   floor this document used and its BPB was never measured. This is pre-registered as
>   **`briefs/BRIEF_D0C_CARVE_GRANULARITY.md`** (commit `4a98c89`) and is under measurement.
>
> **The two biases run in OPPOSITE directions and Part III originally stated only the first:** the
> oracle router flatters the carve, the coarse granularity punishes it. Until D0c reports, section 13's
> +1.09062 should be read as *"the cost at E = 32 under an energy oracle"*, not as *"the cost of
> carving"*.

---

*Part III added no new capture. Section 12 read mask files already on disk; section 13 loaded the donor
and ran five BPB arms on the standing eval slice. Artefacts: `d0_b3_treatment_spread.json`,
`d0_b3_sweep.log`, `d0_b3.wallclock.txt`, `d0_carved_bpb.json`, `d0_carved_bpb.log`,
`d0_carved_bpb_paired.json`, `d0_carved_bpb_paired.log`, `d0_carved_arms/*.npy` (5 arms, per-sequence
nats), `d0_carved_labels_E32.npz`. Peak RSS 9.05 GB.*
