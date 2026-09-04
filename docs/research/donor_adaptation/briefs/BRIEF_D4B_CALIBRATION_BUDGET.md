# BRIEF D4b — Is 0.483 recovery a method ceiling, or a sample-size artefact?

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-03.**
**Depends on: `probes/D4_RECONSTRUCTION.md`, `results/d4_reconstruction.json`.**

---

## 1. The question

D4 solves, layer by layer, for replacement weights that make a structured-sparse organ reproduce the
dense organ's output. Its headline is `recovery = 1 − Δ_reconstructed / Δ_naive`, and on the six
column-structured points it means recovery of **0.4452**.

**It was measured at one calibration budget: `calib_tokens_T = 16384`. That knob was never swept.**

D4's own leakage control is the reason this matters. Solving `H` on the *eval* slice recovers
**0.8590** where the honest solve recovers **0.4827** — **on fewer tokens** (12,288 vs 16,384), drawn
i.i.d. from the same shuffled corpus. There is no distribution mismatch for "better data" to fix.
Whatever separates 0.483 from 0.859 is **in-sample optimism**, and the size of that optimism is set
by how many samples the solve has per unknown.

That ratio is small, and it has never been reported. The solve is over the **surviving** columns `S`:

| organ | level | surviving `S` | **T / S** | recovery | note |
|---|---|---|---|---|---|
| down_proj | 25% | 6720 | **2.44** | **0.1217** | the only **INCONCLUSIVE** point; worst-conditioned (cond max 1.32e8; 1783/6720 directions regulariser-dominated at L1) |
| down_proj | 50% | 4480 | **3.66** | 0.4827 | the reference point; the only point with the full ablation battery |
| down_proj | 75% | 2240 | 7.31 | 0.3635 | |
| o_proj | 25% | 1152 | 14.22 | **0.8850** | the only **LARGE** point |
| o_proj | 50% | 768 | 21.33 | 0.4180 | |
| o_proj | 75% | 384 | 42.67 | 0.4001 | |

**The two lowest-`T/S` points are the worst-behaved point and the reference point; the highest
recovery in the probe sits at `T/S` = 14.2.** That is suggestive and it is **confounded** — organ
identity and sparsity level move with `T/S` across these rows, and the relation is not monotone
(`T/S` = 42.67 returns only 0.400). **Confounded evidence is exactly why this brief specifies an
intervention on `T` at a fixed (organ, level) rather than another cross-sectional reading.**

> **The question this brief pre-registers:**
> **Does honest recovery rise materially with calibration budget? If it does, 0.483 is a budget we
> chose, not a ceiling the method has.**

## 2. Why this is worth the compute

This is the second time in one day that a closed negative in this programme turns out to have been
measured at **one extreme of a free parameter that was never swept** — D0c records the first (the
BPB carve was measured only at the coarsest of the three granularities D0 itself swept). Naming the
pattern is not a result, but it is a reason to check before closing an axis:

> **A negative is only as strong as the sweep behind the setting it was measured at.**

D4 is the programme's live route to the goal — running someone else's pretrained model on our
architecture by *solving* for thin replacement weights rather than by carving. Whether its central
number is a ceiling or a budget decides whether that route is closed or merely under-resourced.

## 3. Arms

Fixed at **`down_proj`, 50% structured sparsity** — the reference point, the only one carrying the
full ablation battery, and `T/S` = 3.66. Everything else identical to D4: same donor
(Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`), same eval slice, same
`delta_naive` read verbatim from `d1_pruning.json`'s `block_structured` record
(**1.5571150213992855**), same `LAMBDA_SCALE = 1e-4`, same paired bootstrap (`n_boot = 2000`).

| arm | `T` | `T/S` | what it is |
|---|---|---|---|
| **B0** | 16384 | 3.66 | **replication.** Must reproduce recovery **0.4827** (SE 0.0437), `Δ_recon` **+0.80546** |
| **B1** | 32768 | 7.31 | primary |
| **B2** | 65536 | 14.63 | primary — matches the `T/S` of the probe's one LARGE point |
| B3 | 131072 | 29.26 | run only if B1→B2 has not plateaued (see §4) |
| **L0** | 16384 | — | **leakage control at B0's budget.** Must reproduce recovery **0.8590** |
| **L1** | 32768 | — | leakage control at B1's budget |
| **L2** | 65536 | — | leakage control at B2's budget |

### 3.1 ⚠ THE CONTROL THAT MAKES THIS READABLE — non-negotiable

**The leakage arm is re-run at every budget.** The claim under test is that the honest–leak gap is
in-sample optimism. If that is what it is, **the gap must close as `T` grows.** Reporting honest
recovery alone cannot distinguish "the solve got better" from "the whole instrument drifted".

> The deciding quantity is **`gap(T) = recovery_leak(T) − recovery_honest(T)`**, not
> `recovery_honest(T)` on its own. At `T` = 16384 that gap is **0.3763**.

A second, free control: **report `n_below_lambda`** (directions where the regulariser dominates) at
every budget. At `T` = 16384, `down_proj@50%` has mean 31.8 / max 766 of 4480. If budget is the
binding constraint this count must fall with `T`. If it does not, the regulariser is binding for a
reason that more data cannot fix, and that is a different finding worth having.

### 3.2 Calibration/eval disjointness is not optional and must be re-asserted at every `T`

D4 carries `calib_eval_disjointness` with `verified = true`. Enlarging the calibration draw is
exactly the operation that can silently break it. **Re-assert and record the assertion at every
budget.** If the corpus cannot supply 131072 disjoint calibration tokens, **run B0-B2 and report
that B3 was not affordable** — do not reach into the eval half. An honest gap beats a contaminated
point; this programme has said so before and it applies with full force here.

## 4. Pre-registered decision rule — fixed before any result

Let `r(T)` = honest recovery, `gap(T)` = leak − honest. Reference: `r(16384) = 0.4827`,
`gap(16384) = 0.3763`. Bands use 2×SE per D4's own `inconclusive_rule`.

| outcome | condition | what it means |
|---|---|---|
| **BUDGET-BOUND** | `r(65536) ≥ r(16384) + 0.15` **and** `gap(65536) ≤ gap(16384) − 0.15` | 0.483 is a budget, not a ceiling. D4's headline must be restated as a `T`=16384 statement, and the reconstruction route is **under-resourced, not closed** |
| **PARTIAL** | `r` improves by ≥ 0.15 but `gap` does not close by ≥ 0.15 | more data helps for a reason that is not the overfit story; the mechanism claim does not strengthen |
| **METHOD-CEILING** | `\|r(65536) − r(16384)\| < 0.05` **and** `gap` closes by < 0.05 | 0.483 is what this method does at this operating point. **D4's negative HARDENS**, and the 0.859 leak figure is irreducible in-sample optimism rather than a target |
| **WORSE** | `r(65536) < r(16384) − 0.05` | more calibration data hurts; say so, and the solve or its regularisation is misspecified |
| **INCONCLUSIVE** | any band spans the nearest threshold | report as inconclusive; do not pick the nearer label |

**Run B3 (`T` = 131072) if and only if** `r(65536) ≥ r(32768) + 0.05`, i.e. the curve has not
plateaued by B2. If it has plateaued, B3 adds nothing and must not be run to chase a number.

**0.15 recovery units** is the threshold because the whole honest–leak gap is 0.376: an effect worth
reopening a written-up probe must be a substantial fraction of the gap it claims to explain, not a
detectable sliver. **These thresholds are fixed here, before the run, so they cannot be tuned to the
answer** — and that protection holds regardless of what the figure running the arms happens to
believe.

**No extrapolation.** The curve may not be extended past the largest `T` actually run, in prose, in a
table or in a figure.

## 5. What this brief does NOT claim and does NOT test

- It does **not** claim reconstruction is viable. Even D4's best point (`o_proj@25%`, recovery 0.885)
  leaves **+0.0961 BPB = 19.2 σ_seed** while zeroing only **1.07%** of the model's weights. A
  budget-bound outcome moves a number; it does not by itself make the route work.
- It does **not** sweep `LAMBDA_SCALE`, which is also unswept and varies 734× across depth within
  `down_proj` alone. If B-arms come back BUDGET-BOUND, λ is the next knob, not this brief's.
- It does **not** test any organ but `down_proj`, any level but 50%, or any size but 1.5B. **Nothing
  in D4 has ever been measured at another size** — the same gap that forced S1's Amendment 1.
- It does **not** supply the positive control D4 lacks. There is still no arm where an
  information-free `H` yields recovery ≈ 0 with a well-behaved solve, and this brief does not create
  one.

## 6. Cost

Hessian accumulation is linear in `T`; the solve is unchanged. B0-B2 plus L0-L2 is roughly `(1+2+4)`
× the honest pass and the same again for leak ≈ **14 calibration-passes-equivalent**, against D4's
original 1. On the machine that ran D4 at 10 threads this is hours, not minutes, and it is **third in
the CPU queue** behind S1 (PID 17884) and D0c. **Do not start it while either is running.**

## 7. Reporting

`probes/D4B_CALIBRATION_BUDGET.md`, or a Part II appended to `D4_RECONSTRUCTION.md` if that report is
not under audit at the time. Report: `r(T)`, `gap(T)`, `n_below_lambda(T)` and the conditioning
summary at every budget, each with its band; the disjointness assertion at every budget; the outcome
label from §4 verbatim; and a restatement of D4's verdict in its light — including, if §4 lands on
METHOD-CEILING, an explicit note that D4's negative is now **stronger** than its report stated,
because it survived a budget control.
