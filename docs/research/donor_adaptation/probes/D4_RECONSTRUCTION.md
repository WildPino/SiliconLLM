# D4 - Hessian-weighted layer-wise reconstruction on a real donor

**Probe:** D4 (density programme). **There is no `BRIEF_D4_*.md`.** The pre-registration exists only
as the `prereg` key inside the artefact itself, written in `main()` before any number is computed.
**Instruments:** `benchmarks/donor_adaptation/density/d4_reconstruction.py` (749 lines),
`d4_finish.py` (326 lines). A third script, `d4_conditioning_and_nulls.py` (396 lines), is committed
but **was never run** - see §2.3.
**Artefact:** `benchmarks/donor_adaptation/density/results/d4_reconstruction.json` (89 KB, committed
in `0d69006`, 2026-08-22).
**Written by:** the Builder - **Date:** 2026-09-03. The run itself finished 2026-08-22 and had no
probe report until this one.
**Status:** **COMPLETE RUN, all 9 pre-registered sweep points and all 4 control groups present.**

**One cross-probe anchor, on the same donor and the same baseline.**
`BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md:161` records D0's carved-model result as **+1.09 BPB
(0.7676 -> 1.8582)** at 1.5B, 28 layers, 25% active, oracle router. D4's baseline is the same
0.7675949677540373. Every D4 point in §4.1 is measured against that same number, so the two probes'
deltas are directly comparable - see §4.4 for where D4's nine points land.

**Naming collision, stated once:** `decisions/DONOR_V2_DENSITY_PROGRAMME.md:349` uses the label
"D4" for a *different* probe ("Can a small core learn to drive a frozen donor FFN at all?", budgeted
at 1 T4 session). The D4 written up here is the density-probe D4: Hessian-weighted layer-wise
reconstruction, CPU-only, no GPU session spent. They are not the same experiment.

---

## 0. Read this first: what `recovery` is, and what it is not

D1 measured what happens when you block-mask an organ's weights and change nothing else:
catastrophic at every organ and level. D4 applies **exactly one variable** on top of D1's own masks -
it solves, per layer, for replacement weights on the surviving support:

    minimise || W' X - W X ||_F^2  subject to W' having support M

with `H = X^T X` from a calibration set, which is row-separable and closed-form (`d4_reconstruction.py`
module docstring, lines 10-21).

D4's pre-registered primary quantity is

    recovery = 1 - delta_reconstructed / delta_naive

**`recovery` is a fraction whose denominator is D1's catastrophe.** A recovery of 0.885 is not "the
model is 88.5% recovered"; it is "88.5% of an enormous loss was removed, and whatever remains is
still whatever it is". Every table in this report therefore carries the **residual `delta` in BPB**
and in multiples of **sigma_seed = 0.005 BPB** next to every recovery figure. Read those columns
first.

Three of the nine sweep points are **theorems, not measurements** (§4.2). The artefact says so
before it runs them, and excludes them from its own summary. This report reports all nine and marks
which is which.

---

## 1. The pre-registration, quoted

From `d4_reconstruction.json["prereg"]`, verbatim:

| key | text (verbatim) |
|---|---|
| `primary_quantity` | "recovery = 1 - delta_reconstructed / delta_naive, per (organ, sparsity) point; delta_naive read verbatim from d1_pruning.json's block_structured record for that exact (organ, level)." |
| `se_method` | "paired bootstrap (n_boot=2000, >= 10 required) of delta_reconstructed over sequence resampling on the SAME 24x512 heldout slice D1 used, propagated through the recovery ratio holding delta_naive fixed at its D1 point value (D1's own paired_se on delta_naive is reported alongside per point, not re-bootstrapped jointly -- assumption stated here, not hidden)." |
| `inconclusive_rule` | "a point's recovery is INCONCLUSIVE iff its [point - 2.0*SE, point + 2.0*SE] interval spans 0." |
| `magnitude_bands_fixed_before_any_result.LARGE` | "recovery point estimate >= 0.70 AND the lower band bound > 0.5" |
| `magnitude_bands_fixed_before_any_result.MODERATE` | "0.30 <= recovery point estimate < 0.70 (and not inconclusive)" |
| `magnitude_bands_fixed_before_any_result.NEGLIGIBLE` | "recovery point estimate < 0.30 (and not inconclusive)" |
| `magnitude_bands_fixed_before_any_result.INCONCLUSIVE` | "band spans 0, regardless of point estimate" |
| `scope` | "gate_proj, o_proj, down_proj (one MLP-expansion/ROW-structured, one attn-output/COLUMN-structured, one MLP-contraction/COLUMN-structured) at sparsity {0.25, 0.50, 0.75}. 0.90 excluded per brief..." |
| `control3_control4_scope_decision` | "Hessian ablation (control 3) and leakage control (control 4) are run on down_proj@0.5 only, a time-budget decision stated up front... o_proj's sweep numbers stand on real-H only and are NOT independently ablated -- flagged in the report." |

**The `se_method` assumption, kept visible as the prereg demands:** the reported `recovery_se`
propagates only the bootstrap uncertainty of `delta_reconstructed`. `delta_naive`'s own uncertainty
(D1's `paired_se`, carried in each record as `delta_naive_d1_paired_se`) is **not** propagated. The
ignored term is largest, in relative terms, exactly where the denominator is smallest - at
`down_proj@25%`, whose `delta_naive_d1 = 0.44314865475346077` carries
`delta_naive_d1_paired_se = 0.019649008156453353`, i.e. 4.4% relative. That is the one point the
run classifies INCONCLUSIVE, so the omission bites precisely at the boundary the classification
turns on. This is a limitation of the pre-registered SE, not an error in applying it.

### The mask-geometry derivation, quoted in full

`prereg.derivation_note_mask_geometry`, verbatim:

> "D1.\_struct_axis makes q/k/v/gate/up_proj ROW-structured (whole output neurons zeroed) and
> o_proj/down_proj COLUMN-structured (whole input features zeroed, same surviving column set S
> shared across every output row). The row-separable SparseGPT-style solve in this brief only has
> freedom to act when S varies WITHIN a row's support -- i.e. only for COLUMN-structured organs.
> For gate_proj (ROW-structured), every kept row already has full support (S = all columns) and
> every dropped row has none: the closed form is therefore mathematically forced to return exactly
> W (kept rows) or exactly 0 (dropped rows), independent of H's content. This is stated here,
> before gate_proj is run, not discovered after."

**A reader who skips this will misread the table.** gate_proj's three `recovery = 0.0` rows are not
evidence that reconstruction fails on MLP-expansion organs. They are the arithmetic identity
`W' = W` on kept rows, `W' = 0` on dropped rows, re-measured end to end. §4.2 gives the empirical
confirmation, which is bit-exact.

---

## 2. The instrument, and the three sessions it took

### 2.1 What ran, in order

| session | script | log | what it produced |
|---|---|---|---|
| 1 | `d4_reconstruction.py` | `results/d4.log` | **ABORTED.** Identity control failed. Preserved as `d4_reconstruction.ABORTED_session1.json` |
| 2 | `d4_reconstruction.py` (fixed) | `d4_run.log` | baseline, rank diagnostics, all 4 control groups, 6 of 9 sweep points (gate_proj x3, o_proj x3). Killed before down_proj |
| 3 | `d4_finish.py` | `d4_finish.log` | the 3 down_proj sweep points, the gate_proj relabel, the `wrong_layer_H` arm, the conditioning diagnostic. Wrote `done_finish: true`, `finish_wallclock_seconds: 2831.1467711925507` |

### 2.2 Session 1 is the positive-control evidence, and it is the strongest thing in this probe

`d4_reconstruction.ABORTED_session1.json["controls"]["identity"]`: at mask = 0% - where
reconstruction must return the weights unchanged - the run recorded
`bpb = 1.1168092754215935`, `delta = 0.3492143076675561`,
`max_abs_weight_deviation = 0.6545197367668152`, `fired: false`.

`ABORTED_BY_PRINCIPAL.reason` in the same file names two causes: (1) a regularisation asymmetry -
`H[S,S]+lambda*I` on the system matrix but the unregularised `H[:,S]` on the right-hand side, so
even at `S = full` the closed form computed `W @ H @ (H+lambda*I)^-1 != W`; and (2) `T = 4096`
calibration tokens against `d_ffn = 8960`, giving `rank(H)/N = 4096/8960 = 0.457`, below the 0.5
Failure-Mode-1 threshold. The record marks itself `this_record_is_evidence_do_not_delete: true`.

**This is the project's standing law satisfied in the only way that counts:** the identity control
did not merely pass in the final run, it *fired on a real, previously-unknown defect* and stopped
the probe. `d4_reconstruction.py:171-198` carries the fix and the docstring explaining it.

### 2.3 One committed script never ran

`d4_conditioning_and_nulls.py` writes `log["scope_reality_check"]` at line 373. That key **is absent
from `d4_reconstruction.json`** (top-level keys are: `git_revision, git_branch, command_line,
threads, prereg, arch, hardware, eval_slice, calib_slice, calib_eval_disjointness, baseline_bpb,
baseline_reproduces_d1_exactly, block_sizes, calib_tokens_T, rank_diagnostics, controls,
controls_summary, organ_sweep, organ_sweep_summary, conditioning_diagnostics, done_finish,
finish_wallclock_seconds`). Its docstring states it would have added a **second fully-ablated point,
`down_proj@75%`**, against real / shuffled / wrong-layer H. That second ablated point does not exist
in the artefact. `d4_finish.py` implemented the conditioning diagnostic and the wrong-layer arm
instead, at `down_proj@50%` only.

---

## 3. The controls, reported before the results

All four control groups are on `down_proj` (`prereg.control3_control4_scope_decision` states this
as a time-budget decision, up front).

### 3.1 C1 - identity (mask = 0%)

| field | value | source |
|---|---|---|
| `bpb` | 0.7675949736956782 | `controls.identity.bpb` |
| `delta` | 5.941640845996687e-09 | `controls.identity.delta` |
| `paired_se` | 9.033640019032512e-09 | `controls.identity.paired_se` |
| `max_abs_weight_deviation` | 2.9802322387695312e-08 | `controls.identity.max_abs_weight_deviation` |
| `fired` | **true** (test: `abs(delta) < 0.001`, `d4_reconstruction.py:526`) | `controls.identity.fired` |

The residual `delta` of 5.94e-09 BPB is **six orders of magnitude below sigma_seed = 0.005**. The
weight deviation 2.98e-08 is fp32 round-off. Compare the same field in session 1: 0.6545. The
control moved by 7 orders of magnitude between a broken solve and a fixed one.

**Not measured:** the identity control was run on `down_proj` only.
`controls.identity.o_proj_note` states o_proj was not separately measured (time budget);
`controls.identity.gate_proj_note` states gate_proj at 0% invokes no solve at all, so its deviation
is 0 by construction and was not run as an eval.

### 3.2 C2 - saturation (mask = 100%)

| field | value | source |
|---|---|---|
| `bpb` | 4.5128979664456885 | `controls.saturation.bpb` |
| `delta` | 3.7453029986916513 | `controls.saturation.delta` |
| `paired_se` | 0.16723957321713498 | `controls.saturation.paired_se` |
| `fired` | **true** (test: `delta > 0.5`, `d4_reconstruction.py:554`) | `controls.saturation.fired` |

`controls.saturation.note` states this single eval serves as both the naive-100% and the
reconstructed-100% arm, because `S` is empty for every row at full block-column removal, so the two
are the same operation; it also states D1 logged no 100% point for `down_proj`, so nothing external
exists to compare against and **the self-consistency is the control**. That is a weaker claim than
C1's: the arm establishes the eval path responds to total destruction of the organ (+3.745 BPB), not
that reconstruction was correctly disabled.

### 3.3 C3 - the Hessian ablation, four arms, all at `down_proj@50%`

`delta_naive_d1` for this point = 1.5571150213992855 (`controls.hessian_ablation.delta_naive_d1`,
matching `d1_pruning.json`'s `down_proj/block_structured/50%` record exactly).

| arm | `bpb` | `delta` | `paired_se` | `delta`/sigma_seed | `recovery_point` |
|---|---|---|---|---|---|
| `real_H` | 1.5730582771923323 | +0.8054633094382949 | 0.06897565928813412 | 161.1 | **+0.48272073779464697** |
| `identity_H` | 2.3247099891533227 | +1.5571150213992855 | 0.05796843399937317 | 311.4 | **0.0** |
| `shuffled_H` | 4.8085744267031460 | +4.0409794589491080 | 0.10216506853237914 | 808.2 | **-1.5951708148815644** |
| `wrong_layer_H` | 4.6169918063702795 | +3.8493968386162423 | 0.11776385583944751 | 769.9 | **-1.4721339051478814** |
| *(C2 saturation, for reference)* | 4.5128979664456885 | +3.7453029986916513 | 0.16723957321713498 | 749.1 | *(n/a)* |

**What each arm proves, one at a time:**

- **`real_H`** is the treatment, not a control. Its recovery +0.4827 is the number the sweep row for
  `down_proj@50%` also reports (§4.1) - see §7.3, they agree to all 16 digits across two processes.

- **`identity_H` is a tautology, not a null.** Set `H = I`; then `H_reg = (1+lambda)I`,
  `A = (1+lambda)I_{|S|}`, `rhs = (1+lambda)W[:,S]`, so `sol = W[:,S]` - the solve returns the naive
  mask, for any mask, independent of any data. The artefact confirms this to the last bit: this
  arm's `delta = 1.5571150213992855` and `paired_se = 0.05796843399937317` are **numerically
  identical, digit for digit**, to `d1_pruning.json`'s `down_proj/block_structured/50%` record
  (`delta = 1.5571150213992855`, `paired_se = 0.05796843399937317`). It therefore contributes
  **zero evidence about Hessian quality**. What it does contribute, unplanned but real, is proof
  that **D4's recomputed mask equals D1's mask at this point** - see §7.4.

- **`shuffled_H`** (`d4_reconstruction.py:155-168`) permutes the token axis per feature
  independently. Its own docstring claims this preserves each feature's marginal while sending
  off-diagonals to ~0. `audits/CONTROLLER_DENSITY_AUDIT.md:510` filed a BLOCK against that claim -
  that with nonzero column means (near-certain for a SwiGLU intermediate) the off-diagonal does not
  vanish, so this arm is not the clean "H carries no usable information" null it is used as. **What
  the row shows is not a null at all: recovery = -1.5952.** The solve did not degrade to no-op; it
  applied a confidently wrong correction and made the model worse.

- **`wrong_layer_H`** was added in session 3 to answer that BLOCK with a stronger null.
  `controls.hessian_ablation.wrong_layer_H.construction`: "layer L reconstructed using layer
  (L+14 mod 28)'s real H -- real spectrum and real column means (unlike shuffled_H), wrong layer
  correspondence (unlike real_H). offset=14 is a perfect derangement on 28 layers: no layer ever
  uses its own H." Its recovery is **-1.4721**, an anti-null too.

**The separations, computed from the rows above:**

| comparison | difference in `recovery_point` |
|---|---|
| `real_H` - `shuffled_H` | 0.48272 - (-1.59517) = **+2.0779** |
| `real_H` - `wrong_layer_H` | 0.48272 - (-1.47213) = **+1.9549** |
| `shuffled_H` - `wrong_layer_H` | -1.59517 - (-1.47213) = **-0.1230** |

And on the `delta` scale, with a quadrature combination of the two arms' `paired_se` (**an
approximation**: these SEs are paired against the baseline, not against each other; the ablation
arms carry a `recovery_point` but **no `recovery_se` of their own**, and the artefact records no
arm-to-arm paired bootstrap - so treat these z-values as sanity checks computed by this report, not
as statistics the run itself produced):

| comparison | difference in `delta` (BPB) | quadrature SE | z |
|---|---|---|---|
| `real_H` vs `identity_H` | -0.75165 | 0.09010 | -8.34 |
| `shuffled_H` vs `identity_H` | +2.48386 | 0.11747 | +21.15 |
| `wrong_layer_H` vs `identity_H` | +2.29228 | 0.13126 | +17.46 |
| `shuffled_H` vs `wrong_layer_H` | +0.19158 | 0.15590 | +1.23 |
| `shuffled_H` vs C2 `saturation` | +0.29568 | 0.19598 | +1.51 |
| `wrong_layer_H` vs C2 `saturation` | +0.10409 | 0.20454 | +0.51 |

**Three readings, each naming its rows:**

1. **The mechanism uses the donor's own per-layer activation correlations.** `real_H` (+0.80546)
   beats `identity_H` (+1.55712) by 0.75165 BPB at z = -8.34, and beats both wrong-H arms by more
   than 2 BPB. A least-squares refit that ignored H entirely would land on `identity_H`'s row, which
   is by derivation the naive mask. It did not.

2. **A wrong H is not ignorance-tolerant.** `shuffled_H` (+4.04098) and `wrong_layer_H` (+3.84940)
   are both *worse* than `identity_H` (+1.55712), i.e. worse than not reconstructing at all, at
   z = +21.15 and +17.46. The solve does not fall back to safety when its H is wrong; it applies a
   confident correction in the wrong direction.

3. **But "worse than deleting the organ outright" is NOT established by these rows.**
   `shuffled_H` (+4.04098) exceeds C2 `saturation` (+3.74530) by only 0.296 BPB at z = +1.51, and
   `wrong_layer_H` (+3.84940) exceeds it by 0.104 at z = +0.51. Both are consistent with landing at
   the saturation level. The honest statement is: **a wrong-H reconstruction lands in the same
   region as removing `down_proj` entirely**, not demonstrably beyond it.

**Does the four-arm set constitute a genuine positive control? No - and the gap is specific.**
The set has one tautological arm (`identity_H`), two anti-null arms (`shuffled_H`,
`wrong_layer_H`), and the treatment (`real_H`). **There is no arm in which a genuinely
information-free H produces a recovery near 0 while the solve stays well-behaved.** So the set
establishes *H-specificity* (reading 1) and *failure asymmetry* (reading 2). It does not establish
what fraction of `real_H`'s +0.4827 would survive if H carried only generic activation scale rather
than this layer's correlations - because no arm isolates that. The instrument's demonstrated
firing-on-a-known-positive lives in C1 and session 1 (§2.2), not in C3.

### 3.4 C4 - leakage, and `leakage_looks_better_than_honest = true`

| field | value | source |
|---|---|---|
| `bpb` | 0.987097467038977 | `controls.leakage.bpb` |
| `delta` | 0.2195024992849397 | `controls.leakage.delta` |
| `paired_se` | 0.016284498195784543 | `controls.leakage.paired_se` |
| `recovery_point` | **0.8590325722452501** | `controls.leakage.recovery_point` |
| `compare_to_honest_real_H_delta` | 0.8054633094382949 | `controls.leakage.compare_to_honest_real_H_delta` |
| `leakage_looks_better_than_honest` | **true** | `controls.leakage.leakage_looks_better_than_honest` |

This arm builds `H` from the **eval slice itself** (`d4_reconstruction.py:627-641`) - the thing an
honest run must never do. It recovers **0.8590** where the honest `real_H` recovers **0.4827**.

**What that means, and it must not be buried.** The naive reading - "better calibration data would
get us to 0.86" - is wrong, and the artefact's own slice records say why:

- `calib_slice.corpus_sha256 = 10d4d281...` and `eval_slice.corpus_sha256 = f46b0310...` are two
  different physical files. `corpus/manifest.json` records `n_chunks: 35802` split
  `calib: 17901` / `heldout: 17901` from one globally-shuffled 8 KiB-chunk pool, seed 20260821.
  The two halves are **i.i.d. draws from the same distribution.** There is no distribution mismatch
  for better data to fix.
- The leakage H was built from **fewer** tokens, not more: `rank_diagnostics.down_proj_leakage_H.T_tokens
  = 12288` (24 x 512) against the honest `calib_tokens_T = 16384` (32 x 512). Against
  `d_ffn = 8960` that is T/D = 1.371 for the leak arm versus 1.829 for the honest arm.

**So the 0.4827 -> 0.8590 gap cannot be attributed to calibration budget or to domain mismatch; on
these rows it is the solve fitting the specific eval sample.** The honest number is the honest
number, and the +0.376 recovery units the leak arm shows is an upper bound on how much of an
apparent gain a leaky pipeline could manufacture on this slice. Read the other direction: the
honest `real_H` solve at T/D = 1.83 is **overfitting its thin calibration sample**, and roughly
0.38 recovery units of what a leak-free-looking pipeline might report would be that overfit.

### 3.5 Controls scorecard

| control | recorded flag | is it a real test? | this report's reading |
|---|---|---|---|
| C1 identity, mask 0% | `fired: true` | **yes, the strongest in the probe** | **FIRED.** `delta = 5.94e-09` BPB, 1e-6 of sigma_seed; `max_abs_weight_deviation = 2.98e-08`. And in session 1 the same control **caught a real bug** (`delta = +0.34921`, deviation 0.6545) and aborted the run. Demonstrated firing on a known positive. Measured on `down_proj` only |
| C2 saturation, mask 100% | `fired: true` | **partly.** Self-consistent by construction | FIRED at `delta = +3.74530`. `controls.saturation.note` states there is no D1 100% record for `down_proj` to compare against, so this is a response check on the eval path, not an external cross-check. Measured on `down_proj` only |
| C3a `identity_H` | reported as an ablation arm | **no - tautology** | Algebraically forced to reproduce naive masking. `delta` and `paired_se` bit-identical to D1's record. Zero Hessian-quality evidence; **repurposed here as a mask-equality proof (§7.4)** |
| C3b `shuffled_H` | reported as the information-free null | **no - it is an anti-null** | `recovery = -1.5952`. `CONTROLLER_DENSITY_AUDIT.md:510` BLOCKed the "off-diagonal -> 0" premise; the row confirms the arm is not a null of any kind |
| C3c `wrong_layer_H` | added session 3 to strengthen C3b | **yes for H-specificity, no as a null** | `recovery = -1.4721`. Real spectrum, real column means, wrong layer. Establishes correspondence matters; does not isolate "any real structure would do" |
| C3d `real_H` | the treatment | n/a | +0.48272, reproduced bit-exactly by the independent sweep run in session 3 |
| C4 leakage | `leakage_looks_better_than_honest: true` | **yes, and it is informative** | FIRED. 0.8590 vs 0.4827 on **fewer** tokens from an i.i.d. corpus half -> the honest solve is overfitting its calibration sample by ~0.38 recovery units |

**Does the instrument demonstrably fire on a known positive? Yes - C1, in session 1, on a real
defect it found itself.** The Hessian ablation battery (C3) does *not* contain a clean
information-free null, and that is the single largest gap in D4's control set.

---

## 4. The organ sweep - all nine points

### 4.1 The table

`delta_naive_d1` in every row was checked against `d1_pruning.json`'s `block_structured` record for
that exact (organ, level): **9 of 9 match verbatim**, as the prereg requires. Every `recovery` was
recomputed from `1 - delta/delta_naive_d1` and agrees with the recorded value to better than 1e-12.
Every band was re-derived from `prereg.magnitude_bands_fixed_before_any_result` and agrees with the
recorded verdict (the three gate_proj rows subject to the documented override, §4.2).

| # | tag | axis | `bpb` | `delta` (residual) | `delta_naive_d1` | `recovery` | `recovery_se` | [-2SE, +2SE] | band |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gate_proj 25% | **row** | 1.0803537370961582 | +0.3127587693421209 | 0.3127587693421209 | 0.0000 | 0.0587 | [-0.117, +0.117] | **ZERO_BY_CONSTRUCTION** (theorem) |
| 2 | gate_proj 50% | **row** | 2.0734435288686330 | +1.3058485611145958 | 1.3058485611145958 | 0.0000 | 0.0476 | [-0.095, +0.095] | **ZERO_BY_CONSTRUCTION** (theorem) |
| 3 | gate_proj 75% | **row** | 4.3567971411756901 | +3.5892021734216530 | 3.5892021734216530 | 0.0000 | 0.0371 | [-0.074, +0.074] | **ZERO_BY_CONSTRUCTION** (theorem) |
| 4 | o_proj 25% | col | 0.8636777086132023 | +0.0960827408591649 | 0.8355597811910537 | **0.8850** | 0.0093 | [+0.866, +0.904] | **LARGE** |
| 5 | o_proj 50% | col | 2.0817229473415373 | +1.3141279795875000 | 2.2578215328137845 | 0.4180 | 0.0740 | [+0.270, +0.566] | MODERATE |
| 6 | o_proj 75% | col | 3.3522508982259605 | +2.5846559304719230 | 4.3085281914722540 | 0.4001 | 0.0471 | [+0.306, +0.494] | MODERATE |
| 7 | down_proj 25% | col | 1.1567915935001456 | +0.38919662574610825 | 0.44314865475346077 | 0.1217 | 0.1083 | [-0.095, +0.338] | **INCONCLUSIVE** |
| 8 | down_proj 50% | col | 1.5730582771923323 | +0.8054633094382949 | 1.5571150213992855 | 0.4827 | 0.0437 | [+0.395, +0.570] | MODERATE |
| 9 | down_proj 75% | col | 2.805257935487052 | +2.0376629677330147 | 3.2012581423283817 | 0.3635 | 0.0322 | [+0.299, +0.428] | MODERATE |

Baseline for every `delta` above: `baseline_bpb = 0.7675949677540373`.

`organ_sweep_summary` records `n_column_structured_points: 6`, `n_row_structured_excluded: 3`,
`recovery_mean: 0.445171613696286` (recomputed identical), `recovery_min: 0.12174702197249798`,
`recovery_max: 0.8850079395609454`. Its `note` states the row-structured points are excluded because
"averaging them in would silently drag any 'does reconstruction work' summary toward a result that
was true before the run started."

### 4.2 Rows 1-3 are theorems, and the artefact proves it bit-exactly

Each of the three gate_proj records carries `recovery_verdict: "ZERO_BY_CONSTRUCTION"`,
`excluded_from_summary_stats: true`, and a `verdict_override_reason` stating that
`classify_recovery()` emitted INCONCLUSIVE (band spans 0) and was overridden because
`prereg.derivation_note_mask_geometry` proves recovery = 0 algebraically, so INCONCLUSIVE "would
mischaracterise a confirmed algebraic fact as unresolved measurement noise". `d4_finish.log` records
the three relabels, and `d4_finish.py:183-202` asserts exactly three were relabelled.

**The empirical confirmation is stronger than "approximately zero".** For all three gate_proj rows,
the reconstructed `bpb` **and** `paired_se` are identical, digit for digit, to `d1_pruning.json`'s
`gate_proj/block_structured/{25,50,75}%` records:

| level | D4 `bpb` | D1 `bpb` | identical? | D4 `paired_se` | D1 `paired_se` | identical? |
|---|---|---|---|---|---|---|
| 25% | 1.0803537370961582 | 1.0803537370961582 | **yes** | 0.018368469733788445 | 0.018368469733788445 | **yes** |
| 50% | 2.0734435288686330 | 2.0734435288686330 | **yes** | 0.062182783688584736 | 0.062182783688584736 | **yes** |
| 75% | 4.3567971411756901 | 4.3567971411756901 | **yes** | 0.1329291286390836 | 0.1329291286390836 | **yes** |

That is the theorem confirmed to the last bit, not a recovery of 0 within noise. **These three rows
carry no information about whether reconstruction works.** They are, however, a free and exact
cross-run reproducibility check (§7.4).

### 4.3 Rows 4-9: what the six real measurements say

Reading only from the table in §4.1:

- **One point of nine reaches the pre-registered LARGE band: `o_proj@25%`, recovery 0.8850,
  band [+0.866, +0.904].** It is the only row in the sweep whose band lower bound exceeds 0.5.
- **Four points land MODERATE:** `o_proj@50%` (0.4180), `o_proj@75%` (0.4001),
  `down_proj@50%` (0.4827), `down_proj@75%` (0.3635). All four bands exclude 0.
- **One point is INCONCLUSIVE: `down_proj@25%`, recovery 0.1217, band [-0.095, +0.338].** It is the
  *lowest*-sparsity down_proj point, i.e. the easiest one, and it is the only column-structured
  point the run cannot distinguish from zero recovery.
- **No point lands NEGLIGIBLE.** The band is defined (`recovery < 0.30` and not inconclusive) and
  is simply unoccupied.

**Do not read a monotone trend in these six numbers.** Differencing the recovery values with a
quadrature combination of their `recovery_se` (same approximation caveat as §3.3): `o_proj@50%` vs
`o_proj@75%` differ by +0.0179 at z = +0.20 (indistinguishable); `down_proj@50%` vs `down_proj@75%`
differ by +0.1192 at z = +2.20; `down_proj@25%` vs `down_proj@50%` differ by -0.3610 at z = -3.09,
i.e. **recovery is measurably *worse* at 25% sparsity than at 50% for `down_proj`.** Whatever shape
these six points have, "recovery falls smoothly as sparsity rises" is not it.

### 4.4 The denominator: what is left after the recovery

This is the column that decides whether D4 is a live path. sigma_seed = 0.005 BPB.

| point | `recovery` | residual `delta` (BPB) | residual / sigma_seed | resulting `bpb` vs baseline 0.76759 |
|---|---|---|---|---|
| o_proj 25% | 0.8850 | **+0.09608** | **19.2** | 0.86368 |
| down_proj 50% | 0.4827 | **+0.80546** | **161.1** | 1.57306 |
| down_proj 75% | 0.3635 | **+2.03766** | **407.5** | 2.80526 |
| o_proj 50% | 0.4180 | **+1.31413** | **262.8** | 2.08172 |
| o_proj 75% | 0.4001 | **+2.58466** | **516.9** | 3.35225 |
| down_proj 25% | 0.1217 | **+0.38920** | **77.8** | 1.15679 |
| gate_proj 25% *(theorem)* | 0.0000 | +0.31276 | 62.6 | 1.08035 |
| gate_proj 50% *(theorem)* | 0.0000 | +1.30585 | 261.2 | 2.07344 |
| gate_proj 75% *(theorem)* | 0.0000 | +3.58920 | 717.8 | 4.35680 |

**Not one of the nine points returns the donor to within 19 sigma_seed of its own baseline.** The
best-recovered point in the entire probe, `o_proj@25%` at recovery 0.8850, still sits +0.09608 BPB
above baseline - 19.2 sigma_seed, and 19.2x the noise floor every other decision in this programme
is judged against. `down_proj@50%`, the point the whole ablation battery is staked on, sits at
+0.80546 BPB = 161 sigma_seed.

### 4.5 What each point actually buys, in weights

The recovery fraction says nothing about how much of the model was removed to earn it. Computed
from `arch` (`n_layers 28`, `d_model 1536`, `d_ffn 8960`, `n_heads 12`, `n_kv_heads 2`,
`vocab 151936`, `tie_word_embeddings true`, so head_dim = 128 and the kv width is 2 x 128 = 256):

    total = 28 * (2*1536^2 + 2*1536*256 + 3*1536*8960) + 151936*1536 = 1,543,569,408 weights

**This total is derived by this report from the artefact's `arch` block; the artefact itself does
not record a parameter count.** Per-organ totals over 28 layers: `o_proj` 66.06 M,
`down_proj` 385.35 M, `gate_proj` 385.35 M.

| point | weights zeroed | as % of the 1543.57 M model | residual `delta` |
|---|---|---|---|
| o_proj 25% | 16.52 M | **1.07%** | +0.09608 |
| o_proj 50% | 33.03 M | 2.14% | +1.31413 |
| o_proj 75% | 49.55 M | 3.21% | +2.58466 |
| down_proj 25% | 96.34 M | 6.24% | +0.38920 |
| down_proj 50% | 192.68 M | **12.48%** | +0.80546 |
| down_proj 75% | 289.01 M | 18.72% | +2.03766 |
| gate_proj 25/50/75% | 96.34 / 192.68 / 289.01 M | 6.24 / 12.48 / 18.72% | +0.31276 / +1.30585 / +3.58920 |

**The LARGE row buys 1.07% of the model for +0.0961 BPB (19.2 sigma_seed).** This is a weight count,
not a byte count and not a bandwidth figure - D4 measured neither, and no speed claim can be
derived from it (§9).

---

## 5. Was the solve well-posed?

A recovery number from an ill-conditioned solve is not a measurement. D4 carries two diagnostics.

### 5.1 rank(H)/N - full rank everywhere, and the gate could not fire

| arm | method | ratio_min | ratio_mean | ratio_max | threshold | `ABORTED` |
|---|---|---|---|---|---|---|
| `down_proj_real_H` | **exact** (`eigvalsh`, all 28 layers) | 1.000 | 1.000 | 1.000 | 0.5 | false |
| `o_proj_real_H` | **exact** (`eigvalsh`, all 28 layers) | 1.000 | 1.000 | 1.000 | 0.5 | false |
| `down_proj_identity_H` | analytic | 1.000 | 1.000 | 1.000 | - | - |
| `down_proj_shuffled_H` | theoretical `min(T,D)/D` | 1.000 | 1.000 | 1.000 | 0.5 | false |
| `down_proj_leakage_H` | theoretical, `T_tokens = 12288` | 1.000 | 1.000 | 1.000 | 0.5 | false |
| `down_proj_wrong_layer_H` | derived from `down_proj_real_H` permuted | 1.000 | 1.000 | 1.000 | - | - |

Both exact checks are genuinely exact (28 `eigvalsh` passes each, `rank = N = 8960` and
`rank = N = 1536` at every layer). But **the safety gate this diagnostic exists to be could not have
fired in this run's regime**, and the artefact's own numbers show why: `rank(H) <= min(T, D)`, and
`calib_tokens_T = 16384` exceeds both `d_ffn = 8960` and `d_model = 1536`, so the diagnostic's
own upper bound is 1.0 before a single eigenvalue is computed.
`audits/CONTROLLER_DENSITY_AUDIT.md:414` filed this as a BLOCK, pre-belief, while the run was live.
**The completed artefact confirms the BLOCK exactly:** every arm reports ratio 1.000 and every
`ABORTED` is false. The gate that caught session 1's failure (`0.457 < 0.5`) is inert against any
recurrence in the T > D regime.

### 5.2 Conditioning of `H_reg[S,S]` - report-only, and the down_proj spread is large

Added by `d4_finish.py`. `conditioning_diagnostics.method_note`, verbatim: condition number /
lambda_max / lambda_min are "of H_reg[S,S] = H[S,S] + lambda*I (the matrix actually inverted by the
solve)"; `n_eigendirections_below_tikhonov_lambda` counts eigenvalues of the **raw** H[S,S] below
lambda - "directions where the Tikhonov term dominates the effective scale rather than the
calibration data. Reported for visibility only, per the brief: **NOT used to gate/abort any point
(no threshold has been justified)**."

| point | cond min | cond mean | cond max | worst layer | `n_below_lambda` mean | max (layer) | `n_surviving` |
|---|---|---|---|---|---|---|---|
| o_proj 25% | 2.02e3 | 4.63e5 | 5.36e6 | L27 | 0.0 | 0 | 1152 |
| o_proj 50% | 1.16e3 | 2.10e5 | 2.63e6 | L27 | 0.0 | 0 | 768 |
| o_proj 75% | 4.66e2 | 1.73e4 | 2.78e5 | L25 | 0.0 | 0 | 384 |
| down_proj 25% | 5.68e3 | 1.14e7 | **1.32e8** | L2 | 89.9 | **1783 (L1, 26.5%)** | 6720 |
| down_proj 50% | 1.75e3 | 2.43e6 | 6.37e7 | L2 | 31.8 | 766 (L1, 17.1%) | 4480 |
| down_proj 75% | 3.55e2 | 8.81e5 | 2.41e7 | L2 | 4.1 | 110 (L1, 4.9%) | 2240 |

`conditioning_diagnostics.gate_proj.note` states this is N/A for gate_proj: it is ROW-structured,
no solve is performed and no H is formed, so there is no `H_reg[S,S]` to condition.

**Three observations, each from the rows above:**

1. **o_proj's solves are never regulariser-dominated.** `n_eigendirections_below_tikhonov_lambda` is
   0 at every layer and every level (min = mean = max = 0 across all three o_proj rows). Its
   condition numbers reach 5.36e6 but no direction falls below lambda.
2. **down_proj's are, at two layers.** Across all three down_proj levels, exactly **2 of 28 layers
   (L1 and L2)** have any direction below lambda; at `down_proj@25%`, layer L1 has 1783 of 6720
   surviving directions (26.5%) set by the Tikhonov term rather than by the data, with
   `lambda_used = 4.832` at that layer.
3. **The worst-conditioned point is the INCONCLUSIVE one.** `down_proj@25%` carries the highest
   condition mean (1.14e7), the highest condition max (1.32e8) and the highest regulariser-dominated
   count (1783), and it is the single point in §4.1 whose recovery band spans 0. **This report
   states the co-occurrence and does not claim causation** - no arm in the artefact varies
   conditioning while holding everything else fixed, and `d4_conditioning_and_nulls.py`, which
   would have added a second ablated point, never ran (§2.3).

**lambda itself is not constant across layers.** `lambda = LAMBDA_SCALE * trace(H) / N` with
`LAMBDA_SCALE = 1e-4` (`d4_reconstruction.py:54`). Measured range at 50% sparsity: o_proj from
0.00571 (L2) to 0.14035 (L27), mean 0.02716; down_proj from 0.01101 (L0) to **8.0859 (L26)**, mean
0.75450 - a 734x spread across depth in one organ. `LAMBDA_SCALE = 1e-4` is asserted in the source
with no justification and **no sensitivity sweep exists anywhere in the artefact**
(`CONTROLLER_DENSITY_AUDIT.md:750` filed this as a FLAG before the results landed; the completed
artefact contains no such sweep, so the FLAG stands).

---

## 6. ACHIEVED vs nominal

| quantity | requested / nominal | **achieved** | source field |
|---|---|---|---|
| sparsity, all 9 points | 0.25 / 0.50 / 0.75 | **exact**, `zero_frac_divergence_pct = 0.0` on all 9 rows | `organ_sweep[*].zero_frac_achieved`, `.zero_frac_divergence_pct` |
| calibration tokens | `N_CAL x SEQ_LEN_CAL = 32 x 512` | **16384** | `calib_tokens_T`, `calib_slice` |
| eval sequences | D1's slice, 24 x 512 | **24 x 512, 51870 scored bytes, 0 rejected** | `eval_slice` |
| bootstrap resamples | `n_boot = 2000` (prereg: ">= 10 required") | **2000** (`bootstrap_recovery` default, never overridden) | `d4_reconstruction.py:316` |
| block size, all 3 organs | D0-derived, read off the weights | **64 units (768 B/unit)** for gate_proj, o_proj, down_proj | `block_sizes`, `d4_run.log` |
| rank(H)/N | > 0.5 required | **1.000** on every arm | `rank_diagnostics.*.ratio_min` |
| Cholesky solve | Cholesky, `lstsq` fallback allowed | **`any_fell_back: false` on all 9 sweep points and on the `real_H` and `wrong_layer_H` arms. The field is ABSENT from the `identity_H`, `shuffled_H` and `leakage` records** - not logged, so not verifiable for those three | `organ_sweep[*].any_fell_back`, `controls.hessian_ablation.*.any_fell_back` |
| baseline reproduces D1 | asserted `true` | **verified true, bit-identical**: 0.7675949677540373 in both files | `baseline_bpb` vs `d1_pruning.json.baseline_bpb` |
| calib/eval disjointness | asserted `verified: true` | **verified independently**: both `corpus_sha256` recomputed from the files on disk and matched | `calib_eval_disjointness` |
| `delta_naive` read verbatim from D1 | prereg requirement | **9 of 9 match verbatim** | `organ_sweep[*].delta_naive_d1` vs `d1_pruning.json.organ_sweep` |
| ablated points | prereg says down_proj@0.5 only | **down_proj@0.5 only** - as pre-registered, and no more | `controls.hessian_ablation` |
| threads | `D_THREADS`, default 10 | **10**, same as D1's `threads: 10` | `threads` |
| CUDA | brief's hardware pinning | **`cuda_available: false`**; `hardware.note` records CPU-only, matching D0-D3 | `hardware` |

**Nothing in the sweep was requested-but-not-achieved.** The two things reported as achieved that a
reader could misread are (a) `rank(H)/N = 1.000`, which is achieved but uninformative (§5.1), and
(b) `zero_frac_divergence_pct = 0.0`, which is D4's own block-count arithmetic, not a comparison
against D1's recorded `zero_frac` - see §7.4.

---

## 7. Reproducibility manifest

### 7.1 Donor and slices

| item | value | source |
|---|---|---|
| donor revision | `8faed761d45a263340a0528343f099c05c9a4323` | `arch.revision` |
| architecture | 28 layers, d_model 1536, d_ffn 8960, 12 heads, 2 kv heads, vocab 151936, tied embeddings, silu | `arch` |
| eval slice | `heldout`, 24 x 512, seed 1234, 0 rejected, 51870 scored bytes, 4.22945205479452 B/token | `eval_slice` |
| eval `ids_sha256` | `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65` | `eval_slice` |
| calib slice | `calib`, 32 x 512, seed 42424, 0 rejected, 67648 scored bytes, 4.136986301369863 B/token | `calib_slice` |
| calib `ids_sha256` | `c5509846cdc3aa44e45e77895b59a4638c49eb03790030d851e7bf1357ca4c0c` | `calib_slice` |
| shuffle seed | 777 (+ layer index) | `d4_reconstruction.py:75` |
| bootstrap seed | `11 + point index` | `d4_reconstruction.py`, `d4_finish.py` |
| wrong-layer offset | 14 (derangement on 28 layers) | `controls.hessian_ablation.wrong_layer_H.construction` |

### 7.2 Disjointness - verified, not assumed

`calib_eval_disjointness.verified: true`, with the assertion that the two are different physical
files split from a globally-shuffled chunk list before either slice was drawn. **Checked
independently for this report:** `sha256(corpus/calib.txt) = 10d4d28102625f399f715b6aa220b234c5015dcff199997ccafa06e5c59c89d0`
and `sha256(corpus/heldout.txt) = f46b0310c15faec59ca805d5688317d53b7655ae7099008fe5c3439460d58312`,
both matching the recorded values exactly and differing from each other.
`corpus/manifest.json` records `n_chunks: 35802`, chunk size 8192 B, seed 20260821, split
`calib: 17901 chunks / 140933631 B` and `heldout: 17901 chunks / 140596742 B`, with per-source byte
counts on both sides. The halves are i.i.d., which is what makes §3.4's reading of the leakage
control valid.

### 7.3 A free cross-process determinism check

`controls.hessian_ablation.real_H` was measured in **session 2** (`d4_run.log`). The sweep point
`down_proj/reconstructed/50%` was measured independently in **session 3** (`d4_finish.log`), in a
different process, after `d4_finish.py` re-captured the calibration activations and re-formed
`H_real_down` from scratch (1322 s, per `d4_finish.log`). The two agree to every digit:

| field | session 2 `real_H` | session 3 sweep point |
|---|---|---|
| `bpb` | 1.5730582771923323 | 1.5730582771923323 |
| `delta` | 0.8054633094382949 | 0.8054633094382949 |
| `paired_se` | 0.06897565928813412 | 0.06897565928813412 |
| `lambda_mean` | 0.754496031624772 | 0.754496031624772 |

The capture, the H formation, the Cholesky solve, the weight write-back and the BPB eval are
bit-deterministic across processes on this platform at `threads = 10`.

### 7.4 The mask-equality question, partly settled

`CONTROLLER_DENSITY_AUDIT.md:699` filed a FLAG: D4 *recomputes* D1's mask rather than reading it,
and no runtime assertion compares the two. The completed artefact settles it at **4 of the 9
points**, without any new computation:

| point | evidence that D4's mask == D1's mask | settled? |
|---|---|---|
| gate_proj 25/50/75% | reconstructed `bpb` and `paired_se` bit-identical to D1's `block_structured` rows (§4.2) | **yes, 3 points** |
| down_proj 50% | `identity_H` arm's `delta` and `paired_se` bit-identical to D1's row - and `identity_H` is algebraically the naive mask (§3.3) | **yes, 1 point** |
| o_proj 25/50/75%, down_proj 25/75% | no tie-back exists in the artefact | **no, 5 points** |

**A note on the `zero_frac` fields, so nobody reads a discrepancy into them.** D4 records
`zero_frac_achieved = 0.25 / 0.5 / 0.75` exactly, while D1 records `zero_frac = 0.2500015803745815`
and similar. These are **different quantities**: D4's is a block count
(`zeroed_blocks * block_size / total_units`, `d4_reconstruction.py` main sweep), D1's is measured
on the weights. The bit-identical BPB agreements above are the evidence that the masks themselves
agree; the `zero_frac` fields are not comparable and their divergence is definitional.

### 7.5 Timing, and what the git stamp does and does not pin

| item | value | source |
|---|---|---|
| session 3 wall clock | 2831.1467711925507 s | `finish_wallclock_seconds` |
| session 2 stage timings | baseline 190 s; combined calib capture 266 s; down_proj H formation 263 s; eval-slice capture 98 s | `d4_run.log` |
| session 3 stage timings | baseline 106 s; calib capture 132 s; down_proj H 1322 s; o_proj H 29 s | `d4_finish.log` |
| exit code | `EXIT_CODE=0` | `d4_finish.log` |
| `git_revision` in the artefact | `e748d6de2084263fa1ebf08b86832e4cca7038f8` | `git_revision` |

**The git stamp does not pin the instrument.** `e748d6de` is dated 2026-08-21; the D4 scripts were
first committed in `0d69006` on 2026-08-22. The stamp records the repository HEAD at run time,
while the scripts themselves were uncommitted working-tree files. Further, `d4_finish.py` loads the
existing log and preserves its `git_revision`, so **session 3's code state is not stamped at all**.
To reproduce, use the scripts as committed in `0d69006`, not the tree at `e748d6de`.

### 7.6 What the artefact does not record

`d4_reconstruction.json` carries **no library or platform versions, no peak RSS, and no machine-state
record** (compare `D0_COACTIVATION.md` §7, which carries all three). The only environment fields are
`threads: 10` and `hardware.cuda_available: false`. A future re-run cannot verify it matched this
one's numerical environment beyond the thread count.

---

## 8. Verdict against D4's own pre-registration

**The probe executed its pre-registration completely and without deviation.** All 9 pre-registered
(organ, level) points ran; `delta_naive` was read verbatim from D1 at 9 of 9 points; the SE method
is the pre-registered paired bootstrap at `n_boot = 2000` with the `delta_naive`-fixed assumption
stated openly in `prereg.se_method` and repeated here (§1); the four pre-registered bands were
applied unchanged, and the only verdict override in the run - the three gate_proj rows - is the one
the pre-registration itself derives in advance and is documented per-row with its reason. The
`control3_control4_scope_decision` limitation (ablation at `down_proj@0.5` only) was pre-registered
and honoured, not discovered afterwards. **No band was invented, re-cut, or applied after seeing a
result.**

The result, stated so that each clause names its rows:

1. **The reconstruction mechanism is real and it uses the donor's own per-layer activation
   correlations.** `controls.hessian_ablation`: `real_H` recovery +0.48272 against `identity_H`
   0.0 (which is algebraically the naive mask), `shuffled_H` -1.59517 and `wrong_layer_H` -1.47213;
   the real-vs-wrong-layer separation is 1.9549 recovery units, on an arm built from real spectra
   and real column means with only the layer correspondence broken.

2. **A wrong H is not ignorance-tolerant.** `shuffled_H` (+4.04098 BPB) and `wrong_layer_H`
   (+3.84940) are both worse than `identity_H` (+1.55712, z = +21.15 and +17.46 on a quadrature
   approximation), i.e. worse than not reconstructing at all. They land in the same region as C2
   `saturation` (+3.74530), which zeroes the organ outright - **without being demonstrably beyond
   it** (z = +1.51 and +0.51).

3. **On the pre-registered bands: 1 LARGE, 4 MODERATE, 1 INCONCLUSIVE, 0 NEGLIGIBLE among the six
   column-structured points; 3 theorems excluded.** `organ_sweep_summary.recovery_mean = 0.44517`
   over the six.

4. **And none of it is enough.** No point of the nine returns the donor to within 19 sigma_seed of
   its own baseline (§4.4). The single LARGE row, `o_proj@25%`, leaves +0.09608 BPB = 19.2
   sigma_seed while zeroing 1.07% of the model's weights (§4.5). The most-ablated point,
   `down_proj@50%`, leaves +0.80546 BPB = 161 sigma_seed for 12.48% of the weights. Halving an
   unacceptable loss leaves an unacceptable loss.

5. **The control battery demonstrably fires on a known positive - via C1, not via C3.** The identity
   control caught a real defect in session 1 (`delta = +0.34921`, `max_abs_weight_deviation =
   0.6545`) and aborted the run; in the final run it reads 5.94e-09 BPB. The Hessian ablation
   contains no clean information-free null: `identity_H` is a tautology, `shuffled_H` and
   `wrong_layer_H` are anti-nulls.

6. **The leakage control is the loudest warning in the artefact and must not be filed as a pass.**
   `controls.leakage`: an H built from the eval slice recovers 0.8590 against the honest 0.4827,
   on **fewer** tokens (12288 vs 16384) drawn from an i.i.d. half of the same shuffled corpus. There
   is no distribution mismatch for better data to fix; on these rows the honest solve at T/D = 1.83
   is overfitting its calibration sample, and ~0.38 recovery units separate an honest number from
   what a leaky pipeline would report on this slice.

### Scope of everything above, stated explicitly

**One donor** (Qwen2.5-1.5B, revision `8faed761...`). **One size** (1.5B; `arch.n_layers = 28`,
`d_model = 1536`, `d_ffn = 8960`). **One calibration budget** (`calib_tokens_T = 16384`).
**One eval slice** (24 x 512, 51870 scored bytes). **One block size** (64 units / 768 B, all three
organs). **One lambda scale** (1e-4, unswept). **Three organs of seven.** **One ablated point of
nine.** Nothing here is a statement about layer-wise reconstruction in general.

---

## 9. What D4 does not answer

1. **Nothing here is measured at any size other than 1.5B.** Every number in this report comes from
   one donor at one width. The programme was burned by exactly this gap in the sparsity work:
   `BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md:163-164` (AMENDMENT 1, 2026-09-03) records `S_inter` at
   **46.54% / 50.49% / 71.66% for 0.5B / 1.5B / 14B**, marked `[T]` there as read from
   arXiv:2509.00454 Table 1 - **a prior-art transcription this report cites from the S1 brief and
   did not re-check against the source paper**. That rising trend is why S1 Amendment 1 made the
   multi-size arm mandatory. **D4 has no multi-size arm and no Amendment.** Whether recovery at
   `down_proj@50%` is 0.48 or something else at 14B is unmeasured, and the direction is not
   derivable from anything in the artefact.

2. **There is no clean information-free null.** Reading 1 of §8 ("uses the donor's own per-layer
   correlations") is established; the strictly weaker question - "how much of `real_H`'s 0.483 would
   any real activation structure at matched scale deliver?" - has no arm that isolates it.
   `identity_H` is algebraically the naive mask, `shuffled_H` and `wrong_layer_H` are anti-nulls.

3. **Eight of nine points are un-ablated.** Only `down_proj@50%` has the four-arm battery. o_proj's
   three points and down_proj's other two stand on `real_H` alone; gate_proj's three are theorems.
   `d4_conditioning_and_nulls.py` (committed, never run) would have added `down_proj@75%`.

4. **`LAMBDA_SCALE = 1e-4` is unswept.** No sensitivity analysis exists in the artefact, and lambda
   varies 734x across depth within `down_proj` alone (0.011 at L0 to 8.086 at L26, §5.2).

5. **Conditioning is reported, never gated.** `conditioning_diagnostics.method_note` says so
   explicitly: no threshold has been justified. `down_proj@25%` solved with up to 1783 of 6720
   directions regulariser-dominated at L1 and produced the run's only INCONCLUSIVE point; whether
   that link is causal is untested.

6. **No speed and no memory measurement anywhere.** D4 measures BPB. It says nothing about whether a
   block-column-sparse `down_proj` is faster on the engine, nothing about bandwidth, and nothing
   about bytes moved. §4.5's percentages are weight counts, not traffic.

7. **No combined-organ arm.** Every point sparsifies exactly one organ at one level. There is no
   measurement of what happens when o_proj, down_proj and gate_proj are all sparsified under a
   single budget, and nothing in the artefact licenses adding the per-organ deltas.

8. **No arm beyond one-shot reconstruction.** No fine-tuning, no distillation, no iterative
   mask-and-resolve, no joint mask+weight solve (the module docstring puts the ADMM joint solver
   explicitly out of scope). D4 measures the ceiling of a *single* closed-form update on a *fixed*
   mask.

9. **The 0.90 sparsity level was never run** (`prereg.scope` excludes it) and neither were q/k/v/up
   or the embedding. Four of the donor's seven linear organs are untouched.

### To close the gaps, in priority order

These are re-runs; none was started for this report, and none should be started while PID 17884 is
live.

1. **A second size.** The whole verdict is 1.5B-only. Repeat `down_proj@{25,50,75}%` with the full
   four-arm ablation on at least one other donor width. This is the gap S1 Amendment 1 already
   forced elsewhere.
2. **A genuine information-free null**: an H with the real per-feature scale (matching diagonal) and
   provably no cross-feature structure, so a "recovery near 0 with a well-behaved solve" arm exists.
   Both current nulls are anti-nulls and neither can play this role.
3. **A calibration-budget sweep.** §3.4 shows the honest solve overfits at T/D = 1.83. Re-run
   `down_proj@50%` at T/D of roughly 4x and 8x and see whether the honest recovery moves toward the
   leak arm's 0.859 or stays at 0.483. This is the single cheapest test of whether D4's ceiling is
   a method limit or a sample-size limit.
4. **A LAMBDA_SCALE sweep** at the T/D margin actually used.
5. **Run `d4_conditioning_and_nulls.py`** to obtain the second ablated point it was written for.
6. **Mask-equality assertions** for the 5 points §7.4 could not settle.
