# HANDOFF — COSTRUTTORE — the D4 reconstruction report
updated: 2026-09-03T21:40+02:00   status: **DONE.** Report written, every exact value machine-verified against the artefact.
written by: the Adapter/Principal, before spawning you; sections 3/5/9 rewritten by the Builder

## 1. The task, in one paragraph

D4 ran to completion and **was never written up.** Its results have been sitting in
`benchmarks/donor_adaptation/density/results/d4_reconstruction.json` since 2026-08-22 with no probe
report, no audit, and no verdict. Write the report:
`docs/research/donor_adaptation/probes/D4_RECONSTRUCTION.md`. Report what D4 measured, against its
own pre-registration, with a table for every claim.

**This matters more than its age suggests.** D0 has just closed the FFN co-activation carve as a
negative at 1.5B (+1.09 BPB with an *oracle* router — see `probes/D0_COACTIVATION.md` Part III). D4
sits on the other route to the same goal: instead of carving the donor into experts, **solve for
replacement weights layer by layer** so that a structured-sparse layer reproduces the dense layer's
output. If the programme has a live path to running someone else's pretrained model on our
architecture, D4 is where it is. Write it up so it can be judged.

**You are not told what D4 found.** Read the artefacts and report them.

## 2. Pre-registration you are bound by

D4's pre-registration is **inside its own artefact**, under the `prereg` key. There is NO
`BRIEF_D4_*.md` in `docs/research/donor_adaptation/briefs/` — verified; the `prereg` key is the only
pre-registration that exists. It fixes, before any result:

- `primary_quantity`: `recovery = 1 - delta_reconstructed / delta_naive`, per (organ, sparsity) point,
  with `delta_naive` read **verbatim** from `d1_pruning.json`'s `block_structured` record.
- `se_method`: paired bootstrap, n_boot 2000, `delta_naive` held FIXED (its own SE ignored) — stated
  openly in the artefact. Keep it visible.
- `inconclusive_rule`: INCONCLUSIVE iff `[point - 2*SE, point + 2*SE]` spans 0.
- `magnitude_bands_fixed_before_any_result`: LARGE / MODERATE / NEGLIGIBLE / INCONCLUSIVE.
- `derivation_note_mask_geometry`: q/k/v/gate/up_proj ROW-structured, o_proj/down_proj COLUMN-
  structured. For ROW-structured organs recovery = 0 is a theorem, not a datum.

Wider programme context: `docs/research/donor_adaptation/decisions/DONOR_V2_DENSITY_PROGRAMME.md`.
NOTE a naming collision: that programme doc's "D4" (line 349) is a *different* D4 ("can a small core
learn to drive a frozen donor FFN", 1 T4 session). The density-probe D4 written up here is the
Hessian reconstruction probe. Say so in the report.

## 3. The artefacts — ALL READ, ALL FIGURES EXTRACTED (Builder, 2026-09-03)

| file | tracked? | holds |
|---|---|---|
| `results/d4_reconstruction.json` | yes (commit 0d69006) | the canonical run |
| `results/d1_pruning.json` | yes | `delta_naive`; **all 9 values verified verbatim-equal to D4's `delta_naive_d1`** |
| `results/d4_reconstruction.ABORTED_session1.json` | yes | **session 1's identity control FAILED**: delta +0.34921, max_abs_weight_deviation 0.6545, `fired=false`. Two root causes recorded. This is the positive-control evidence. |
| `d4_reconstruction.py` (749 l) | yes | main sweep, sessions 1-2 |
| `d4_finish.py` (326 l) | yes | session 3: down_proj sweep, gate relabel, wrong_layer_H arm, conditioning |
| `d4_conditioning_and_nulls.py` (396 l) | yes | **NEVER RUN** — it writes `scope_reality_check`, which is absent from the JSON; it would have added a down_proj@75% ablation |
| `d4_run.log`, `d4_finish.log`, `results/d4.log` | yes | the three session logs |
| `results/d4_reconstruction.{PRE_LAUNCH_snapshot,PRE_FINISH_backup}.json`, `d4_quickcheck.log` | no | scratch |
| `audits/CONTROLLER_DENSITY_AUDIT.md:391-950` | yes | Controller **pre-belief** audit of D4: 2 BLOCKs + 5 FLAGs. Methodological, filed before the results landed |

### Every figure needed for the report (all recomputed from JSON by the Builder)

baseline_bpb = 0.7675949677540373, **bit-identical to D1's** (verified, not repeated).
Both corpus sha256 verified against the files on disk. manifest: 35802 chunks split 17901/17901.
sigma_seed = 0.005 BPB. calib T=16384 (T/D = 1.829 down_proj, 10.67 o_proj). leak H T=12288 (1.371).

ORGAN SWEEP — all 9, recovery recomputed to 1e-12, bands re-derived and all agree:

| tag | bpb | delta_recon | delta_naive_d1 | recovery | rec_SE | [-2SE,+2SE] | verdict | delta/sigma_seed |
|---|---|---|---|---|---|---|---|---|
| gate_proj 25% | 1.0803537370961582 | 0.3127587693421209 | 0.3127587693421209 | 0.0000 | 0.0587 | [-0.117,+0.117] | ZERO_BY_CONSTRUCTION (was INCONCLUSIVE) | 62.6 |
| gate_proj 50% | 2.0734435288686330 | 1.3058485611145958 | 1.3058485611145958 | 0.0000 | 0.0476 | [-0.095,+0.095] | ZERO_BY_CONSTRUCTION | 261.2 |
| gate_proj 75% | 4.3567971411756901 | 3.5892021734216530 | 3.5892021734216530 | 0.0000 | 0.0371 | [-0.074,+0.074] | ZERO_BY_CONSTRUCTION | 717.8 |
| o_proj 25% | 0.8636777086132023 | 0.0960827408591649 | 0.8355597811910537 | 0.8850 | 0.0093 | [+0.866,+0.904] | LARGE | 19.2 |
| o_proj 50% | 2.0817229473415373 | 1.3141279795875000 | 2.2578215328137845 | 0.4180 | 0.0740 | [+0.270,+0.566] | MODERATE | 262.8 |
| o_proj 75% | 3.3522508982259605 | 2.5846559304719230 | 4.3085281914722540 | 0.4001 | 0.0471 | [+0.306,+0.494] | MODERATE | 516.9 |
| down_proj 25% | 1.1567915935001456 | 0.38919662574610825 | 0.44314865475346077 | 0.1217 | 0.1083 | [-0.095,+0.338] | INCONCLUSIVE | 77.8 |
| down_proj 50% | 1.5730582771923323 | 0.8054633094382949 | 1.5571150213992855 | 0.4827 | 0.0437 | [+0.395,+0.570] | MODERATE | 161.1 |
| down_proj 75% | 2.805257935487052 | 2.0376629677330147 | 3.2012581423283817 | 0.3635 | 0.0322 | [+0.299,+0.428] | MODERATE | 407.5 |

gate_proj: bpb AND paired_se are **bit-identical to D1's block_structured rows at all 3 levels**.
summary: n_col=6, n_row_excluded=3, mean 0.445171613696286 (recomputed identical), min 0.1217, max 0.8850.

CONTROLS (all on down_proj):
- identity (mask 0%): bpb 0.7675949736956782, delta 5.941640845996687e-09, se 9.03e-09,
  max_abs_weight_deviation 2.98e-08, fired=true. Session 1's same control: delta +0.34921,
  dev 0.6545, fired=false.
- saturation (mask 100%): bpb 4.5128979664456885, delta +3.7453029986916513, se 0.16724, fired=true.
- hessian_ablation @50%: real_H delta +0.80546 rec +0.4827 | identity_H delta +1.5571150213992855
  rec 0.0000 | shuffled_H delta +4.04098 rec -1.5952 | wrong_layer_H delta +3.84940 rec -1.4721.
  **identity_H's delta AND paired_se are bit-identical to D1's naive record** -> it is a tautology
  (H=I => H_reg=(1+lam)I => the solve returns W[:,S]), i.e. a mask-equality cross-check, NOT a null arm.
- leakage (eval-slice H) @50%: bpb 0.987097467038977, delta +0.2195024992849397, se 0.01628,
  recovery +0.8590, leakage_looks_better_than_honest=true. **The leak H has FEWER tokens
  (12288 < 16384) yet recovers 0.859 vs 0.483** -> the gap is eval-sample overfit, not calibration budget.

RANK: every arm ratio_min = ratio_mean = ratio_max = 1.000 (down/o real_H exact eigvalsh; shuffled
and leakage theoretical; wrong_layer derived; identity_H analytic). Gate threshold 0.5 — unreachable.

CONDITIONING (of H_reg[S,S], report-only, no gating threshold):

| point | cond min/mean/max | worst layer | n_below_lambda mean/max | n_surviving |
|---|---|---|---|---|
| o_proj 25% | 2.02e3 / 4.63e5 / 5.36e6 | L27 | 0.0 / 0 | 1152 |
| o_proj 50% | 1.16e3 / 2.10e5 / 2.63e6 | L27 | 0.0 / 0 | 768 |
| o_proj 75% | 4.66e2 / 1.73e4 / 2.78e5 | L25 | 0.0 / 0 | 384 |
| down_proj 25% | 5.68e3 / 1.14e7 / 1.32e8 | L2 | 89.9 / 1783 (L1, 26.5%) | 6720 |
| down_proj 50% | 1.75e3 / 2.43e6 / 6.37e7 | L2 | 31.8 / 766 (L1, 17.1%) | 4480 |
| down_proj 75% | 3.55e2 / 8.81e5 / 2.41e7 | L2 | 4.1 / 110 (L1, 4.9%) | 2240 |

Only 2 of 28 layers (L1, L2) have any n_below>0, at every down_proj level.
lambda_used per layer: o_proj 0.00571 (L2) .. 0.1404 (L27), mean 0.02716;
down_proj 0.01101 (L0) .. 8.0859 (L26), mean 0.75450. LAMBDA_SCALE = 1e-4 (asserted, unjustified).

WEIGHT-COUNT ARITHMETIC (derived by the Builder from `arch`; formula reproduced in the report):
total = 28*(2*1536^2 + 2*1536*256 + 3*1536*8960) + 151936*1536 = 1543.57 M (tied embeddings).
o_proj@25% removes 16.52 M = 1.07% of the model. down_proj@50% removes 192.68 M = 12.48%.

## 4. What is RUNNING right now

| what | PID | note |
|---|---|---|
| S1 `run` stage, 1.5B, CPU fp32 | **17884** | **DO NOT KILL.** ~10 GB RSS, 6 torch threads, until ~22:30-23:00 local |

Nothing in the analysis above required a re-run. Everything is JSON arithmetic. **No D4 computation
was re-run and none is needed for the report** — the things that *would* need compute are in §9.

## 5. What is NEXT — ALL DONE

**Deliverable: `docs/research/donor_adaptation/probes/D4_RECONSTRUCTION.md`** (753 lines, sections
0-9, D0 house style: table per claim, controls before results, controls scorecard, ACHIEVED vs
nominal, reproducibility manifest, verdict, "what this does not answer" + priority-ordered gap list).

1. ~~Read `prereg` in full; read D0's structure.~~ DONE — §1 quotes the prereg verbatim, §0 explains
   what `recovery` is not.
2. ~~Controls before results, incl. the Hessian 4-arm and `leakage_looks_better_than_honest`.~~
   DONE — §3, with §3.5 scorecard. The leakage reading is §3.4 and is NOT buried.
3. ~~All 9 sweep points, 3 marked as theorems, each banded.~~ DONE — §4.1 (all 9), §4.2 (theorems,
   proven bit-exact), §4.4 (residual in BPB and sigma_seed), §4.5 (weight cost).
4. ~~`rank_diagnostics` + `conditioning_diagnostics`.~~ DONE — §5.
5. ~~Verdict + scope statement.~~ DONE — §8, six numbered findings + explicit scope paragraph.
6. ~~"What D4 does not answer".~~ DONE — §9, nine items + six prioritised re-runs.

**Verification performed before closing:** a script re-extracted every exact float from the artefact
(`organ_sweep` bpb/delta/delta_naive_d1, all four ablation arms' bpb/delta/paired_se/recovery_point,
identity/saturation/leakage bpb/delta/paired_se, baseline, summary stats, git rev, wallclock) and
asserted each appears verbatim in the report. Result: **ALL EXACT VALUES MATCH THE ARTEFACT.** Two
transcription roundings were caught and fixed this way. Every cited source line number in the
report was checked against the file.

**Nothing was re-run.** PID 17884 untouched.

## 6. Traps — these are the ones this project actually falls into

- **Exact tables, overstated prose.** Every sentence must name the row it comes from.
- **Charged bytes vs moved bytes**: a ceiling is a denominator. `recovery` IS a fraction whose
  denominator is D1's catastrophic naive delta — a big fraction of a catastrophe is a catastrophe.
  Print the residual `delta` in BPB and in sigma_seed next to every recovery number.
- **Do not let a theorem masquerade as a result.** gate_proj's three zeros.
- **Report ACHIEVED, always.**
- **No literature number unless read in that paper's own table.** Nothing in this report needs one.
- sigma_seed = **0.005 BPB**.

## 7. Everything needed to restart cold

- Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`.
- Donor Qwen2.5-1.5B rev `8faed761d45a263340a0528343f099c05c9a4323`; 28 layers, d_model 1536,
  d_ffn 8960, 12 heads / 2 kv, vocab 151936, tied embeddings, silu.
- Eval slice heldout 24x512 seed 1234, 51870 scored bytes. Calib slice calib 32x512 seed 42424,
  67648 scored bytes. Disjointness VERIFIED (distinct corpus files, hashes recomputed on disk).
- `d4_reconstruction.json`: `git_revision e748d6de...`, `threads 10`, `finish_wallclock_seconds 2831`.

## 8. Update this file

Rewrite sections 3 and 5 as you go — after each report section you finish.

## 9. What D4 does not answer (drafted, fold into the report)

- **Nothing here is measured at any size other than 1.5B.** One donor, one size, one calibration
  budget (T=16384). The same gap that forced S1 Amendment 1.
- No clean information-free H null exists. shuffled_H (-1.595) and wrong_layer_H (-1.472) are both
  ANTI-nulls; identity_H is a tautology. So "the solve uses H's structure" is established; "how much
  of any real activation structure would do" is NOT.
- Only down_proj@50% is ablated. o_proj's three points and down_proj@{25,75}% stand on real_H alone.
  `d4_conditioning_and_nulls.py` would have added down_proj@75% — it was never run.
- No sensitivity sweep on LAMBDA_SCALE=1e-4 exists in the artefact.
- No speed or memory measurement: D4 measures BPB only. Nothing about whether a block-sparse organ
  is actually faster on the engine.
- No end-to-end arm: every point sparsifies ONE organ at one level. No combined organ budget.
- No fine-tune / distillation arm — reconstruction is the only weight update tested.
