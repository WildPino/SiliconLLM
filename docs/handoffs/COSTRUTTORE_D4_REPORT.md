# HANDOFF — COSTRUTTORE — the D4 reconstruction report
updated: 2026-09-03T20:35+02:00   status: NOT STARTED
written by: the Adapter/Principal, before spawning you

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

D4's pre-registration is **inside its own artefact**, under the `prereg` key — read it first and
quote it in the report. It fixes, before any result:

- `primary_quantity`: `recovery = 1 − delta_reconstructed / delta_naive`, per (organ, sparsity) point,
  with `delta_naive` read **verbatim** from `d1_pruning.json`'s `block_structured` record for that
  exact (organ, level).
- `se_method`: paired bootstrap, n_boot 2000, over sequence resampling on the same 24x512 heldout
  slice D1 used, propagated through the ratio with `delta_naive` held fixed — **and the artefact
  states this assumption openly rather than hiding it. Keep it visible in the report.**
- `inconclusive_rule`: a point is INCONCLUSIVE iff `[point − 2·SE, point + 2·SE]` spans 0.
- `magnitude_bands_fixed_before_any_result`: LARGE / MODERATE / NEGLIGIBLE / INCONCLUSIVE, with
  numeric cutoffs. **Use these bands. Do not invent new ones and do not re-band a result.**
- `derivation_note_mask_geometry`: q/k/v/gate/up_proj are ROW-structured, o_proj/down_proj are
  COLUMN-structured. **For the row-structured organs a recovery of 0 is a theorem, not a datum.**
  The artefact excludes them from its summary for exactly that reason. Explain this in the report —
  a reader who does not understand it will misread the whole table.

Wider programme context: `docs/research/donor_adaptation/decisions/DONOR_V2_DENSITY_PROGRAMME.md`.

## 3. The artefacts

| file | holds |
|---|---|
| `results/d4_reconstruction.json` | the run: `prereg`, `arch`, `eval_slice`, `calib_slice`, `calib_eval_disjointness`, `baseline_bpb`, `rank_diagnostics`, `controls`, `controls_summary`, `organ_sweep` (9 points), `organ_sweep_summary`, `conditioning_diagnostics` |
| `results/d1_pruning.json` | `delta_naive` — the denominator of every recovery number. Do not recompute it; D4's prereg says it is read verbatim |
| `results/d4_reconstruction.PRE_LAUNCH_snapshot.json`, `.PRE_FINISH_backup.json` | intermediate snapshots, untracked |
| `results/d4_quickcheck.log` | a partial re-check, untracked |
| `results/d_baseline.json` | the shared donor baseline |

`baseline_bpb = 0.7675949677540373` and the artefact asserts `baseline_reproduces_d1_exactly = true`.
That is checkable and you should check it, not repeat it.

## 4. What is RUNNING right now

| what | PID | note |
|---|---|---|
| S1 `run` stage, 1.5B, CPU fp32 | **17884** | **DO NOT KILL.** ~10 GB RSS, 6 torch threads, until ~22:30-23:00 local |

Do not start a heavy competing torch job. Reading and analysing JSON is free; if you need to re-run
any D4 computation, say so in the report rather than starting it, or wait for 17884 to finish.

## 5. What is NEXT, in order

1. Read `prereg` in full. Read the D0 report's structure (`probes/D0_COACTIVATION.md`) — match its
   house style: a table for every claim, ACHIEVED vs nominal, a controls scorecard, a
   reproducibility manifest, and an explicit "what this run does not answer" section.
2. **Report the controls before the results.** `controls_summary` carries an `identity`, a
   `saturation`, a `leakage` and a **Hessian ablation with four arms** (`real_H`, `identity_H`,
   `shuffled_H`, `wrong_layer_H`). Work out what each arm proves and whether the set constitutes a
   genuine positive control — this programme's standing law is that **an instrument's null results
   are worth nothing until it has been shown to FIRE on a known positive.**
   Note `leakage_looks_better_than_honest = true` and explain what it means; do not bury it.
3. The organ sweep: 9 points, of which the summary counts 6 as column-structured and excludes 3.
   Report **all 9**, with the 3 excluded ones clearly marked as theorems. Give each point its band
   from the pre-registered bands.
4. `rank_diagnostics` and `conditioning_diagnostics` — these say whether the solve was well-posed.
   A recovery number from an ill-conditioned solve is not a measurement. Report them.
5. Verdict against the pre-registration, and an explicit statement of scope: one donor, one size
   (1.5B), one calibration budget (`calib_tokens_T = 16384`).
6. **Say plainly what D4 does not answer.** In particular whether anything here has been measured at
   any size other than 1.5B — the programme has just been burned by exactly that gap (see S1
   Amendment 1).

## 6. Traps — these are the ones this project actually falls into

- **Exact tables, overstated prose.** Three probes have now been audited and three times the tables
  were perfect and the summary sentences were not. **Every sentence must name the row it comes from.**
  A Controller will audit this report and will check exactly that.
- **Charged bytes vs moved bytes**: a ceiling is a denominator; that is where the wrong unit hides.
- **Do not let a theorem masquerade as a result.** The row-structured zeros are the live instance.
- **Do not report a requested value as achieved.** Report ACHIEVED, always, and say so.
- **No literature number enters a claim unless you have read it in that paper's own table.** Marking
  it `[L?]` is not enough. This project has fabricated three such numbers in one session before.
- σ_seed = **0.005 BPB** is the noise constant every delta is judged against.

## 7. Everything needed to restart cold

- Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`.
- D4 artefacts: `benchmarks/donor_adaptation/density/results/d4_*`. Instrument:
  `benchmarks/donor_adaptation/density/d4_reconstruction.py`.
- Donor: Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`.
- Eval slice: `heldout`, 24 x 512, seed 1234. Calib slice is disjoint and the artefact carries the
  assertion under `calib_eval_disjointness` — verify it, do not assume it.
- `d4_reconstruction.json` records `git_revision e748d6de...` and `threads 10`.

## 8. Update this file

Rewrite sections 3 and 5 as you go — after each report section you finish. Assume you will be killed
without warning and that your successor reads only this file.
