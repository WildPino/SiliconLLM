# HANDOFF — CONTROLLORE — audit of D0 Part III
updated: 2026-09-03T21:15+02:00   status: IN PROGRESS (10 findings settled; writing audit file)
written by: the Adapter/Principal, before spawning you

## 1. The task, in one paragraph

Audit **Part III** of `docs/research/donor_adaptation/probes/D0_COACTIVATION.md` (sections 12-15,
lines 1518-1778, commit `7c49a1b`) against the artefacts it claims to rest on. Issue
**BLOCK / FLAG / PASS** per finding, each with a reproduction path a third party can run. You did
not write Part III and you must not rewrite it — you audit it.

**You are not told whether Part III is right.** Form your own verdict from the files.

## 2. What you are bound by

- The audit contract this project uses is `docs/research/donor_adaptation/audits/CONTROLLER_AUDIT_MANDATE.md`.
  Read it. Match the format of `CONTROLLER_D0_AUDIT.md` (the audit of Parts I and II, 2026-08-29) —
  verdict summary table first, then one section per finding.
- Write to `docs/research/donor_adaptation/audits/CONTROLLER_D0_PART3_AUDIT.md`. Do not touch the
  probe report itself.
- **σ_seed = 0.005 BPB** is the project's noise constant; every delta is judged against it.
- Standing law here, and it is the reason you exist: *"three probes audited, three times the tables
  were perfect and the prose was not."* **The failure mode of this project is the plausible artefact
  and the overstated summary sentence, not the wrong number in a cell.** Check every summary sentence
  against the row it claims to summarise.

## 3. What Part III claims — the claims, not their truth

Part III does two things: it answers the Controller's BLOCK **B3** from the previous audit, and it
reports an end-to-end BPB measurement of a carved donor. Its load-bearing claims:

1. B3 (unseeded `torch.svd_lowrank` sketch) is repaired; same-seed repeat now gives ARI 1.0000 and
   bitwise-identical labels.
2. Section 11.3's error bar was the **null arm's** sd (0.00079), not the treatment's; the treatment's
   own sd across 8 seeds is 1.4x-200x larger, so the quoted "140 sd" is honestly "41.6 sd".
3. 190 of 210 cells clear 2x their own measured treatment sd, and all 20 failures sit at `p = 0.20`.
   **Part III itself flags that the treatment sd was measured at `p = 0.10` only and extrapolated
   across density — check whether that caveat is adequate or whether it invalidates the count.**
4. The partition is deterministic but not identified: ARI across seeds 0.42-0.53.
5. Carving all 28 layers at E=32/k=8 with an **oracle** router costs **+1.09062 BPB = 218 σ_seed**;
   co-activation beats its random null by **−0.72051** (z = −6.3 paired on sequences), i.e. recovers
   39.8% of the damage.
6. The carve reproduces across two processes three days apart to 3.8e-08.
7. Section 14 assigns a status to each finding of the previous audit; section 15 declares D0 closed
   as a negative at 1.5B.


## 3b. CONTROLLER'S RUNNING LEDGER (written by the Controller, 2026-09-03 ~21:15)

Everything below is recomputed by me from the rawest artefact, not read out of a JSON that also
reports it. Reproduction commands are in the audit file
`docs/research/donor_adaptation/audits/CONTROLLER_D0_PART3_AUDIT.md`.

### Settled — CONFIRMED (no finding)
- All five BPB values reproduce exactly from `d0_carved_arms/*.npy` as `nats.sum()/ln2/51870`
  (max |diff| 4.4e-16 vs the JSON).
- All paired SEs / z / frac_tokens_worse in `paired_vs_baseline` and `paired_coact_vs_null` reproduce
  from the `.npy` + the pinned slice byte counts. Two *independent* SE estimators (jackknife over the
  24 sequences, and a linearised ratio/delta-method SE) agree with the reported bootstrap to <=3%.
  z(seq) = 15.37 / 15.55 / 15.62 / 12.94 and -13.34 / -6.28. **13.2 and the coact-vs-null table are
  clean.**
- 13.4's cross-process table reproduces (max dev 3.73e-8; report says 3.8e-8).
- The eval slice `ids_sha256` on disk matches the pinned value; 24x512, 51870 bytes, 12264 tokens.
- 12.2's ARI table reproduces exactly (L1 .493 / L7 .529 / L14 .424 / L21 .514 / L27 .449, min .414
  max .472, 28 pairs).
- 12.3's whole table reproduces exactly - margins (= `coactivation_best - random_max`), null sds,
  treatment sds and every ratio, incl. the 41.6 correction of the 140.
- 209/210 and 195/210 both reproduce from `d0_coactivation.analyse.json`.
- 190/210 reproduces *arithmetically*, and so does the exact failing-cell list.
- **I reproduced the carve partitions myself**: `torch.manual_seed(7)` + `balanced_labels(...,32,7)`
  gives labels **bitwise identical** to `d0_carved_labels_E32.npz` for L0, L14, L27, in my own process.
  Without the `manual_seed` the same call returns a different partition (ARI 0.34-0.38). So the repair
  is real and load-bearing, and 13.4's partitioning claim holds.

### Settled — FINDINGS (draft verdicts)
1. **BLOCK - the B3 repair is not in the repo.** `grep manual_seed benchmarks/donor_adaptation/density/*.py`
   returns only d2/d2b/d4. `d0_layout.py:143,182` still calls `torch.svd_lowrank` off the global RNG and
   neither `d0_coactivation.py` nor `common.py` seeds it. The repair exists ONLY as a line inside two
   scratchpad scripts in a session-temp dir (`.../1f268f4b-.../scratchpad/b3_sweep.py`,
   `carved_bpb_paired.py`) that are not committed anywhere. 12.1's "B3 is closed on reproducibility" is
   false at the repo level.
2. **BLOCK - "Every co-activation number in Parts I and II is now re-derivable from its seed" is false,**
   and I proved it: the main run's `coactivation_best` at p=0.10 does not equal the seed-7 sweep value at
   any cell I checked, and falls OUTSIDE the whole 8-seed range in 4 of 15 (e.g. L1 bs12 0.394092 vs
   seed-7 0.392766, 8-seed range [0.386931, 0.393278]). Parts I/II were drawn from an unrecorded global
   RNG state; nothing recovers it. The repair fixes the future, not the past.
3. **BLOCK - 190/210 is an artefact of the p=0.10 -> all-p sd extrapolation; the caveat is not adequate.**
   Split by density: p=0.05 70/70, p=0.10 70/70, p=0.20 **50/70**. Every failure is at the one density
   where the sd is imported. **19 of the 20 "failures" have 2*sd LARGER than the entire treatment
   statistic at that cell** (L27 p=0.20 bs128: statistic 0.000133, threshold 0.010401 = 78x) - the test
   cannot be passed there by any amount of structure. Re-run with a density-aware sd: magnitude-matched
   imputation -> 199/210; proportional (relative-sd) scaling -> 209/210. Honest statement: at the only
   density where the treatment sd was measured, the count is **70/70**.
4. **FLAG - "18 of the 20 at block >= 32" contradicts its own printed list.** All 20 are at
   bs in {32,42,64,128,140}. It is **20 of 20**. (Understates, so it leans the same way as #3.)
5. **FLAG - "the treatment sd is 1.4x to 200x the null's" is wrong at both ends.** Over the 70
   (layer,block) cells at p=0.10 the ratio runs **0.96x to 4545x**. At L7 bs16 the treatment sd is
   SMALLER than the null's (0.000630 vs 0.000656), so "always larger" is false.
6. **FLAG - "the damage compounds with depth rather than accumulating linearly" (13.3) inverts its own
   number.** 28 layers cost 16.8x one layer. 16.8 < 28: that is SUB-linear. Same direction of lean.
7. **FLAG - "the marginal slice SE ... would swamp every effect below" (13.1) is false for two of the
   four effects.** 0.0622 does not swamp +1.09 or +1.81; a marginal two-sample SE gives z=9.9 for
   all_coact. It also calls two different quantities "the same comparison".
8. **FLAG - "No trainable router can beat it" / "no routing improvement can move it" is not established.**
   The oracle is top-k by retained activation ENERGY (`(h**2) @ onehot`), per token, per layer, greedily.
   That is an oracle on a proxy, not on BPB, and it ignores cross-layer interaction. Ceiling framing is
   load-bearing for 13.3 and for 15's "do not open the joint brief".
9. **FLAG - F8 is not "PARTLY ANSWERED".** The audit's F8 asked for a *positive* control on the
   fidelity/geometry/duplication paths. Section 13 supplies a *negative* control on a *different* metric.
   Under this programme's planted-control law that is a category substitution, not a partial answer.
   Also: the carve hook itself has **no** control of any kind - the baseline arm installs no hook, so the
   hook code path is never exercised against a known answer.
10. **FLAG - section 14 silently omits 7 of the previous audit's 15 findings** (F2, F3, F7, F9, F10, F11,
    F12). F10 (provenance) is live again: both carve runs record `git_dirty_at_launch: true`, and the
    scripts that produced every Part III number are uncommitted scratchpad files.

### Direction of the lean (trap 6 in section 6)
Part III leans the SAME WAY as Parts I/II did: findings 3, 4, 6 and 8 each make the negative look wider
or more final than the artefacts support. The correction in 12.3 is honest and correct; the re-count in
12.4 over-corrects.

## 4. The artefacts. Every number above must come from one of these

| file | holds |
|---|---|
| `benchmarks/donor_adaptation/density/results/d0_b3_treatment_spread.json` | the repair string, `determinism_check`, per-layer `per_seed` / `treatment_stats` (14 block sizes) / `ARI_across_seeds_E64`; 8 seeds, `p_achieved` 0.10, `k_active` 896 |
| `.../d0_b3_sweep.log` | the sweep as it ran, per-layer sd and ARI |
| `.../d0_carved_bpb_paired.json` | `config`, `bpb`, `marginal_slice_se`, `paired_vs_baseline`, `paired_coact_vs_null`, `mem_peak` |
| `.../d0_carved_bpb.json` + `.log` | the earlier unpaired run, 3 days before — the cross-process check |
| `.../d0_carved_arms/*.npy` | per-sequence nats, 5 arms — the paired SEs must be re-derivable from these |
| `.../d0_coactivation.analyse.json` | the 210 cells: `deep_layers[L].densities[p].verdict_by_block[bs]` |
| `docs/research/donor_adaptation/audits/CONTROLLER_D0_AUDIT.md` | the previous audit, whose findings section 14 claims to resolve |

## 5. What is NEXT, in order

1. Read the previous audit and the audit mandate.
2. Read Part III (lines 1518-1778).
3. **Recompute independently.** Do not accept a number because the JSON also contains it — recompute
   from the rawest thing available. In particular: the paired SEs and z-scores must be re-derived
   from `d0_carved_arms/*.npy`, not read from `paired_vs_baseline`.
4. Re-derive the three cell counts (209/210, 195/210, 190/210) from `d0_coactivation.analyse.json`
   yourself, and decide whether the `p = 0.10` → all-`p` sd extrapolation in claim 3 is defensible.
5. Check every **summary sentence** in sections 13.3, 14 and 15 against the row it names.
6. Check section 14's status assignments against what the previous audit actually said. B2 and F4 are
   claimed as "still open" — verify Part III did not quietly claim credit for something it did not do.
7. Write the audit. Verdict summary table first.

## 6. Traps

- **The planted-control law:** an instrument's null results are worth nothing until it has been shown
  to FIRE on a known positive. Ask what fires here and whether it is a real positive control.
- **Charged bytes vs moved bytes**: a ceiling is a denominator, and that is where the wrong unit hides.
- **Do not count log lines as evidence.**
- The previous audit found that three independent over-statements all pushed the *same* direction —
  they made the negative look wider than the data made it. **Check whether Part III leans, and which
  way.** A correction that over-corrects is still a lean.
- Part III was written by the same figure that ran the measurements. Treat the "free determinism
  check" and "the measurement was good enough" framings as claims to test, not as context.

## 7. Everything needed to restart cold

- Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`, commit `7c49a1b`.
- **A long CPU job is running: PID 17884, ~10 GB RSS, 6 torch threads, until roughly 22:30-23:00
  local. DO NOT KILL IT** and do not start a competing heavy torch job. Your checks are seconds to
  minutes of CPU; that is fine alongside it.
- The eval slice everything shares: `heldout`, 24 x 512, seed 1234,
  `ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
  12,264 predicted tokens, 51,870 scored bytes.
- Donor: Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`, 28 layers, d_model 1536,
  d_ffn 8960.

## 8. Update this file

Rewrite sections 3 and 5 as you go — after each finding you settle. Assume you will be killed without
warning and that your successor reads only this file.
