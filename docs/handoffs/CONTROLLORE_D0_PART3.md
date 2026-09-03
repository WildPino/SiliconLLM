# HANDOFF — CONTROLLORE — audit of D0 Part III
updated: 2026-09-03T20:50+02:00   status: IN PROGRESS (Controller spawned, briefing read, recomputation started)
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
