# C50-A4-bis result: A4 PASS on text written after the model's release

18 October 2026. Protocol: [C50_A4BIS_POSTRELEASE_PROTOCOL_20261018](C50_A4BIS_POSTRELEASE_PROTOCOL_20261018.md).
Verdict files: `results/c50_a4bis_verdict_plant_pc.json`, `results/c50_a4bis_verdict_plant_pc2.json`,
`results/c50_a4bis_verdict_final.json`. Run record: `results/c50_a4bis_run.json`.

## Verdict

**A4 PASS.** The coarser plant PC2 fires, so the document gate sees a measured loss on this text, and P0 passes
the same gate. T0 = P0 is qualified for ROADMAP §5 S0.

## The runs, in the registered order

| Step | Arm | Run | G1 dBPB against BF16 | CI95 | Outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | BF16 | 18:57-19:43, 2,772 s | (reference: BPB 0.7800) | | COMPLETE |
| 1 | PC (Q2_K, FIT imatrix) | 19:43-20:27, 2,483 s | +0.0019 | (-0.0045, +0.0083) | PLANT DOES NOT FIRE |
| 3 | PC2 (Q2_K, no imatrix) | built 20:29-20:34, scored 20:36-21:21, 2,714 s | +0.0351 | (+0.0274, +0.0437) | PLANT FIRES |
| 3 | P0 (T0, Q4_K_M, FIT imatrix) | 21:26-22:05, 2,291 s | **-0.0035** | **(-0.0063, -0.0007)** | **PASS** (preferred +0.010 met) |

- **Inputs:** 72 documents (24 code, 24 technical, 24 prose), 151,739 scored positions, 589,604 bytes; inputs
  manifest `a102771a…`.
- **Every panel** is COMPLETE, with its CONFIG confirmed. Every model hash equals its pin; BF16's equals the A2
  record. PC2 was pinned and pushed before it scored.
- **Steps 1 and 2** ran from the owner's command C in the owner's terminal. Step 3 ran in the session.

## By category (G1 dBPB, reported)

| Arm | code | technical_general | prose |
| --- | --- | --- | --- |
| PC | +0.014 | +0.009 | -0.018 |
| PC2 | +0.035 | +0.055 | +0.015 |
| P0 | -0.002 | -0.006 | -0.003 |

## Reported, not gated

| Arm | top-1 disagreement with BF16 (CI95) | top-1 accuracy | mean abs dlog p | bits on BF16's tail (1,605 positions) | bits outside it |
| --- | --- | --- | --- | --- | --- |
| PC | 15.5% (14.3-16.7%) | 63.03% | 0.447 nats | -3,149 | +4,243 |
| PC2 | 17.7% (16.4-19.0%) | 62.29% | 0.528 nats | -2,695 | +23,410 |
| P0 | 7.7% (7.0-8.3%) | 63.46% | 0.207 nats | -1,019 | -1,069 |

- BF16's top-1 accuracy is 63.46%. "Tail" means positions where BF16 gives the target log p < -15. Negative bits
  are bits saved against BF16.
- **BF16's BPB here:** code 0.492, technical 0.923, prose 0.925. On A4's FIT text they were 0.419, 0.600 and 1.306.
  The technical text is much less predictable here than on FIT.

## Registered predictions, scored

- **PC fires, with dBPB in +0.015..+0.080: WRONG.** PC's dBPB is +0.0019, and it does not fire.
- **P0's dBPB in +0.002..+0.015: WRONG.** It is -0.0035: P0 scores below BF16.
- **P0 passes G1: RIGHT.**
- **PC's top-1 disagreement exceeds P0's: RIGHT** (15.5% against 7.7%).

## What the numbers say, and what they do not

- **The plant needed the coarser step.** With the FIT imatrix, Q2_K keeps BPB within +0.008 even here, while it
  changes 15.5% of the top-1 choices. On its own, a BPB gate on 72 documents does not see that format's loss.
  Removing the importance weighting moves BPB by +0.035, in every category.
- **P0 below BF16 is measured, not explained.** P0 saves bits on BF16's tail and outside it (-1,019 and -1,069).
  The gate needs only P0's upper bound (-0.0007 ≤ +0.020). No cause is claimed, and none is needed for the
  qualification.
- **The imatrix is FIT text,** not this text, so it does not leak this text into P0. Its effect here is reported, not
  tuned.
- **Limits stand** (protocol): "written after the release" is not "never seen"; there are 72 documents; the result
  is on llama.cpp, not on the engine (A7), and not a rate.

## Next

- **S0** (v0 of T0 on the engine) follows. It reads rates, so it needs an empty machine and the owner's go (rule 7
  as amended): ~1 h, since A1b has already passed (`c50_a1b_verdict_2.json`).
- **S2's F0 arm** re-runs under its addendum 1, one long job at a time, never beside S0.
