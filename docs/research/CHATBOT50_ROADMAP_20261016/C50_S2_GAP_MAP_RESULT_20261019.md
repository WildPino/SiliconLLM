# C50-S2 result: the Ling-mini -> Ling-flash gap is MIXED; S3-S5 proceed

19 October 2026. Protocol: [C50_S2_GAP_MAP_PROTOCOL_20261017](C50_S2_GAP_MAP_PROTOCOL_20261017.md) (with its
addendum 1). Run record: `results/c50_s2_run.json`. Per-arm scores: `results/c50_s2_scores_m0.json`,
`results/c50_s2_scores_f0.json`, `results/c50_s2_scores_t0.json`. Verdict: `results/c50_s2_verdict.json`.

## Verdict

**MIXED, proceed to S3-S5.** On the paired items, F0 removes about a fifth of M0's errors in both families:
RER_K = 0.219 (CI95 0.180..0.258) and RER_R = 0.247 (CI95 0.171..0.320). D = RER_K - RER_R = -0.027, and its CI95
(-0.111..+0.058) contains 0. So the outcome is not REASONING, and the CI95 of RER_K lies above 0: the registered rule
says proceed. The verdict script's planted cases (`--selftest`) passed again before it ran.

## The runs

| Arm | Model | Sessions | Panels |
| --- | --- | --- | --- |
| T0 | P0 (Ling-mini Q4_K_M, FIT imatrix, `44de7315`) | 18 October | nq, tqa, mmlu run; gsm8k, arc, docs, chat reused from A4 (checked) |
| M0 | Ling-mini Q4_K_M, no imatrix (`1f545368`) | 18 October | all seven COMPLETE |
| F0 | Ling-flash-2.0 Q4_K_M, no imatrix (`5348c73f`) | 18 October (chat), 19 October 08:47-16:33 (six panels, addendum 1) | all seven COMPLETE |

- **F0's six panels** ran from the owner's terminal (command D) under addendum 1: no projection from the first
  item, a 6 h wall limit per process. The longest process was the documents, 8,197 s. The first session's six
  STOPPED records are kept under `superseded`. Two earlier in-session attempts were stopped by the session host's
  memory-pressure reaper and recorded nothing (ROADMAP §12).
- **Every panel** has return code 0, its CONFIG line confirmed, and the model hash equal to its pin.
- **F0 per panel (s):** nq 2,044; tqa 1,785; gsm8k 4,634; arc 4,874; mmlu 6,478; docs 8,197; chat 997. Against M0,
  the six panels took 2.3-3.5x. These are wall times on a box that was in use, not rates.

## Accuracy (reported)

| Panel | n | M0 | F0 | T0 | F0 fixed / broken vs M0 |
| --- | --- | --- | --- | --- | --- |
| nq | 600 | 30.2% | 39.3% | 29.0% | 94 / 39 |
| tqa | 600 | 46.7% | 64.5% | 48.7% | 130 / 23 |
| gsm8k | 100 | 77.0% | 85.0% | 81.0% | 13 / 5 |
| arc | 600 | 53.8% | 64.8% | 53.2% | 108 / 42 |
| mmlu | 570 | 49.5% | 56.0% | 49.3% | 79 / 42 |
| chat | 40 | 70.0% | 62.5% | 70.0% | 3 / 6 |

- **Documents (72 FIT documents, paired document bootstrap):** F0 - M0 dBPB = **-0.235** (CI95 -0.278..-0.198);
  T0 - M0 = -0.0052 (CI95 -0.0085..-0.0019).
- **The imatrix effect T0 - M0** on the item panels: nq -1.2 pt, tqa +2.0, gsm8k +4.0, arc -0.7, mmlu -0.2, chat 0.
  On the documents it is the -0.0052 above.

## Registered prediction, scored

- **RER_K > RER on gsm8k alone: WRONG.** RER_K = 0.219; gsm8k alone is (13 - 5) / 23 = 0.348. On 100 items, with 23
  M0 errors, the gsm8k figure is wide; it is scored as registered.

## Donor-gap panel for S4/S5 (written by the verdict)

Items F0 gets right and T0 gets wrong: nq 98, tqa 117, mmlu 79, arc 105, gsm8k 10. Documents where F0's BPB is at
least 0.05 below T0's: **71 of 72**. S4's protocol splits this panel before any training, and none of it may enter
training data.

## What the numbers say, and what they do not

- **The gap is broad, not a knowledge gap.** The larger donor helps short-answer recall (tqa +17.8 pt, nq +9.2) and
  the reasoning panels (arc +11.0, gsm8k +8.0) by about the same share of M0's errors. The rule cannot tell them
  apart at this n.
- **The documents move most.** -0.235 BPB on the FIT documents, against -0.005 for the imatrix (T0 - M0). Almost
  every document is on the donor-gap list.
- **Chat goes the other way** (25/40 against 28/40, 3 fixed and 6 broken). On 40 items this is reported only.
- **Limits stand** (protocol): two artifacts of one family at Q4_K_M without imatrix; proxy panels; not a capacity
  result. A larger stored capacity could help both families; S4 tests whether it does.

## Next

- **S0** ran the same day (v0 of T0), see [C50_S0 result](C50_S0_V0_RESULT_20261019.md).
- **F0 leaves C:** now that S2 is closed, to free room for S1's arms.
- **S3** is an owner decision (the teacher signal), then S4.
