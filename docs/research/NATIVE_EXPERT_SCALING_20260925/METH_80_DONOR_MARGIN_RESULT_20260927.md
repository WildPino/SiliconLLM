# METH-80: donor-margin term helps E1280 but fails joint retention

**Decision.** The fixed E1280 development gate fails. A teacher-decision
margin term improves donor-position top-1 by 20/3,812 positions
(+0.525 point) against the exact no-margin METH-79 continuation,
clearing that individual benefit threshold. The E1280 candidate still
loses 55 positions (−1.443 points) against its own update-64 parent,
exceeding the permitted one-point decline, and ends 23 positions
(0.603 point) below the matched E128 candidate, exceeding the
0.5-point cross-E allowance. Do not promote it to external, semantic
or native integration.

The [protocol](METH_80_DONOR_MARGIN_PROTOCOL_20260927.md),
[24-prompt manifest](meth80_margin_dev_manifest.json) and
[runner](../../../benchmarks/donor_adaptation/s1/meth80_donor_margin_continuation.py)
were committed at `dcb7e13` before model inference. The manifest SHA-256
is `2aae788ce4a4f4aa45ef40c3d5bdd96e9790711633ba0a2f2e716548a9f24f89`.
It excludes all prior development sets through METH-79 and the full
matched raw draw stream. Both arms restored METH-71 update-64 model,
optimizer and RNG states and checked every actual raw/chat draw
against the frozen stream. Relative to METH-79, only the loss adds
a 0.2-logit teacher-confidence/target-margin hinge, weighted 2 on
raw and 4 on full-chat examples. The E1280 and E128 METH-79 update-256
checkpoints were scored on these same fresh prompts before training.

| New-prompt readout | E128 | E1280 |
|---|---:|---:|
| Frozen update-64 parent | 3,726/3,812 = 97.744% | 3,756/3,812 = 98.531% |
| METH-79 no-margin comparator | 3,725/3,812 = 97.717% | 3,681/3,812 = 96.563% |
| METH-80 margin candidate | 3,724/3,812 = 97.691% | 3,701/3,812 = 97.088% |
| Candidate versus comparator | −1 position = −0.026 point | +20 positions = +0.525 point; benefit pass |
| Candidate versus parent | −2 positions = −0.052 point; pass | −55 positions = −1.443 points; **fail** |
| Minimum selected experts/layer | 128/128 | 1,126/1,280 |
| Worst maximum/mean route load | 3.86× | 19.73× |
| Minimum changed B slots/layer | 128/128 | 1,277/1,280 |
| Raw held-out ΔBPB versus donor | −0.006597 | −0.005685 |

The E128 control meets its preregistered gates, including the
no-worse-than-0.5-point comparison. E1280 clears the absolute ≥95%
top-1, raw BPB, changed-slot and route-coverage/load gates. Its
0.603-point gap to E128 also fails the fixed cross-E test. On each
arm's first training update, 573 positions met the teacher-confidence
criterion and 18 violated the student margin across four microbatches;
at the last update the counts were 586 and 14. Thus the added term
was active in the bound training stream, but did not hold the broader
donor behavior within the registered allowance.

The [E128 raw result](meth80_e128_result.json) was committed at
`03d2b2b` before the E1280 run, and the
[E1280 raw result](meth80_e1280_result.json) at `afdded2`. Both
contain all 192 continuation updates' CE/KL/margin values, active
position counts, gradients, prompt counts and per-layer route load.
Terminal checkpoint SHA-256 values, independently reread from disk,
are `5a5ccebf883073c5201d28e6a3428ee5c1fa168890f71336bd0a33061fb81f40`
for E128 and
`aeeba3c88fad3724d5f0f645696e61d415d8b19903a68c96f8fb3fc09f3209b1`
for E1280. Runtime was 390.8/512.7 seconds, GPU peak
2.581/10.231 GB, and final RSS 2.910/4.716 GB; no T4 was used.

The single-seed, 24-prompt result identifies a useful directional
effect of this margin rule, not a general retention law or semantic
improvement. METH-72/74's external utility and semantic failures
still stand. No new pair-permutation, generation, task, blind semantic,
CPU LUT or complete `engine.c` rate result was obtained. Rather than
continue tuning top-1 on already viewed prompts, the next method
step must connect a quality-valid compact core to a fully executable
artifact while separately developing a stronger, precommitted
large-E training rule on untouched evidence.
