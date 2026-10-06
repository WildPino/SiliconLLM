# PQT-008: global adaptation does not preserve the complete model

6 October 2026. **Audited and adjudicated; no arm promoted.** This is a scoped
negative result for the frozen 256-update procedure, not a rejection of every
progressive ternary method. The full research goal remains active.

## Decision and independent evaluation

All 169 unique matrices, including the shared embedding/readout, are ternary
in D, ONE and STAGED. The original 71,552 scalar vector coefficients remain
F32, frozen and counted. ONE converts every matrix before adaptation. STAGED
converts the shared embedding/readout first and then seven matrices per block
in eight-update stages. Both start at the original pretrained checkpoint and
apply exactly 256 Adam updates to every latent matrix on identical calibration
windows. ONE has 256 fully-ternary updates; STAGED has 64. No latent float
matrices, optimizer state or extra readout survive in the deployed archives.

New AG News identities were selected only after final payloads froze, excluding
all sixteen consumed PQT-007 rows and their first-32-token prefixes. This is a
new selection from the same provider split, not untouched project-wide holdout.
No labels or candidate losses influenced selection. Prediction has 1,024
positions; generation has eight own-prefix continuations of at most 32 tokens.

| Measure | D | ONE | STAGED | Absolute gate |
| --- | ---: | ---: | ---: | ---: |
| Source argmax matches | 0/1024 | 30/1024 | 40/1024 | >=99% |
| Source argmax agreement | 0% | 2.929688% | 3.906250% | >=99% |
| Mean source-to-candidate KL | 11.832555 | 7.886773 | 7.633910 | <=.01 |
| NLL delta, nats/token | +11.821253 | +7.939810 | +7.712473 | <=.01 |
| Candidate perplexity | 910,907.44 | 18,783.87 | 14,964.22 | Diagnostic |
| Matching generation positions | 0/256 | 3/256 | 7/256 | >=95% |
| Exact continuations | 0/8 | 0/8 | 0/8 | >=7/8 |
| Complete archive bytes | 154,878,076 | 154,880,786 | 154,884,851 | <=345,822,937.6 |
| Fraction of full FP16 parameter bytes | 15.674879% | 15.675153% | 15.675565% | <=35% |

Source perplexity on these contexts is 6.692208. Perplexity measures this
token-prediction screen; no semantic, factual or AG News classification score
is implied. Unmatched generation tails count as mismatches. D terminates after
226 actual generated tokens; its position denominator remains 256.

Both adaptations improve over D on this evaluation: held-out KL falls by
33.3468% for ONE and 35.4838% for STAGED. STAGED has 3.206168% lower KL than
ONE, with better NLL, argmax and generation-position agreement. The prospective
relative schedule criterion required **at least 10% lower KL** at no worse
other measures and identical raw parameter budget. It fails the KL criterion;
all four other relative criteria pass. Every absolute quality gate fails for
all three arms. Storage and the fitting-time bound pass. Relative improvement
does not establish preservation or useful native capacity.

## Final calibration differs sharply from new-context behavior

| Final frozen calibration measure, 2,048 positions | D | ONE | STAGED |
| --- | ---: | ---: | ---: |
| Argmax matches | 4/2048 | 1424/2048 | 1413/2048 |
| Argmax agreement | .195313% | 69.531250% | 68.994141% |
| Mean KL | 9.008851 | .529333 | .480911 |
| NLL delta | +8.862434 | +.281006 | +.106974 |

These are metrics of the final hard exported checkpoints, independently
replayed; they are separate from the pre-update online losses. The final
wrapper and exported plain model produce exactly the same calibration states.
Each arm exposes 32,768 positions to the optimizer, but these are sixteen
passes over **2,048 unique positions**, not 32,768 independent samples.
Final calibration itself still fails the strict predictive tolerances.

The large calibration/evaluation gap is consistent with limited calibration
coverage and possible distribution effects. Neither cause was isolated:
calibration uses WikiText train, evaluation uses news; fixed scales, surrogate
gradients, precision constraints and schedule exposure can also contribute.
Do not label this a causal diagnosis of overfitting, an embedding bottleneck,
or proof that simply increasing updates would solve it. No new same-domain
validation was evaluated in this run.

Exported hard codes actually change: ONE changes 55,604,260 coefficients and
STAGED changes 56,742,841 relative to D. Every unique matrix has a recorded
final counter of 256; all counters were checked after every actual update.
The audit checks these histories/counters and final code changes; it does not
replay the complete optimizer trajectory.

## Costs, qualification and exact retention

ONE fitting/export/final-calibration costs 301.164056 s; STAGED 262.423226 s.
The combined adaptation/export bound covers 564.397808 s, limit 7,200 s.
Prior source-grid/D-export and teacher-cache phases cost 8.787687 s and
1.983194 s; total elapsed before adaptation is 83.837612 s. Experiment internal
elapsed is 733.461148 s. Process costs are install 18.420383 s, experiment
750.573547 s, independent audit 240.579280 s. These phases are reported
separately; they are not native CPU latency. Peak allocated CUDA bytes are
10,411,631,104 on the student T4 and 3,392,713,728 on the teacher T4; final
process RSS is 5,963,198,464 bytes. Fitting and validation execute reconstructed
F32 weights, while deployed parameter raw data is 154,649,088 bytes per arm.
Archive/NPY headers and descriptors explain differing full archive sizes.

The independent process imports no fitting module. It verifies every original
parameter, tied alias, fixed scale, vector, teacher calibration logit, selection
and exclusion identity. It replays all final calibration/evaluation states,
all 32 source/candidate traces and 994 actual generated choices/pre-choice
states exactly. F32/F64 readout relative RMS is at most 5.679878e-7 versus a
1e-5 limit; no diagnostic F64 argmax difference occurs. An audit pass establishes
the numerical measurement, not preservation.

Private acct3 kernel 137244955/version 1 completed at
2026-10-06T00:06:47.116696Z. Scientific source
`a6d1873e75725450631614e2ee7eda184495860b`, bundle `6b354fa`. Single terminal
retrieval retains 86 files / 1,773,905,418 bytes; all sizes/hashes match and
all eighteen actual server source files match committed Git blobs. Forty-five
small raw files / 2,048,123 bytes are preserved in `pqt_008_evidence/`.
Large immutable payloads/states/logits remain at
`results/progressive_ternary/PQT-008/remote_001/`.
At evidence commit `69e6ff4`, all 45 small raw files and 2,048,123 bytes match
both the immutable fetch identities and working bytes exactly in Git; see
[committed-byte verification](PQT_008_GIT_VERIFICATION.json). Commit unsigned.

The first local launcher validation rejection occurred before authentication,
intent or network dispatch. Its numbered operational repair only increased
the launcher ceiling to the preregistered 10,800 s server budget. No scientific
repair or extra fitting updates were applied; original failure records remain.
Never repush this completed reference or refetch into the existing namespace.

## Reproduction, consumed identities and next dependency

See [protocol](PQT_008_PROTOCOL.md), [adjudication](PQT_008_ADJUDICATION.json),
[retention](PQT_008_RETENTION.json), actual raw
`pqt_008_evidence/PQT-008/independent_audit.json` and bootstrap phase records.
The standard-library adjudicator independently reaggregates raw prediction,
generation and final calibration points, checks actual step/stage evidence,
archive structure/identities, all absolute gates and the relative criterion.
It uses no model or fitting code:

```text
<owned-operation-python> -B scripts/progressive_ternary/adjudicate_pqt008.py
  --run-dir results/progressive_ternary/PQT-008/remote_001
  --retention docs/research/PROGRESSIVE_TERNARY_20261005/PQT_008_RETENTION.json
  --out <new-exclusive-adjudication.json>
```

Prediction rows 4498,5309,4782,1958,5181,5944,5403,5727 and prompt rows
789,7,6790,6,5333,5162,6306,2754 are now consumed. Exclude them and their prefixes
from future fresh evaluation; retain the previous PQT-007 exclusions too.

The next informative method question is calibration breadth at the same
applied-update budget, with separate fresh same-domain and news evaluations.
Freeze an executable protocol before any new selection or fitting. This can
distinguish evidence about transfer from final fitted-set gains; it cannot
be presented as a proven remedy. Original expert preservation, broader
capability and same-artifact native usefulness remain unproven. Native timing
still awaits the owner's explicit uncontended CPU window.
