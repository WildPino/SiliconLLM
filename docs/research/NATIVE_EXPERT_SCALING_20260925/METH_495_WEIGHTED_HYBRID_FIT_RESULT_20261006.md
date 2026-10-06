# METH495 — fixed hybrid fit: independently admitted local failure

6 October 2026. Goal ACTIVE/INCOMPLETE. All 8 main, 8 independent audit and
5 admission gates pass. Both frozen local recipe gates FAIL. This fixed
recipe is CLOSED; no update/width/prior/ID/checkpoint sweep is authorized by
its result. The broader hybrid architecture and transfer objective remain open.

## What was actually built

All 128 source IDs remain in a 127,232,136-byte physical bank. There are 127
development-only regularized readout solves; ID 0 has no development or
consumed-validation UID and retains its complete weight-derived prior. The
two Cartesian key heads receive exactly 128 full-development Adam updates.
The shared dictionary and private L bytes are unchanged from 494.

The native C evaluator implements group4 ternary LUT features, A16 activation
quantization, I64 reduction of safe I32 SIMD lanes, weighted I8 L/B outputs,
F32 scales/bias and ordered F64/F32 key scores and candidate mass. All
17,540 UID feature/output/ID/logit/mass bytes agree with the Python reference
and the independent auditor. It is a working candidate evaluator, not an
addition to `engine.c` or a qualified useful model.

The complete domain includes 11,721 development UIDs, 5,819 consumed-validation
UIDs, 19,962 occurrences, both modes, all 128 ID slots, 768 book/mode/control
views, 1040 metric groups and 384 source/candidate exposure groups. All actual
source controls are accepted. Validation remains consumed diagnostic data.

## Results: canonical UIDs

| Split | UIDs | ID correct | ID fidelity | Oracle-ID RMS | Coupled RMS | Decision-only RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 11,721 | 6,568 | 56.036174% | 2.798599% | 71.082371% | 71.241131% |
| Consumed validation | 5,819 | 1,321 | 22.701495% | 8.919153% | 75.804059% | 76.073224% |

RMS uses source weighted-function energy, not raw coordinate count or a mean
of per-token relative errors. Oracle uses the correct original ID. Coupled
uses the candidate choice. Decision-only compares candidate and oracle
functions at the same input; these squared errors are not mutually orthogonal.

## Frozen eligibility: all occurrence role/mode aggregates

| Role | Mode | Count | ID fidelity | Oracle RMS | Coupled RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 development | 0 | 3,584 | 53.627232% | 2.897652% | 69.066995% |
| 0 development | 1 | 3,091 | 58.589453% | 2.897074% | 69.059034% |
| 1 consumed validation | 0 | 3,584 | 22.767857% | 7.100491% | 75.465311% |
| 1 consumed validation | 1 | 3,065 | 24.274062% | 6.913422% | 75.417606% |
| 2 development | 0 | 3,584 | 52.873884% | 2.697434% | 72.843748% |
| 2 development | 1 | 3,054 | 57.105435% | 2.697108% | 72.661670% |

All six fail coupled RMS≤1% and ID fidelity≥99.9%. UID and occurrence
statistics differ because duplicates/mode exposures differ; neither replaces
the other. All 1040 groups and 384 exposures are in the raw record.

Rare IDs are retained rather than excluded: development exposure 1..4 has
7 development/3 validation UIDs; exposure 5..15 has 36/19. Their oracle RMS
is 9.2704%/182.8715% and 8.1111%/168.7595%, respectively. Coupled relative
RMS reaches 85.0004 and 32.5885 for exposure 1..4, and 77.6214 and 40.2326
for exposure 5..15 (ratios, not percentages). Their small source-energy
denominators matter. ID 0 has no observed UID in either split. The 128 private
blocks have distinct byte hashes; this does not demonstrate 128 useful functions.

## Numerical/optimization evidence

All 128 readout coefficient/empty-prior cases independently qualify. Maximum
normal-equation rounding/arithmetic-envelope ratio is 0.05168780270351749,
required≤1. The auditor independently accumulates data in blocks of 127.
All 128 Adam transitions are checked with block137 gradients; maximum theta
transition discrepancy is 1.841235497401783e-14, below the frozen tolerance.
Every final physical key/B/bias byte agrees. Source UIDs, occurrences, A16
codes/scales, dictionary and fixed L bytes also agree.

Final real key objective is 1.7189274544981379; computed gradient norm
0.07913276880452026. The strong-convexity gradient gap expression is
31.309975493348276. It is weaker than the trivial nonnegative-loss bound;
it does not establish near-optimality. Thus finite128 Adam failure does not
exclude the fixed feature/key class. The readouts solve their chosen
regularized objective, not an unrestricted unregularized function problem.

## Actual execution and costs

| Stage | Actual tool/session/terminal | Exit | Late wall | OS peaks |
| --- | --- | ---: | ---: | ---: |
| Binding after source freeze `0ab3a41` | `aee4a8` / 89779 / `f69d44` | 0 | 23.813 s | 53,239,808 B |
| Main after binding commit `22fb203` | `50fb0b` / 92589 / `7e6b57` | 0 | 252.125 s | parent 891,080,704 B + native/descendant 131,522,560 B |
| Audit after raw commit `02322b7` | `986f0d` / 82118 / `21c282` | 0 | 263.047 s | parent 1,150,582,784 B; native 0 |
| Audit Windows query/finalizer | `26d151` | 0 | metadata only | — |

Limits: builder90s/256MiB; main/audit1200s/1536MiB combined peaks; direct
child120s, CPU0/BLAS1; each new directory768MiB/all495 outputs1GiB/raw8MiB.
Hashing, compilation, new controls, native feature/prediction operations,
solves, all optimizer updates and complete record writing are charged. The
existing compiler snapshot has 5353 freshly hashed files (402,562,552 B).
Admission takes substantial time; it is included, not hidden as fitting time.

Native commands: compile0, new controls0, intentional negative magic2,
features0, predict0. Both Windows Event1000 queries are available and return
zero events. No first fault, namespace rerun or numbered repair. No GPU/new
resource/source FFN/model invocation. Engine and three foreign files preserve
their exact hashes. Main performs 127 readout solves/128 Adam updates; audit
performs no optimizer update or function solve.

The physical format/cost remains 494's: 14,587,008 logical weight bytes per
12-bank token, excluding core/head/state/other traffic. Native batch file
processing durations are not batch1 whole-model rate or DRAM evidence.
No composed source476 run, own-state evaluation, fresh generation/task,
SAME≥50 or useful-n/family promotion follows from this failure.

The binding retains ancestor metadata/provenance fields. The current 495
numerical configuration is its prospective protocol, `fit_inputs`, new helper
code and actual command/fit/update records; ancestor configuration fields do
not describe this execution.

Hashes:

- Binding `c2fac85bb7ace619f2de0031a53dfc30acdc8800824d5bc9ceb811fe77c4f3d4`.
- Main `6bb28ccf065afe0a132a0f9893a6d6e5f42395a7df3206ceea0cf9a473cd5a1b`.
- Audit `eb8523f9c86a7864fade4e49f1f56bc938d80795e0e0322925e65ab68874e342`.
- Admission `de9e28638e3abc6b9a512c7fdf7c6e6751f4505d9501d51d1e6763fd8e98280f`.

[Protocol](METH_495_WEIGHTED_HYBRID_FIT_PROTOCOL_20261006.md),
[main](meth495_weighted_hybrid_fit_result.json),
[audit](RETENTION_495_20261006.json),
[admission](ADMISSION_495_20261006.json),
[whole-project algebra and next](METH_495_WHOLE_ALGEBRA_AND_NEXT_20261006.md).
