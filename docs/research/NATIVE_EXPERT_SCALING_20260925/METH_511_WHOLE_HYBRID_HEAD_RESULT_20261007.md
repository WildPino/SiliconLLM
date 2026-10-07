# METH511: fresh whole quality and warm rate; fixed cost recipe fails

7 October 2026. Goal ACTIVE/INCOMPLETE. All scientific phases terminal.
[Final evaluation](EVALUATION_511_20261007.json), SHA256
`2d698cec6755589a0e9e0272fb69999a8064d9a3975b22ca586fa4eab7e38546`.
The numerical auditor passes all eight controls. All fifteen prospectively fixed
quality criteria and the SAME-artifact warm accepted-rate criteria pass.
Four of five economic criteria pass; the all-book cost criterion fails. The
fixed recipe is ineligible. No criterion is relaxed and the goal is not complete.

## Method and declared scope

[Frozen protocol](METH_511_WHOLE_HYBRID_HEAD_NEXT_20261006.md) applies the qualified
510 top-eight I8-shortlist/F32-row readout to the complete 506 conditional WO
artifact. Original full WI, original normalized parent routing and upstream
operators are retained. All coefficients used for refinement already exist in
the donor's shared embedding. No additional learned donor capacity is claimed.

Actual donor: original Switch base128, 7,415,217,408 distinct parameters, twelve
sparse banks with 128 original parents each. Logical aliases are not new distinct
parameters. Original F32 reference is Transformers 4.57.6, CPU1, Torch 2.6.0+cu124,
GPU uninitialized. All 3,320 canonical tensor names and 29,956,961,280 canonical
F32 bytes are SHA-verified before inference. All 384 official/reference teacher
and own-history generation bridges agree in choices and meet relative L2<=1e-6.

One deterministically selected project-excluded PG19 cohort: 24 books, four
bounded infilling tasks each, source length29, teacher length14, generation cap64,
fixed closing ID32095. 383 previously exposed source rows are excluded. Two
retained full-text/tokenizer controls agree BYTE. Project exclusion does not
establish exclusion from donor pretraining. This is an infilling scope, not a
general chat or open-domain reasoning qualification.

Native hardware: Ryzen53600X, 80GiB RAM, three actual physical workers on logical
0/2/4. Same original source executable and candidate executable for all quality
and performance observations. Source binary SHA `a4ad2bd4..`, candidate
`b49c153c..`; source payload `6bee473e..`, conditional payload `07d10d07..`.
Both payloads contain 7,541,946,880 bytes; full identities are in the evaluation.

## Independently verified quality

| Predefined comparison to original F32 donor | Mean / agreement | Relevant bound | Criterion |
| --- | ---: | ---: | --- |
| All-position NLL difference | +0.02847173 | upper95 +0.03405880 | <=+0.05 PASS |
| Masked-position NLL difference | +0.00833092 | upper95 +0.01298048 | <=+0.05 PASS |
| Masked teacher token accuracy difference | -0.00390625 | lower95 -0.00781250 | >=-0.02 PASS |
| Teacher field exact difference | 0 | lower95 0 | >=-0.05 PASS |
| All / masked teacher top1 agreement | 0.98735119 / 0.98958333 | direct | >=0.95 PASS |
| Generated known-token accuracy difference | +0.00260417 | lower95 -0.00390625 | >=-0.02 PASS |
| Generated known-field exact difference | +0.00520833 | lower95 0 | >=-0.05 PASS |
| Normalized generated prose edit to donor | 0.03436673 | upper95 0.04761987 | <=0.10 PASS |

All remaining fixed signal, health, LCS and repetition criteria also pass.
Donor known-field exact signal is 0.21354167; both donor and candidate health
95/96. Paired uncertainty uses 10,000 book-bootstrap draws, seed351351. All
cases, including unhealthy cases, remain in the metrics and denominators.

The independent auditor reconstructs all 4,983 head states / 160,093,824 full
head cells, exact I8 integer and native AVX F64-to-F32 refinement, route masses,
IDs, repeat controls and quality/rate/economic decisions. No unrefined-tail
overtake occurs on this finite cohort. Forced-teacher upstream encoder, decoder
and routes are BYTE equal between source and candidate. Fourteen of 96 own
generations differ across arms; all five generation wires within each arm agree.
Earlier 506 used another cohort: its edit score cannot be subtracted from 511
to claim a causal readout improvement. Same-cohort retained diagnosis is next.

## Complete warm denominators and cost

| SAME artifact, batch1 | Accepted IDs | Warm rate IDs/s | lower95 |
| --- | ---: | ---: | ---: |
| Candidate, ordinary including sentinels | 1142 | 89.98790 | 86.50956 |
| Candidate, prose only | 667 | 52.55861 | 50.46847 |
| Original I8 source, ordinary | 1135 | 84.70650 | 81.17145 |
| Original I8 source, prose only | 660 | 49.25664 | 47.15919 |

For each case retain one untimed warmup and all three timed repetitions; the
denominator is the sum of case mean encoder + cross-KV + greedy decoder times,
including router and head. Candidate12.69059487s, source13.39920843s. Failed cases
contribute zero accepted IDs and all their time. Each arm uses its own IDs.
Tokenizer/frontend, model load and profile-output serialization are outside the
declared warm native interval. No profile-clock subtraction is made.

The first request with load and warmup charged gives candidate prose27.13819/s
and ordinary46.46448/s. The warm result is not a first-request qualification.
Book-bootstrap rate/cost intervals use 10,000 draws, seed485485.

Matched elapsed ratio of sums0.94711527 (5.28847% lower observed total time),
upper95 0.99166937: both aggregate criteria pass. Six zero-based books
`[1,3,15,19,20,22]` regress; worst ratio1.25869861. All-book criterion fails.
Different generated lengths and measured variability must be examined before
attributing these tails to conditional columns or routing.

## Execution, interruptions and retained costs

All 576 C subprocesses ran once and exited0. Original R6 completed553 before
call554's untimed preflight observed an idle service CPU tick0.015625s and stopped
before process creation. [R7](METH_511_R7_RETAIN_CLOSED_CALLS_20261007.md) retains
the553 closed calls and runs only23 remaining calls after rebinding idle identity.
This is two separately bound quiet intervals, with the first controller's
interruption explicit. Sampled idle/topology evidence cannot exclude all OS work.
No completed C, donor, cohort or numerical audit observation was replayed.

| Phase | Actual seconds | Peak bytes | RAW SHA prefix |
| --- | ---: | ---: | --- |
| Fresh cohort R3 | 8.547 | 735,428,608 | 90b790c2 |
| Native R6 through first preflight fault | 1290.281 | parent pre-fault122,990,592; native1,325,694,976 | 1384c005 |
| Native R7 remaining23 | 54.828 | parent159,424,512 | 278df0c1 |
| Donor R8 complete | 1027.328 | 5,845,938,176 | 4b731686 |
| Independent numerical auditor R9 | 72.156 | 755,077,120 | 48e95e9f |

R7 donor import fault51.203s/544,870,400B occurs before tensor load/inference;
[R8](METH_511_R8_TORCH_IMPORT_ADMISSION_20261007.md) admits105 existing Torch
testing sources without operator/version changes. R8 auditor's old new-call
count assertion faults before numerical work9.203s/98,525,184B;
[R9](METH_511_R9_AUDIT_CALL_COUNT_20261007.md) changes only retained call counting.
Combined native1345.109<2400s, donor1078.531<3600s, audit81.359<1200s.
Runtime assembly116.984s, original canonical input/exposure binding239.469s,
and metadata rebindings remain separately priced in their receipts. Finalizer
physical output inventory5,001,280,125B excludes foreign junction targets.

[Typed UTC log receipt](meth511_r9_windows_terminal.json), SHA `34e4de5a..`,
queries all four stages with available positive controls179810/179791 and finds
zero relevant Event1000 matches. All stage parents and576 native process
instances are closed. Exact foreign file SHA checks pass. Two metadata finalizer
faults (positive-control list order, Git newline-normalized diff membership) are
preserved; the literal R2 repair makes no numerical/model/C call. Its successful
terminal guard records1.000s/91,324,416B. Exact first-fault PID/peak is unavailable.

## Decision and next

Keep the exact conditional WO and qualified fixed-eight original-row readout as
components of a useful warm whole baseline. Close THIS fixed economic recipe
with its failed all-book criterion. The result qualifies fresh quality AND>=50
warm accepted prose IDs/s only in the declared scope.

Next: bounded retained-data diagnosis of same-cohort readout recovery and cost
components, no new inference. Then return to compact conditional representation
and useful larger expert pools. Full source-sized WI, original128 parents,
unverified physical DRAM, missing compact core/LUT winner+mass scaling and
additional actual families/scales remain goal requirements. Further local timing
polish alone does not resolve those requirements.
