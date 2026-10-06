# PQT012 results: nonlinear output residuals do not meet the fidelity gate

6 October 2026. **No candidate advances.** All40 artifacts and120 metrics
completed under the frozen protocol; the independent NumPy/F64 audit passes.
Every ternary arm fails the absolute1% output-RMS requirement. The int8 control
passes fidelity but exceeds the35% FP16 artifact-storage limit. This closes
the tested residual hypothesis on the declared four expert functions, not the
full research goal or all possible pretrained ternarization methods.

## Evidence and scope

[Preregistration](PQT_012_PREREGISTRATION.md),
[exact original-input preparation](PQT_012_PREPARATION_RESULTS.md),
[execution protocol](PQT_012_EXECUTION_PROTOCOL.md),
[complete raw retention](PQT_012_REMOTE_RETENTION_001.json),
[local gate reaggregation](PQT_012_ADJUDICATION_001.json).
Scientific source31aecfa9; bundle85e63bdf; original export/raw31aecfa9.
Kernel137357524 v1 completed16:32:59 UTC, monitored without scientific edits.
One terminal fetch completed16:49:15 UTC. Full fetched994 files635,606,145B
and all19 client-operation/clock files retained as1,013 entries635,782,256B in
three lossless archives. [Actual archive Git proof](PQT_012_REMOTE_GIT_001.json)
passes against evidence commita9ddda48: every committed retained member equals
its complete original fetched/client bytes. No raw file was omitted.

Initial staged-archive identity checking rejected inherited Git text
normalization before an evidence commit. A binary attribute confined to this
owned evidence folder and explicit renormalization restore all three staged
archive hashes to their original values. No working raw/archive bytes were
changed or experimental call repeated. The check on actual committed bytes
was performed independently of this staged check and passes ata9ddda48.

Original float Switch-base-128 encoder11 experts81/87/18/91 use the fixed
natural calibration/development/held-out contexts, original capacity/routing
and full WI/ReLU/WO function. There are1,024/992/940 rows across the four experts.
These are consumed diagnostic inputs, not fresh final confirmation or a
whole-model evaluation. No optimizer, pretraining, forced routing, attention
operation or native timing occurred in this expert-only screen.

## Fixed-arm comparison

Ranges below cover the four experts separately; they are not pooled task
scores. RMS is relative to original output energy. Storage includes the actual
complete exported artifact and is a fraction of the9,437,184B FP16 expert pair.
All four experts must meet the point and97.5th-percentile document-bootstrap
RMS<=1% on both development and held-out, with complete storage<=35%.
Residual arms must also meet their predeclared paired improvement gate.

| Arm | Calibration RMS % | Development RMS % | Held-out RMS % | Complete storage % FP16 |
| --- | ---: | ---: | ---: | ---: |
| D | 48.05–72.63 | 48.13–73.09 | 47.78–74.08 | 15.64 |
| PR | 7.11–35.46 | 7.22–54.09 | 7.47–55.18 | 15.64 |
| D-LR32 | 0.81–30.35 | 2.14–45.81 | 3.01–45.91 | 17.77 |
| D-LR128 | 0.27–12.83 | 2.04–48.81 | 3.04–49.19 | 24.02 |
| D-LR256 | 0.27–0.44 | 2.02–50.38 | 3.04–50.85 | 30.53–32.35 |
| PR-LR32 | 0.43–16.59 | 1.25–49.82 | 2.34–51.14 | 17.77 |
| PR-LR128 | 0.24–7.36 | 1.20–50.35 | 2.35–51.78 | 24.02 |
| PR-LR256 | 0.23–0.25 | 1.18–50.84 | 2.35–52.22 | 30.53–32.35 |
| I4 control | 5.73–10.29 | 5.74–10.39 | 5.77–10.17 | 28.14 |
| I8 control | 0.18–0.86 | 0.18–0.89 | 0.18–0.86 | 50.18 |

D/PR are symmetric group64 ternary, respectively direct and progressively
compensated. LR arms additionally execute an F32 low-rank output correction
and bias; they are mixed representations, not pure ternary models. Their
effective rank is capped by calibration row count. Integer controls have
their own explicitly packed alphabets.

No ternary point estimate reaches1% on either evaluation split for any selected
expert/arm, so bootstrap requirements cannot rescue them. Int8's worst upper
bootstrap RMS is0.9213% across development/held-out and passes fidelity. Its
50.18% storage exceeds the declared cap. I4 fails fidelity. Every residual arm
also fails the joint relative-improvement requirement on at least one of the
fixed expert/split combinations. Full exact values and per-expert outcomes
are retained in the adjudication and original independent audit metrics.

## What the experiment establishes

Progressive compensation changes reconstruction substantially and sometimes
helps: expert81 held-out RMS falls from74.08% direct to7.47% progressive.
It is not uniformly beneficial: expert91 worsens from51.27% to55.18%.
No aggregate average substitutes for the all-expert gate.

Rank256 residuals fit calibration tightly but fail generalization at this
coverage. D-LR256 calibration RMS is0.27–0.44%; held-out is3.04–50.85%.
PR-LR256 calibration is0.23–0.25%; held-out is2.35–52.22%. Increasing rank within
this screen does not establish improved verification fidelity. This is evidence
of a calibration-to-evaluation gap, not proof that output regression can never
generalize with more or different admissible calibration.

The original-function witnesses and int8 control distinguish approximation
failure from numerical noise or a universally unreachable1% gate on these
functions. All120 packed F32/F64 forwards agree within maximum relative
RMS3.6044236e-7; original function agreement maximum2.6559169e-7. The auditor
verifies eight ridge equation/SVD replays and actual all40-sealed-before-
evaluation input phase order. Shared metric/codec helpers have independent
scalar/golden qualification; the audit is not a second full progressive
algorithm implementation.

## Actual cost

Pinned Python3.13.15/Torch2.11.0+cu128/NumPy2.1.3, both physical TeslaT4 devices,
single CPU thread, F32 with TF32/AMP disabled. Eight progressive matrix fits,
15,360 committed columns,16 Cholesky calls/eight inverses, eight ridge solves/
SVDs,24 residual exports,120 artifact/split predictions and12 original function
checks. Whole-model forwards, optimizer updates and SDPA calls are zero.

Complete install/worker/audit children:2.621652/45.841835/27.695173s, all exit0.
Complete server wall76.170852s. Worker before-report35.421171s and process peak
RSS1,856,229,376B; complete worker child cumulative peak1,888,755,712B includes
later report/wrapper work. Audit before-report26.992266s/RSS1,217,064,960B.
Peak GPU allocations280,032,768/279,860,736B; reservations316,669,952B each.
Fit-through-all40-sealed phase21.515092s precedes evaluation. These clocks
measure fitting/validation overhead, not native deployment throughput.

Private dataset creation26.978443s; kernel push11.618199s; exact initial
dataset/kernel statuses4.478235/4.573741s. Single terminal fetch942.676553s,
994 files635,606,145B. Download latency dominated this small experiment's
postprocessing. Lossless terminal retention before-report12.094s/RSS27,570,176B;
distinct complete retention child12.271163s/exit0. Original preparation/export/
controls, admission stops and monitoring costs remain in their separate records;
these totals do not relabel them as model-training time. GPU quota billing is
not measured by these wall clocks.

## Decision and next justified hypothesis

Reject promotion and native benchmarking of all these artifacts. Preserve the
negative result; do not tune residual ranks, ridge strength or rows on the
observed development/held-out outputs, and do not rerun the same fit unchanged.
Whole-model quality/native benefit remains unverified for a successful candidate.
There is no candidate here for that next stage.

The literature identifies a distinct untested mechanism: matched rowwise
asymmetric versus symmetric levels with compensated rounding on the same
original contexts. Proceed toward one bounded [PQT013 comparison](PQT_013_DIRECTION.md),
with complete prospective implementation/execution and independent controls
before fitting. Its objective is to isolate alphabet/granularity effects, not
to rescue PQT012 after observing outcomes. Special-token filtering requires
separate qualified calibration-only metadata. Preserve the1%/35% gates.

If that justified finite comparison and any explicitly motivated remaining
mechanism fail jointly, a scoped negative research closure is valid. Do not
claim universal impossibility or silently replace teacher fidelity with a
looser task score. No new experiment is dispatched by this results record.
