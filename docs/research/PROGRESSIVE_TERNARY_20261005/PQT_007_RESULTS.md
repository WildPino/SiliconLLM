# PQT-007 results: full-matrix conversion does not preserve behavior

6 October 2026. **COMPLETE / RETAINED / AUDITED / ADJUDICATED.** The overall
research goal remains active. No tested arm passes all frozen gates. Both
ternary complete-matrix models lose nearly all source token agreement under
this source-guided, independently fitted matrix procedure. These findings do
not test end-to-end adaptation or reject every progressive ternary method.

[Protocol](PQT_007_PROTOCOL.md), [adjudication](PQT_007_ADJUDICATION.json),
[retention](PQT_007_RETENTION.json); exact small raw records in
`pqt_007_evidence/`, full arrays/archives in
`results/progressive_ternary/PQT-007/remote_001/`.

## Scope and reference

Original pinned Qwen2.5-0.5B coefficients, F32 SDPA reference, TF32/AMP disabled,
24 blocks. All 169 unique matrices are converted, including the shared token
embedding/readout: 493,961,216 coefficients. All 71,552 norm/bias vector
coefficients remain explicit F32. Tied storage is counted once per model.
This is complete matrix conversion in a counted mixed representation.

D uses direct group-64 symmetric ternary rounding. PR uses original-input
curvature compensation and two discrete coordinate passes, fixed group scales,
on every matrix. I4 and row-I8 are declared precision controls. No original
parameter receives gradients and no optimizer updates occur. PR fits the
shared embedding/readout using the original readout role, while inference
changes both its lookup and readout roles.

Fitting uses 2,048 positions from sixteen consumed WikiText train windows.
All final representations freeze before news selection or any news evaluation.
New-to-chain AG News test bytes were admitted independently of fitting; no
claim of unseen pretraining or exhaustive parent-project novelty. Prediction
rows: 301, 6594, 2327, 4122, 4351, 2183, 3136, 6864. Prompt rows: 3109, 1754,
5, 2586, 6999, 1230, 3104, 117. IDs belong to AG News, not earlier WikiText
window numbering. These sixteen news rows are now consumed evidence.

## Frozen whole-model outcomes

Prediction: 1,024 next-token labels in eight 128-position original contexts.
Generation: eight prompts, 32 own-prefix greedy tokens each. Main gates:
agreement >=99%, KL <=0.01 nats, NLL delta <=0.01 nats/token, generation
position agreement >=95%, >=7/8 exact traces, <=35% of full FP16 weight bytes.

| Arm | Prediction matches | Mean KL | NLL delta | Generation matches | Exact traces | Serialized bytes / FP16 | All gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D | 0/1024 (0%) | 9.981001 | +9.804997 | 0/256 (0%) | 0/8 | 154,877,981 / 15.6749% | Fail |
| PR | 4/1024 (0.3906%) | 11.052894 | +10.790920 | 0/256 (0%) | 0/8 | 154,879,336 / 15.6750% | Fail |
| I4 | 649/1024 (63.3789%) | 0.494719 | +0.464231 | 38/256 (14.8438%) | 0/8 | 278,369,640 / 28.1732% | Fail |
| I8 | 984/1024 (96.0938%) | 0.012238 | +0.006939 | 213/256 (83.2031%) | 6/8 | 496,301,783 / 50.2296% | Fail |

All arms pass the declared fitting-time budget. D/PR/I4 pass storage but fail
every quality gate. I8 passes NLL delta only among quality gates and exceeds
storage. I8 is substantially closer to the reference; its failed strict
behavior-preservation gates are not evidence of catastrophic task incapacity.
Conversely, near-original likelihood does not establish exact changed-state
generation preservation. Gates are unchanged after observing these outcomes.

Source perplexity on these labels is 20.8120; D 377,198.96, PR 1,011,000.63,
I4 33.1076, I8 20.9569. These are language-model diagnostics on the selected
news texts, not classification accuracy, semantic/factual ratings or broad
benchmarks. The ternary outcome is severe under the measured diagnostic.

## Local fitting improves while complete behavior fails

Descriptive statistics from the already frozen calibration records:

| Arm | Median matrix local relative RMS | Tied readout local RMS |
| --- | ---: | ---: |
| D | 44.2932% | 67.0640% |
| PR | 20.0593% | 16.3532% |
| I4 | 10.7019% | 14.6633% |
| I8 | 0.9564% | 1.21495% |

PR reduces local RMS for **169/169 matrices** compared with D, yet increases
whole-model KL/NLL and yields only four matching predictions. This directly
limits the assumption that independently improving original-input matrix
reconstruction suffices for complete-model preservation. The median is an
unweighted descriptive statistic, not a pooled whole-model error.

Accumulation of changed intermediate states, the tied lookup/readout coupling,
and residual matrix errors are plausible explanations. This experiment does
not isolate their causal contributions. Do not claim that the embedding alone
caused the failure, or diagnose a specific quantization bug from these outcomes.
Fresh causal interventions or explicit global behavior adaptation are needed.

## Independent verification and counted costs

Separate cuda:1 audit imports no fitting module. It rehashes source/data and
every scientific artifact; independently decodes every matrix and vector,
verifies all original coefficients and tied aliases, and reproduces all 169
original calibration-input raw hashes. Every local objective is checked with
F64 products; maximum source-energy-normalized F32/F64 discrepancy 1.562e-6,
limit 1e-5. This number is a numerical metric tolerance, not function fidelity.

All original and candidate teacher-forced states are reproduced **exactly**,
as are all 40 greedy traces: 1,280 chosen tokens and every pre-choice hidden
state. News selection and all F32-readout point metrics/gates reproduce exactly.
Additional F64 readout maximum relative RMS 5.472e-7, limit 1e-5; no F64/F32
argmax differences on these audited readouts. Standard-library adjudication
independently reaggregates raw points, traces, archive bytes and frozen gates.

Total fit/pack time 720.6853s; PR dominates at about 614.2s. Experiment process
937.6433s, independent audit process 396.8120s, installation 21.3808s. These
are process times, not measured service quota charges or CPU-native latency.
Peak CUDA allocated 3,109,350,400 bytes; final RSS 5,739,257,856 bytes.
Resident F32 reconstruction is used for behavior measurement; this GPU memory
is not the packed model's native execution memory. Each payload includes its
exact code/scale/vector NPY members, representation descriptor and all ZIP
headers; archives are uncompressed. FP16 reference is 988,065,536 bytes.

Private acct3 kernel 137240077/v1; source `71781d6`, sixteen actual embedded
files; bundle `18e998e`. Terminal COMPLETE at 23:15:00 UTC on 5 October (6
October local). All three bootstrap phases return zero. Monitoring delegated;
temporary monitor-service capacity failures were resumed without job mutation.
Retrieved once, 58 files / 1,111,912,098 bytes, every fetched size/hash verified;
4,357,851 bytes of small exact raw evidence retained. No scientific repair.
The initial CPU apparatus import failure and numbered import repair remain
retained separately. Never repush or retrieve this completed reference.

## Decision and resumption

No promotion. Do not extend this same independent-matrix recipe merely by
adding more adjacent held-out rows or interpreting local gains as capacity.
The next method decision should address complete-network behavior explicitly:
a bounded progressive global adaptation versus a matched one-shot adaptation
is a distinct, currently untested hypothesis. Freeze its stage schedule,
calibration-only optimizer budget, deployment codes/scales, independent token
identities, metrics and resource stops before accessing its outcomes. Preserve
current final archives as fixed comparators; never tune on these news rows.

PQT-006 packed expert CPU correctness is already qualified. Its ready latency
measurement still awaits an explicit uncontended CPU reservation from the
owner; silence/idle observations are insufficient. No useful native-capacity
claim from a faster failed representation. Full-model native integration and
broader validation remain separate requirements. Overall goal remains active.
