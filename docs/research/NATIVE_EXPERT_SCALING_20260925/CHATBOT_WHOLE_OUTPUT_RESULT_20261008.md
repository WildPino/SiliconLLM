# Actual whole-output adaptation: improved, fixed recipe closed

8 October2026. Goal ACTIVE/INCOMPLETE. ONE fixed final1280-update checkpoint;
no intermediate choice, new source acquisition, reserved endpoint answer or C
execution. The first complete-output learner is implemented and numerically
adopted; this particular recipe does not qualify a deployable chatbot.

## Fixed configuration and actual execution

[Protocol](CHATBOT_WHOLE_OUTPUT_PROTOCOL_20261008.md), source/protocol freeze
`39d01413d37994bfd079dcb8c4e92ab083b5281b`, actual fit freeze
`91baa570e88a9b0b4240720a3ea6c1a9a8051959`.
Binding SHA `e351c828346fb9ab2d473ac777a708f3aed87e4dd02a93a799f1c217b41b6463`,
507 input files/72 safe runtime roots. Python3.12.10/Torch2.6.0+cu124/
Transformers5.13.1/Tokenizers0.22.2/NumPy2.4.6, local RTX306012GiB/Ryzen53600X.
Launcher `chatbot_directional_launch.py` used
`--binding docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_whole_output_fit_binding_20261008.json`,
the SHA/freeze above, directory `results/native_expert_scaling/chatbot_whole_output_fit_20261008`
and RAW `chatbot_whole_output_fit_20261008.json`; complete exact worker argv
retained in the paired terminal receipt.

Frozen218 BF16 source core tensors/tied full vocabulary head; ALL24 compact
shared512+selected4x128 SwiGLU blocks,16 parents/query32/frozen F32 routing.
165150720 trained F32 G/U/D elements/1224 objects, fully resident gradients and
AdamW moments. Fixed8epochs/1280updates, manifest FIT order repeated, batch1,
lr1e-4,clip1,T1 full-vocabulary teacher-to-student KL. Cached qualified teacher
vectors; each student forward recomputes its complete hidden states/routes.

Actual counts:200 initial+1280 training+200 final=1680 full student forwards,
1280 backward/update calls,8 complete epochs. Source forwards/endpoints0.
First actual loss3.36528682709,global gradient norm311.367523193; ALL24 layer
norms positive. Fixed final24 coefficient files and ALL initial/final logits
retained:33 files,2547890438 bytes. Fit pre-final-serialization elapsed3789.015s/
family3823.578s;
worker OS peak THROUGH EXIT3034857472B/family3068379136B;GPU peak allocated
3171494400B/reserved3653238784B. Original time/memory/output gates PASS,exit0.

## Independently adopted final comparison

FIRST CPU-only auditor freeze `45d4e99d5f1f0c3d86cb60936205df618453fdd1`,
binding SHA `2f863b86e464e3b9db663d5eeb5dfa927275dce3bba44bb6da7438897494506d`,
543 inputs/29 roots. ALL400 saved cases scored independently in F64; masks,
offsets, argmax IDs/source-ID probabilities and summaries checked. Results:

| Split | Initial KL | Final KL | Initial ID disagreement | Final ID disagreement |
|---|---:|---:|---:|---:|
| FIT160/2437 labels |5.66691596351|0.431673607410|79.40090%|15.88018%|
| DEV40/613 labels |5.57413381935|0.554575460311|79.44535%|18.43393%|

| DEV category | Final KL | Final ID disagreement |
|---|---:|---:|
| Arithmetic |0.373854284485|16.25%|
| Code |0.324083421179|22.50%|
| Extraction |0.264972821884|7.50%|
| Factual text |0.237111869077|16.25%|
| History |0.662942248185|24.00%|
| Instructions |0.702325007471|18.66667%|
| Multilingual |1.45095533863|29.87013%|
| Translation |0.452035760481|12.12121%|

Two relative KL<=20% initial gates PASS. Six absolute/category gates FAIL:
FIT/DEV KL<=.10,FIT/DEV ID disagreement<=.15,each DEV category KL<=.20 and
each category disagreement<=.25. All original criteria unchanged. Actual
verdict CLOSE_FIXED_WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT is independently
confirmed; candidate_transfer_eligible=false. This closes this objective/
schedule/representation recipe, not a universal conditional-capacity bound.

FIRST lossF643.36528661581,global normF64311.367521772;logit gradient maximum
gap0.000102188519 lies in the prewritten BF16/F32 envelope. All1224 recorded
parameter gradient norms/24 positive layer norms,3672 sampled clip/Adam
coordinates,1280 ordered updates/1680 forward counts and ALL165150720 final
finite elements/unchanged NEW buffers checked. Maximum saved F32/F64 per-label
KL gap1.69238729e-5/source-ID logprob1.843751e-6 within original envelopes.
Not independent full-network differentiation, every Adam coordinate, actual
route arithmetic, device allocator reread or fresh behavior verification.
Audit compute59.563s/family87.781s/worker OS667574272B/family699789312B,exit0.

## NEW saved-only diagnosis and next bounded question

[Drift protocol](CHATBOT_WHOLE_OUTPUT_DRIFT_PROTOCOL_20261008.md), actual freeze
`e0c276ba03f48d233fdbe67b0fe56f1c0d2b58a4`,binding SHA
`33cf0d5e302b8da64750f332af59041385c17559ca2fe6b046b480a0697b3c4c`,
18 inputs/2 psutil roots. New analysis only of the adopted three JSONL files;
no model/gradient/update/old audit replay. Final FIT **case** mean0.42633201977
versus last epoch online case mean0.212765548564, so unequal label weighting
alone does not explain the gap.

| Category in original update order | Last epoch online case KL | Fixed final case KL |
|---|---:|---:|
| Arithmetic0..19 |0.192859|0.262029|
| Code20..39 |0.229869|0.335255|
| Instructions40..59 |0.217326|0.471523|
| Factual60..79 |0.153328|0.214750|
| Translation80..99 |0.189624|0.429424|
| Multilingual100..119 |0.408706|1.343083|
| Extraction120..139 |0.111256|0.247870|
| History140..159 |0.199158|0.106721|

7/8 category means increase after their online observations; last history mean
decreases. Category/update position/state/optimizer effects are confounded;
these are not causal forgetting estimates or fixed-checkpoint learning curves.
They motivate [one order-stability control](CHATBOT_WHOLE_OUTPUT_STABILITY_NEXT_20261008.md),
before increasing capacity or number of epochs.

Initial whole student:ALL384 leaves selected on FIT AND DEV. Final:6 leaves
have0 occurrences on each split:layer6/id6,11/id0,19/id0,20/id0,22/id7,23/id0.
Maximum initial/final normalized occurrence-distribution total variation:
FIT0.330678164279/DEV0.333073929961;layer0 exactly0 both. Counts are repeated
input rows, not distinct states, per-row ID/mass agreement or useful capacity.
On first generated decision,disagreement73/160 FIT and19/40 DEV; overall mean
disagreement hides that concentration. This still is teacher-forced output,
not a freshly generated conversation or a retrospective new pass threshold.
Diagnostic compute.532s/family2.250s/worker OS32083968B/family62275584B,exit0.

## Pipeline state and retained provenance

The available conversion stages now include actual ALL24 initialization,
installation, resident optimizer and one full learner/FIRST audit. Prepared
[native code/export/client/decoded-reference loader](CHATBOT_COMPACT_NATIVE_NEXT_20261008.md)
and [export/FIRST codec protocol](CHATBOT_COMPACT_EXPORT_PROTOCOL_20261008.md)
remain UNEXECUTED. No compact native chatbot, fresh own-history quality,
accepted50/SAME artifact, useful large n/LUT mass/DRAM or other family/scale
is admitted.64 reserved tasks and16 excluded dialogues remain unqueried.

| Retained file | SHA256 |
|---|---|
| Fit RAW |34b8a7b317c6435b1ead9298ae75f32116ffd4f39606c402992e15bcd0c5387e|
| Fit terminal |63605e0ecce2852bef37ebbf1ed2382db13d71d67b2ea54c69623759f354dbc0|
| Fit log |9d0edb095999b586b589dead0817fdb55914972d91d6e4db3a71af9979b9c474|
| FIRST audit RAW |7609daee3d927d46bdff69b815b35f314f70894cea80e07227c7574e622c1e72|
| FIRST audit terminal |f05cdda4ee59c8e57760881908fb8602fa8917183f9ac6a0ba49df3cc9bdf239|
| FIRST audit log |40b9d6a55aa08bf7c4a2fddfbb6cf05b1658723b9f06eed8d80138330c33a3bd|
| Fit/audit4-instance closure |1f72bea1b75191a107127d72de53d86796c12817a4d7b4f4d290641d3c1fa777|
| Drift RAW |b27c067a151280d938a5583a84004bf9a12216617e65b8acac9c75293bae21e6|
| Drift terminal |3806d0175718416706bd0c87cfc467efbb67c8020abfcfb7e68e2e6956c2d285|
| Drift log |2509cdc375e3e343394beec501af158141658bc0ea1b67cb73a8cce3528e052b|
| Drift2-instance closure |bc1728a522f7653235abecb4918cfa515fb3f93b6fbc8b60fec4769d4a8e4208|

Six actual owned instances closed,no matched application faults. The original
fit/audit closure helper succeeded once; its outer shell erroneously tested
stale LASTEXITCODE and raised afterward. Existing successful closure was read
and validated, helper not replayed. The diagnostic helper tests the immediate
PowerShell success value. No scientific job was repeated for this wrapper fault.
Three protected foreign tracked SHAs remain original; publisher preserved.
