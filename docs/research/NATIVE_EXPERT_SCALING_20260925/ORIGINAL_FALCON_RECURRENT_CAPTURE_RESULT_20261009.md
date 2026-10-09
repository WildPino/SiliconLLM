# Source recurrent operands: all48 adopted, interrupted parent limits retained

9 October2026. Goal ACTIVE/INCOMPLETE. Useful pinned Falcon-H1-1.5B-Instruct,
source revision80ebc50d7799a440b96c93bb6686a3924a09b0cb/weights1fb78851.
No training/native/RESERVED/T4 or new source reply generation. Original engine
and actual source-informed Adam25 unchanged. This supplies NEW internal history
information for a principled core map; it does not qualify a compact chatbot.

## Available data and verified continuity

ALL48 existing forced histories/24FIT+24DEV/12domains,22,547 full input IDs/max1507,
ALL24 source SSM blocks/seven fields at EVERY position. Actual conv/SiLU x3072,
B256/C256,gate3072,delta48 after source softplus/clamp,BF16; pre-gate scan+Dskip
y3072/F32; actual source mamba out_proj response2048/BF16. Source MuP,conv4,
chunk128/qualified SSD storage schedule and eager parallel attention unchanged.
One full base-model prefill/zero state/use_cacheFalse per completed history;
no LM-head/answer/teacher-output calls. Full-prefill arithmetic is not claimed
bit-identical to earlier incremental cached teacher logits.

[Canonical stored adoption](original_falcon_recurrent_adopted_result_20261009.json),
SHA `bdec178a948e49bbec1d9eb4c01076bec9c182be8347306eb5dc02e0c873c849`:
all field IDs/positions/shapes/dtypes/extents/hashes and finite coordinates checked.
Logical payload16,121,285,376B. FIT24/11,148 IDs and DEV24/11,399 IDs. Basis/selection
must use FIT only; DEV acquisition is not fitting. Independent projection analysis
uses stored data, with no additional source forwards.

## First family disappeared; completed45 were preserved

Original session29367 observation call was aborted and the handle later returned
Unknown process id. Independent Win32 inspection found no source Python process.
Original namespace has45 durable admitted cases and63 atomic unadmitted fields
for broad_dev_smol_summarize_039/sites0..8. Last completed event1267.25s;
20,380 completed input IDs/14,571,863,040B full-case payload. No original result,
terminal or first_fault.json exists. Cause, exit code, original held-family total,
GPU/OS peaks and final411 parameter identity/version check are UNKNOWN/MISSING.
Do not infer successful exit, resource PASS or reconstruct a final runtime counter.

Original [log](original_falcon_recurrent_capture_result_20261009.worker.log),
[binding](original_falcon_recurrent_capture_binding_20261009.json) SHA
`aa1c57ccf7925f2f60e4a81104ce5ef96929d2912de29377395bd7d8668e6dc2`,
[protocol](ORIGINAL_FALCON_RECURRENT_CAPTURE_PROTOCOL_20261009.md), freeze
`0092994ba329f878e53c03eaf27d7f0dddc1524c`, exact
[original worker bytes](original_falcon_recurrent_capture_frozen_20261009.py.txt)
SHA `c399c5144e11f64ea7b5789f49920a36f902681499efd47a0f68e66b1b0ae2b8` retained.
Metadata-only repair makes event/launcher case count binding-driven, not fixed48.
No source operator changes. Former mutable NEXT input bytes are preserved in
[NEXT byte archive](original_engine_transfer_next_frozen_20261009.md.txt), SHA
`1f85ec259ab23ee08305da7cf28d4ad4dabd1e11deefd31d815669b365c41882`.
Historical pre/post checks occurred before operational NEXT documentation updates.

## Exactly three missing histories completed

New separately frozen source family captures only the unadmitted three DEV cases:
smol_summarize_039/980 IDs,systemchats_30k_034/890,systemchats_30k_039/297.
2,167 new complete IDs/1,549,422,336B payload. No45 completed history replay.
The interrupted case requires recomputing its prefix to reach later layers;
all63 old partial fields become byte witnesses. Stored adoption proves exact
byte/hash equality with the new prefix, retaining both original and new files.
This is an explicit repeated partial attempt. Total48 completed base forwards,
at least49 attempts including the original partial attempt; no source generations.

[Resume protocol](ORIGINAL_FALCON_RECURRENT_RESUME_PROTOCOL_20261009.md),
[7,702-input binding](original_falcon_recurrent_resume_binding_20261009.json) SHA
`926db4a7718e476a5552ddd78997ec85cffbc23e369ffa05b542ef3320303b48`, freeze
`096971012d814ff004faeec7c41095b7cc3e5a27`.
[Raw result](original_falcon_recurrent_resume_result_20261009.json) SHA
`eae91e1414bb268d4cdc578b5eb332d346c59983528751fbc3bd2a2738ba67de`,
[held terminal](original_falcon_recurrent_resume_result_20261009.terminal.json),
[log](original_falcon_recurrent_resume_result_20261009.worker.log).
All411 identities/versions unchanged in THIS new source instance; the corresponding
aggregate check for the vanished original instance remains missing.

Source completion launcher11444/worker24724/creation1791557095.1317732,exit0,
session91668 CLOSED. Actual held family301.75s; worker255.5s.
Held worker OS3,735,830,528B + launcher43,270,144B=3,779,100,672B conservative sum.
GPU allocated5,054,807,040B/reserved5,932,843,008B, allocator only. All NEW family
600s/reserve60/OS12GiB/GPU10/11GiB/output3GiB gates PASS. These values describe
the three-case completion only. Combined family time cannot be given exactly;
1267.25 original worker +301.75 new held family gives only a known lower bound.
Binding and later stored-adoption preparation are additional unheld costs.

Old namespace7,669 files/14,838,227,391B; completion508 files/1,549,680,157B;
total8,177 files/16,387,907,548B retained. This includes262,765,440B duplicated
partial-prefix witnesses and small receipts/helper files. Data remain off-repo
in results/native_expert_scaling/original_falcon_recurrent_capture_20261009 and
original_falcon_recurrent_resume_20261009. No original files overwritten/deleted.

Stored-only [adoption code](../../../benchmarks/native_expert_scaling/original_falcon_recurrent_resume.py)
pre/post input checks, completed output hashes, all finite coordinates and63
partial witnesses; session92800 exit0/CLOSED. No source/GPU/optimizer/predictions.
No separate held-family time/OS record for this CPU-only adoption was collected.

## Reproduction and next decision

Python312 `-I -S -B -X utf8`, repository root; completed namespaces are immutable.
Original command was original_falcon_recurrent_capture.py --launch, initial
bindingSHAaa1c57cc/freeze0092994, original namespace and original result path.
It did not complete; never rerun that namespace as if it were a fresh experiment.

```text
benchmarks/native_expert_scaling/original_falcon_recurrent_resume.py --bind --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_resume_binding_20261009.json
benchmarks/native_expert_scaling/original_falcon_recurrent_capture.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_resume_binding_20261009.json --binding-sha 926db4a7718e476a5552ddd78997ec85cffbc23e369ffa05b542ef3320303b48 --freeze 096971012d814ff004faeec7c41095b7cc3e5a27 --directory results/native_expert_scaling/original_falcon_recurrent_resume_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_resume_result_20261009.json
benchmarks/native_expert_scaling/original_falcon_recurrent_resume.py --adopt --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_adopted_result_20261009.json
```

[Completed stored projection](ORIGINAL_FALCON_RECURRENT_PROJECTION_RESULT_20261009.md)
fits bases on24 FIT histories/all24 sites and evaluates complete recurrent/gated/
normalized outputs on sites0/12/23 and all48 histories. All144 baseline errors0;
three independent scalar F64 witnesses PASS. Coordinate96/orthogonal96/dual96
distinguish state loss and nonlinear-generator representation cost. DEV mean
dual output errors7.6349%/8.7184%/2.4918% versus coordinate20.1061%/25.7026%/4.4961%.
Centered/recurrent-only errors and independent F64 identity/aggregate audit retained.
No source replay; original reserved-GPU fault retained, missing126 comparisons
completed under the same GPU caps with a bounded chunk-storage schedule.

These are source-local diagnostics retaining all3072 x/gate channels/48 heads.
They do not qualify DN512/1024/P256/24->6 composition or whole chatbot quality.
Original numerical FAIL and whole-recovery five quality-gate failures remain.
Useful same-artifact chatbot+accepted50, useful RAM-driven n/CPU IDs AND mass/
physical DRAM and additional donor families/scales are still required.
