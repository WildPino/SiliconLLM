# Actual balanced Falcon supervision and saved transport adoption

9 October2026. **Original capture COMPLETE; saved-only transport adoption PASS.**
160 NEW calibration cases/5,746 full-vocabulary source rows are available for
whole compact-model recovery. No student update/native run or source-quality
preservation claim.64 RESERVED texts remain unqueried.

## Actual original acquisition

[Cases/builder](../../../benchmarks/native_expert_scaling/chatbot_hybrid_transfer_cases.py),
[cases JSON](../../../benchmarks/native_expert_scaling/chatbot_hybrid_transfer_cases_v1.json),
[collector](../../../benchmarks/native_expert_scaling/chatbot_hybrid_transfer_capture.py),
[protocol](CHATBOT_HYBRID_TRANSFER_CAPTURE_PROTOCOL_20261009.md),
[binding](chatbot_hybrid_transfer_capture_binding_20261009.json),
[result](chatbot_hybrid_transfer_capture_result_20261009.json),
[terminal](chatbot_hybrid_transfer_capture_result_20261009.terminal.json).
Freeze539c6d9d006c089c5f5f36e4b677123651c2a546;
binding SHA55dd3cb1681aac52d228b11b3de8bf0b55684c3877917609a2c0740eef5b6156.

Reuse pinned original Falcon1.5B revision80ebc50d7799a440b96c93bb6686a3924a09b0cb,
untied head/original multipliers/template/BOS/BOTH EOS11/228, BF16 eager
SSM/attention source. All160 canonical template/text/ID/context checks precede
model loading; no source weights/transforms/QAT/stop policy changed.

ONE original greedy cached generation per case, max96 new IDs/context512, full
65,537 raw logits every step.128 FIT/32 DEV, four authored template variants per
eight domains, interleaved case order;64 RESERVED texts distinct/unqueried. All
224 message keys distinct and different from6 pilot/16 screen cases. Split
template families are related; these are limited calibration, not broad coverage.

| Domain | FIT cases/labels | DEV cases/labels | EOS stops | Length stops |
|---|---:|---:|---:|---:|
| Arithmetic |16/1351|4/316|8|12|
| Code |16/829|4/236|13|7|
| Reading |16/114|4/13|20|0|
| Rewrite |16/240|4/71|20|0|
| History |16/59|4/14|20|0|
| Instruction |16/230|4/59|20|0|
| Explanation |16/742|4/195|20|0|
| Planning |16/1078|4/199|16|4|
| TOTAL |128/4643|32/1103|137|23|

137 cases reach EOS;23 responses are explicitly truncated at96. **EOS does not
prove answer truth; truncations are not complete answers.** No unfavorable reply
filtered or regenerated. Capture values are producer observations, not student
recovery. All DEV and screens/pilots are consumed calibration; RESERVED alone
does not constitute a complete independent final quality suite.

Persist5,746 x65,537 BF16 coefficients =376,575,602 coordinates/
753,151,204B. Raw source generation logits are F32 upcasts; producer checks
BF16->F32 roundtrip exact, all values finite and every raw argmax equals its
generated token. EOS labels retained. Each record contains actual prompts/history,
source serialization/IDs/output/text/stop, lossless transport flags and student
prefix/position mapping. Completed source observations saved before admission
assertions; raw data remain immutable at bound paths.

Source family861.782s/worker853.875s, actual exit0, OS peak3,742,052,352B through
exit; GPU allocated5,714,825,216B/reserved6,280,970,240B. All fixed900s/8GiB OS/
10GiB allocated/11GiB reserved/3GiB output/4MiB log caps PASS; no first fault.
322 output files754,963,483B include all packets, case metadata, preflight and
corpus.json. No T4, existing-call replay, student/native/training or reserved query.

## Saved-only adoption

[Adopter](../../../benchmarks/native_expert_scaling/chatbot_hybrid_transfer_adoption.py),
[protocol](CHATBOT_HYBRID_TRANSFER_ADOPTION_PROTOCOL_20261009.md),
[binding](chatbot_hybrid_transfer_adoption_binding_20261009.json),
[result](chatbot_hybrid_transfer_adoption_result_20261009.json),
[terminal](chatbot_hybrid_transfer_adoption_result_20261009.terminal.json).
Freezea591ce2a51a02cc711adf1292e3077c3ac25bf61;
binding SHA029b19da27933f1e1c26a88e3795edf3ef198e803d627741d130ece50b8e1b92.

ALL160 messages/templates/source IDs checked through HF-free Rust Tokenizer
loaded directly from tokenizer.json. This shares the Rust implementation with
sender HF; it is **not independent BPE**. All packet extents/hashes/dtypes/shapes,
376,575,602 finite BF16 bit embeddings,5,746 argmax IDs, EOS/length counts,
student prefix/position mappings and128/32 per-domain counts agree. ALL224 text
uniqueness/screen-pilot exclusion checked. Decision SAVED_CALIBRATION_TRANSPORT_PASS.

Source BF16-lossless flag remains a producer observation: original unretained
source-F32 values cannot be independently reconstructed. Receiver independently
checks saved bit/ID/stop/position transport; no model/truth/capacity inference.
Family8.016s/worker5.281s/OS109,883,392B through exit, actual exit0/all caps PASS,
no Torch/GPU/source/student/native/training call, no fault or data replay.

## Decision and exact continuation

Reusable corpus at
`results/native_expert_scaling/chatbot_hybrid_transfer_capture_20261009/corpus.json`;
packet hashes/coordinates/positions in records, exact commands/PIDs/results/
output hashes in terminal receipts. Three foreign tracked SHA remain unchanged;
all owned processes terminal. Older bindings use their frozen launcher bytes.

**Next:** implement/freeze ONE finite balanced whole recovery/robust-QAT pilot
using these packets and the actual final compact learner under
[balanced recovery NEXT](CHATBOT_HYBRID_BALANCED_RECOVERY_NEXT_20261009.md).
Equal case/domain update weighting matters: source label counts differ sharply
by domain.23 truncated replies provide valid prefix supervision but no full-answer
quality target. Do not replay original source acquisition to make the corpus look
complete. Native old19/32FAIL/poor donor quality stay; NEW trained/native numerical
and fresh own-history quality+accepted50/useful-n/LUT mass/DRAM/family gates remain.
