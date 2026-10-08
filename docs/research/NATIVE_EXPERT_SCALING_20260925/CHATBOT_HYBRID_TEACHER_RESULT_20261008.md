# Useful pretrained hybrid teacher and compact engine-target budget

8 October2026. Goal ACTIVE/INCOMPLETE. This goal turn is PROGRESS: a larger
actual pretrained chatbot is acquired/qualified for the first adaptation pilot,
and a source-compatible compact recurrent/ternary target is costed. No training
or whole native model has run.

## Actual donor, not model-card quality

tiiuae/Falcon-H1-1.5B-Instruct, pinned
`80ebc50d7799a440b96c93bb6686a3924a09b0cb`. New reusable
[metadata/source tool](../../../benchmarks/native_expert_scaling/chatbot_hybrid_source.py)
and [protocol](CHATBOT_HYBRID_TEACHER_PROTOCOL_20261008.md), freeze8ae1e1a.
Metadata4.157s; weights137.906s, all original source files, contiguous complete
safetensor header validated BEFORE payload and producer whole-file SHA verified:
`1fb788513f2c58e3abb91af8730afe19e247bce7a7f01a46e452a084e98b2abd`.
Actual BF16 file3109773032B,411 stored tensors/1554863488 named elements.
[Pinned metadata receipt](chatbot_hybrid_source_metadata_20261008.json) and
[complete package receipt](chatbot_hybrid_source_package_20261008.json).

Source D2048/L24/V65537, dense SwiGLU4608, parallel attention+SSM in ALL24 blocks.
Attention width1024 (8*128),KV256 (2*128), not hidden2048. SSM width3072,
48heads*64/state256/group1, gate-THEN-RMS. Nontrivial input/output/SSM/MLP/key
multipliers retained. Config/operator matrix products1.420036096B/token,
not all scalar/state/context work:FFN679477248,SSM projection480509952,
attention125829120,head134219776. Full-source F32 dense Adam arrays from actual
header count24877815808B, beyond12GiB GPU/16GiB T4 before activations.
The matrix-only metadata lower bound22.7206GB excludes embedding and scalars.

Actual original interaction: source BOS `<|begin_of_text|>`, then plain role
messages; no Tiny leading/assistant-content newline. Untied readout/embedding,
embedding multiplier5.656854249492381/head.01953125, bothEOS[11,228]. Complete
16 text/ID fixtures pass before loading. Local HF5 legacy RoPE is explicitly
checked at rope_parameters/default/theta100000000000. Loaded parameter count
matches411-tensor header. Offline local supported BF16/eager-attention/Torch
fallback; optional remote/SSM kernels absent. Selected runtime files/version/
paths bound, full native DLL trees not rehashed. No remote Python executed.

## Actual unchanged teacher usefulness gate: PASS

Same16 fixed English cases and strict normalization/criteria as Tiny;>=12/16
AND>=2/4 each category,blank<=1,no special-token leak and original stop policy.

| Category | Larger donor | Prior Tiny |
|---|---:|---:|
| arithmetic |3/4|0/4|
| extraction |4/4|1/4|
| instruction |4/4|2/4|
| supplied history |3/4|2/4|
| total |14/16|5/16|

Every absolute/stop/leak gate PASS. Two errors retained:20/5 answered2 instead
of4, and `Green` rather than required`green`. No scoring relaxation or substring
credit. This justifies selecting the larger donor for a pilot; it does not
prove broad conversational quality, own-history behavior or knowledge retention.
These16 donor-selection cases must never become the training data or final gate.

[Actual result](chatbot_hybrid_usability_result_repair1_20261008.json) and
[held-handle terminal receipt](chatbot_hybrid_usability_result_repair1_20261008.terminal.json),
freeze e689f9fe,binding
`2ed06f655359836d074af9c317bce66df6368dff682000444f7be4b6d2131b28`:
exit0,20.360s worker/26.922s family. OS3740688384B through worker exit;
GPU allocated4452947456B/reserved4643094528B. Complete outputs/IDs/all generation
score vectors10.52MB saved. Actual conv cache[1,3584,4]BF16 and recurrent
cache[1,48,64,256]BF16 in every layer. No source-screen generation replay.

Retained first faults: caller supplied wrong metadata SHA before file fetch;
first larger screen checked old rope_theta attribute and stopped before any
source forward. Repair only changed caller hash and wrapper attribute path.
Original source bytes/arithmetic/prompts/criteria unchanged. No source acquisition
restart or completed case replay. No active job/T4; foreign tracked SHA preserved.

## New prospective whole target, original-engine thesis preserved

[Budget tool](../../../benchmarks/native_expert_scaling/chatbot_hybrid_target_budget.py),
[fixed protocol](CHATBOT_HYBRID_TARGET_BUDGET_PROTOCOL_20261008.md), freezeb665d004,
[actual integer result](chatbot_hybrid_target_budget_20261008.json). Four new
source-compatible target budgets, no new weight/model/old triage observation.

First budget candidate:D512/L12/SSM10/SWA2(window128),SSM768/12heads*64/state256,
gate-then-RMS,V65537,72 experts per layer/top8/h128. F32 sensitive core/organs,
byte-pair ternary expert coefficients/scales,int8 activation,packed-only bank.

| Counted quantity | New target candidate |
|---|---:|
| core matrix products/token |16576512|
| full readout products/token |33554944|
| selected expert products/token |18874368|
| proposed tree products/token |245760|
| total matrix products/token |69251584|
| packed-only model coefficient bytes |423663008|
| selected code+scales bytes/token |9732096|
| F32 recurrent/SWA cache bytes |9117696|
| logical coefficients+state bytes/token |229821856|
| all target F32 Adam array bytes |4072822400|
| Adam+teacher file lower envelope |7182595432|
| extra old F32/int8 bank copies |849346560|

**Current first pilot variant:** [preserve source decay heads](CHATBOT_HYBRID_TARGET_HEAD_VARIANT_20261008.md)
and [new integer record](chatbot_hybrid_target_head_variant_20261008.json),
freezee80f28a: SSMD768=48heads*16, keeping all48 A/delta/D head indices within
each retained source block. Other24->12/SSM10 transformations still omit source
blocks, and projected inputs change actual delta; no full-model decay equivalence.
Matrix total69435904,packed424404608B,Adam4075788800B,teacher+Adam7185561832B,
logical decode230563456B. State cache/selected code bytes/FFN slots unchanged.
This avoids discarding3/4 of source head spectra for only184320 extra products.
Naive SSD training workspace grows4 times at fixed chunk length; actual smaller
chunk/checkpointed pilot memory is mandatory. First budget remains historical.

These are exact integer counts for the DECLARED design, not actual allocations,
DRAM/rates/quality. The69.25M first/69.44M current excludes scalar recurrence/conv/normalization/
nonlinearities/context attention/tokenizer/glue.7.18GB training envelope excludes
activations/teacher computed buffers/workspace/optimizer copies/runtime. Actual
teacher peak4.45GB exceeds its3.11GB coefficient file, illustrating why those
exclusions matter. GPU resident feasibility requires a real pilot, with optional
saved teacher supervision or activation checkpointing if needed.

Feature slots72*128*12=110592 equals source24*4608. This is a slot COUNT only;
two source blocks cannot be merged merely by pooling rows. Source hidden/state
width reduction, depth/attention-to-SWA and conditional/ternary approximation all
need learned whole-output recovery. No source knowledge conservation is inferred.

Chosen prospective expert scalar is ternary SwiGLU with signed up, retaining
the original packed ternary LUT matrix machinery. This is a small justified
activation variant: source SiLU/signed contributions are not zero under ReLU.
Original gated-dReLU remains an alternative learned target, not an exact source
conversion. Activation quantization/scales/rounding must be in the real learner
and export. CPU LUT/router/scalar variant has not run yet.

Source per-head A/delta are constant over head channels/states, so exp(delta*A)
can be hoisted algebraically instead of repeated per state in the old scan.
Actual C arithmetic/timing and the new gated RMS/projection/conv remain to be
qualified; Tiny12-packet pass does not validate this larger variant.
V65537 requires32-bit native token IDs; do not silently truncate to old uint16.

## Exact next action: whole pilot, not another generic donor runtime

Implement ONE full target forward/backward/update with source-informed recurrent/
expert initialization, explicit width/composition approximation and actual ternary/
int8 deployment arithmetic in the learner. First measure memory/finite gradients
and output recovery on broad source supervision; do not restart exact-local1%
screening of isolated layers as a substitute. Original source outputs/metadata/
weights and16 selection results are complete and reusable.

Before fitting, freeze calibration data separately from excluded own-history/
task criteria. Neither the16 selection cases nor old Qwen excluded endpoints are
this donor's training corpus. Whole KL/behavior governs recovery; an internal
changed representation need not reproduce each source layer to1%. Price/log
actual pilot cost and stops; only then decide sustained local/T4 adaptation.
Long T4 remains in scope after actual communicated reason/budget/stops and FP16
compatibility. No T4 job is launched or needed for the completed screen/budget.

Then packed-only whole export into original scan/SWA/LUT engine, canonical chat/
32-bit IDs/EOS, fresh own-state dialogue/tasks AND>=50 accepted batch1 IDs/s SAME
artifact. Structured CPU LUT winners+normalized masses, physical DRAM and useful
large n/further families/~10B/~100B remain part of the original full goal.
