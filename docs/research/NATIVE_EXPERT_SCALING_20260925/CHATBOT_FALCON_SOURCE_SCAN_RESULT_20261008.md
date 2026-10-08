# Actual pretrained source and original-engine scan bridge

8 October2026. Goal ACTIVE/INCOMPLETE. This turn made source acquisition,
teacher screening and a real original-scan operator bridge executable.

## Decisions

- Original engine scan: PASS, all12 real-state/output comparisons below fixed1%.
  Keep the original exact-exp F32 scan as an available recurrent operator.
- Tiny90M teacher: FAIL5/16 exact instruction/task screen. Do not select it for
  the first whole chatbot adaptation merely because its scan maps conveniently.
- Next donor candidate: Falcon-H1-1.5B-Instruct. Inspect pinned actual config,
  interaction/header, active cost and local teacher memory before acquiring/
  qualifying its source. Its producer documents parallel SSM+attention and
  instruction evaluations; these are producer evidence, not our quality result.
  [Primary model card](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct).
- No whole adaptation, ternary expert conversion, accepted50 or T4 work ran.

## Acquired source and actual interaction

Model tiiuae/Falcon-H1-Tiny-90M-Instruct, revision
`e6389502a0b12cd8da894b395ba5bf7436873b16`. Acquisition freeze dd0fd617,
one execution23.093s, all seven pinned original files. Weight file182304120B;
SHA `199e9960cd9be797579bd7dfc2b19f72e098e9c4cd3720afd50c815b48080578`,
producer LFS whole-file SHA verified. Header41976B and all386 stored objects
validate contiguous shape/dtype/extent coverage,91131072 BF16 stored elements.
This differs from90996736 matrix products because norms/bias/state parameters
also have stored coefficients. Weight/tensor metadata receipt preserved at
`results/native_expert_scaling/falcon_source_e6389502_20261008` and the copied
[package receipt](chatbot_falcon_source_package_20261008.json).

Source loaded through local Transformers5.13.1/Torch2.6.0+cu124, BF16/eager
attention, optional SSM/hub kernels absent, original naive Torch SSD/step path.
Actual embedding/head tie and multipliers preserved (.11083984375/.078125).
Both producer EOS228/11 retained. Plain chat/history source serialization has
an INITIAL newline and an assistant-history newline; canonical text and exact
HF text/ID parity pass all16 source cases. Tokenization/decode uses original HF
source, not an independently implemented BPE or a C tokenizer. Cache actual:
each layer conv[1,896,4] BF16, recurrent[1,24,32,64] BF16.

Runtime binding covers original files, Python executable, selected HF source
files and exact version/import-path checks in the isolated local view. Full
native DLL trees were not rehashed and numerical runtime is not independently
certified. All source calls ran offline without remote code/downloaded kernels.

## Fixed teacher usefulness screen

[Protocol](CHATBOT_FALCON_SOURCE_PROTOCOL_20261008.md), fixed16 English cases,
greedy64 maximum/context<=256, no training. Exact response after allowed
whitespace/one fullstop/paired quote normalization; no extracting a correct
substring from an explanation. Required>=12/16 AND>=2/4 each category.

| Category | Correct under fixed instruction contract |
|---|---:|
| arithmetic |0/4|
| extraction |1/4|
| instruction |2/4|
| supplied history |2/4|
| total |5/16|

Blank/special leakage/stop-policy gates pass. The arithmetic answers contain
the right results but violate number-only instructions. Other substantive
errors include extracting `cat` instead of `Clara` and an unfinished verbose
attempt to reverse `abc`. Literal/capitalization mistakes also count under the
fixed criteria. This rejects the tiny teacher for our proposed pilot, not the
whole Falcon family or all uses of90M. Supplied history is not model-own-history.

Successful source freeze58f897b, binding
`b35956f7ede8aa210dcf0572207aee8f70f6730b88a31438be12f888b1f03501`.
[Result](chatbot_falcon_usability_result_repair4_20261008.json) and
[actual terminal](chatbot_falcon_usability_result_repair4_20261008.terminal.json):
exit0,42.063s worker/44.0s launcher family, OS1501491200B through worker exit,
GPU allocated379982848B/reserved402653184B. Saved complete IDs/text and EVERY
step's F32 source generation score vector (not reconstructed raw training
logits),31.88MB total output. No completed case replay.

Four wrapper first faults are preserved: checking constructor-created fast-path
flag too early; JSON infinite config bound; independent serializer omitting
source leading newline; HF5 ID-container default. All stopped before the first
source forward. Repairs changed wrappers/receipts only, preserving source inputs,
arithmetic, cases and criteria. Original fault directories/logs/receipts remain.

## Actual original-scan bridge

[Frozen protocol](CHATBOT_FALCON_SCAN_PROTOCOL_20261008.md), freeze2547c449,
binding `f4b68b8d63ec84be5e751c4458316e7d1a874dc0fdb465df0fac04911754f44c`.
One NEW development prompt,24-ID prefill followed by3 source-greedy cached
steps. Hooks at layers0/12/23 capture real projected/convolved X/B/C/gate,
source before/after states and actual gated out_proj input. Delta recomputed
with identical source BF16 softplus/clamp on captured projection; source A/D.
Exactly4 full source forwards and12 packets, no repeated teacher cases.

Map c=head*32+channel,DN768/N64. A/delta/D per head replicated over channels,
group1 B/C shared. Original phase60 exact-exp update/reduction and SiLU gate
text verified before comparisons, unchanged apart from shape scaffolding.
Standalone [C wrapper](../../../benchmarks/native_expert_scaling/chatbot_falcon_scan.c)
does not alter engine.c. Compiler -O2/-fno-fast-math/-ffp-contract=off, one thread.
Oracle bytes are separate and never supplied to native scan.

| Observable, over EACH12 packets | Worst relative RMS | Fixed limit | Result |
|---|---:|---:|---|
| complete evolved state |0.275675%|1%|PASS|
| all gated scan outputs |0.406017%|1%|PASS|

This is **actual numerical evidence** that the pretrained per-head recurrence
executes through the original F32 scan algebra on these real operands, including
reset/prefill and cached source steps. Source BF16 casts, SSD prefill versus
sequential reduction, F32 exponential/reduction explain possible differences;
the experiment does not attribute the residual to any one cause.

[Capture](chatbot_falcon_scan_capture_20261008.json)/
[terminal](chatbot_falcon_scan_capture_20261008.terminal.json): exit0,
12.375s worker/13.860s family, OS1463025664B through exit,
GPU allocated379546112B,8.16MB output.
[Native comparison](chatbot_falcon_scan_comparison_20261008.json):
compile exit0 and all12 native exit0,4.125s compile/compare process; input/output,
compiler/executable/engine hashes and per-packet energy/errors/commands saved.
Native OS through-exit peak was not measured, no speed/resource admission.

Every packet starts from its own REAL source initial state. Prefill is zero-reset;
later calls restart the native scan from saved BF16 cache. Therefore this does
not verify accumulated C-state drift, full source projection/conv arithmetic,
whole logits/chat behavior or autonomous native continuation. Three source
layers/one short development prefix are not a uniform error bound or a proof
for other family variants; gated RMS in larger variants needs separate support.

## Procedure now available and exact continuation

Available stages: pinned source acquisition/header/SHA -> offline original
interaction/usefulness -> real source scan packets -> original scan C evaluation.
Reuse completed source/results rather than rerun. Four source-screen faults,
successful screen/capture and comparison jobs are terminal; no jobs/T4 active,
three foreign tracked SHA unchanged.

Next: establish a useful stronger hybrid teacher's actual source contract and
whole active/training budget, then choose the smallest costed target. Do NOT
carry Tiny's D512/V32768 or no-gated-RMS assumption into a larger donor silently.
Keep readout/tokenizer/SSM source initialization where algebra permits. Required
changes remain explicit learned approximations: fewer blocks/attention sites,
bounded SWA, compact dimension where chosen, source FFNs -> selective ternary
LUT experts, combined whole output/dialogue recovery. Source-compatible
projection/conv preparation is still missing in the engine.

Then one finite pilot of complete output recovery, fresh excluded own-history/
task criteria, actual packed-only export/runtime and SAME-artifact>=50 accepted
IDs/s. Useful large n requires structured CPU LUT winner+normalized mass and
actual DRAM; fixed k does not bound original full-E routing/loader copies.
Long T4 adaptation remains authorized in principle; communicate actual reason,
budget/stops and measured memory/FP16 feasibility before allocation.
