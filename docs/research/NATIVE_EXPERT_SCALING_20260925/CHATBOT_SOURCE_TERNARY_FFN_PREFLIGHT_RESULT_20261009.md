# Source-width ternary FFN conversion: cost and arithmetic preflight

9 October2026. COMPLETE/resource and integer-arithmetic PASS;quality admission
FALSE. This is the first measured stage of the [staged conversion](CHATBOT_STAGED_CONVERSION_NEXT_20261009.md),
an offline control retaining the original donor organs. It is not a new final
source-shaped runtime or an accepted engine artifact.

## Intervention and artifact

Pinned Falcon-H1-1.5B-Instruct revision80ebc50d7799a440b96c93bb6686a3924a09b0cb:
original D2048/L24/FFN intermediate4608,all FFN rows retained. Replace only the
FFN arithmetic package:row-calibrated ternary weights,AQ63 activations,integer
dots,F32 row/activation scales and nonlinear operations,original MUP multipliers
in their original positions,final BF16 cast. No router,core,width,depth or head
transformation. This is an approximation;it does not separately attribute loss
to weight ternarization,AQ63 and the changed internal cast boundaries.

All679,477,248 FFN coefficients converted in7.390s. Actual pair/row packed
payload340,819,968B includes339,738,624B pair codes and1,081,344B F32 scales.
The encoding is the existing export byte(first_trit+1)*3+(second_trit+1).
All339 other parameter objects/875,386,240 coefficients retain original identity.
Sector `results/native_expert_scaling/chatbot_source_ffn_preflight_20261009/ffn.packed.bin`
SHA9aa6f319363294aac6ee611def4384e9c63354929c43c6219d8144f698a2a33b.
This sector alone is not a complete native model. All source-width FFN rows
remain active;their cost is not the compact engine target's selected work.

## Observations

Metadata-only shortest andlongest adopted FIT cases,original forced IDs and
cached source schedule. No source reply regeneration,training or native run.

| Case | Prompt/labels | KL_F64 | Differing greedy IDs | Trajectory seconds |
|---|---:|---:|---:|---:|
| explore-instruct-rewriting008 |41/18|1.745568270|5/18|4.109|
| smol-magpie-ultra022 |1252/256|2.146752359|141/256|41.797|

All274 changed full-vocabulary BF16 rows andper-label KL retained. Six actual
block0,last-prompt gate/up/down integer-dot witnesses compare EVERY output
coordinate to an independent CPU F64 integer oracle:0 differing coordinates.
The arithmetic bound is63*4608=290304<2^24. This validates those integer dots,
not the full nonlinear FFN or a whole C forward. Two FIT cases are not broad
quality evidence;the [complete control protocol](CHATBOT_SOURCE_FFN_CONTROL_PROTOCOL_20261009.md)
was fixed before its additional46 trajectories/20 generations.

GPU peak allocated4,902,440,960B/reserved7,155,482,624B. Worker held OS peak
through exit3,724,275,712B;worker snapshot70.250s/family77.781s.19 outputs
433,599,964B,exit0,session25639 closed. All bound input/resource checks PASS.
No overlapping timed job or fault. CUDA counters do not certify physical
VRAM residency. Runtime binding covers selected files,not the complete DLL tree.

## Reproduction and decision

[Protocol](CHATBOT_SOURCE_TERNARY_FFN_PREFLIGHT_PROTOCOL_20261009.md),
[binding](chatbot_source_ffn_preflight_binding_20261009.json),
[result](chatbot_source_ffn_preflight_result_20261009.json),
[terminal command and output hashes](chatbot_source_ffn_preflight_result_20261009.terminal.json),
[worker log](chatbot_source_ffn_preflight_result_20261009.worker.log).
Freeze eb6075802cdfc34695739b460dd701eb8ec01cb4;33-input binding
1451533dc5e7df26eded6537db7dd66176edcdf03459c05f06cc981da24f5544;
result cf8dbbeec404cfbe1233fcccd54f9b5f575ed2154f1959df1d56b074b425238f.
Tools: `chatbot_source_ternary_ffn.py` and `chatbot_source_ffn_preflight.py`.

Decision SOURCE_FFN_COST_ARITHMETIC_PASS:actual conversion/control fits the
fixed local inference envelope. Extend to all48 adopted prefixes and16 known
screen prompts plus4 actual generated-history followups,reusing these two
trajectories andactual sector WITHOUT recalibration. Measure conversion loss
before adding selection/core compression or pricing recovery/T4. Final engine
quality+50,useful large n,CPU LUT mass/IDs,DRAM andfamilies remain open.
