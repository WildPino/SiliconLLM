# F32 FFN positive control: bounded preservation PASS

9 October2026. COMPLETE/execution/resource andfixed bounded criteria PASS.
This resolves the internal-arithmetic confound ofthe [complete FFN package](CHATBOT_SOURCE_FFN_CONTROL_RESULT_20261009.md)
on two consumed FIT trajectories andthe known16 screen. It does not admit
fresh broad quality,a native forward,rate or larger useful expert capacity.

## Actual intervention and observations

Original pinned Falcon1.5B/all411 parameter objects/1,554,863,488 coefficient
values retained. Worker-local FFN override executes original coefficient
VALUES inF32 linears/nonlinear/product/original MUP multipliers andcasts only
thefinalFFN output toBF16. Non-FFN organs/source cached schedule/qualified SSD
helper/BOS17/EOS11,228 retained. Zero ternary or activation quantizations.
Real-arithmetic function unchanged;finite arithmetic is a new variable.

| Original adopted FIT trajectory | Labels | F32-only KL_F64 | Ternary/AQ63 KL_F64 | F32 differing IDs |
|---|---:|---:|---:|---:|
| explore-instruct-rewriting008 |18|.000162426636|1.745568270|0|
| smol-magpie-ultra022 |256|.000471965223|2.146752359|5|

New274 original forced rows,allchanged BF16logits/per-label F64 KL retained.
No original source reply regenerated.16 new counterfactual generations,
40 emitted IDs,allEOS. Exact requested-answer score14/16,original14/16,
full ternary/AQ63 package1/16. Categories arithmetic3/4,extraction4/4,
instruction4/4,history3/4. Division stillanswers2 ratherthan4;favorite color
`Green` fails the expected lowercase`green`. These are also original failures.

Saved-only [generation comparison](chatbot_source_ffn_cast_saved_generation_comparison_20261009.json)
verifies every original case JSON against its historical terminal hash and
compares actual canonical inputs/serialized text/outputIDs/text:ALL16 inputs
andALL16 generated sequences/text EXACTLY EQUAL toretained originalBF16
source generations. ComparisonSHAe00326683e9a03090a6184566a6904cdef913f15931d3640932e02e12a2901ef.
No score-bit equivalence is claimed. New F32 scores finite/argmax consistent,
no blank/special leak/invalid stop. Four own-history followups were not repeated.

Allseven preregistered gatesPASS:bothFITKL<=.01,5/274=1.8248% disagreement
<=5%,screen>=13/everycategory>=2,blank/leak/stop. Decision SOURCE_FFN_F32_CAST_PASS.
These are bounded consumed-FIT/known-screen preservation controls,not broad
heldout evidence. Result cannot split weighttrits fromAQ63 or subtract KLs as
independent loss contributions. Inthis scope,F32 alone does not explain the
complete package's failure;the next intervention can target ternary/AQ63
function error while keeping theother source organs andengine kernels fixed.

## Resources and reproduction

[Protocol](CHATBOT_SOURCE_FFN_CAST_CONTROL_PROTOCOL_20261009.md),
[binding](chatbot_source_ffn_cast_binding_20261009.json),
[result](chatbot_source_ffn_cast_result_20261009.json),
[terminal command/alloutput hashes](chatbot_source_ffn_cast_result_20261009.terminal.json),
[worker log](chatbot_source_ffn_cast_result_20261009.worker.log).
Tool `benchmarks/native_expert_scaling/chatbot_source_ffn_cast_control.py`.
Freeze d94f854c60614d34f9ccfef4a716a3ef91aff346;46-input binding
002510630511bc5815db4cb99b47eb9c3488e59db26b489951147d8f907fe8a7;
resultSHA7a108d0e9f3498a29afd905c1a2cb2e9644e32eddd1913c2291007dbbb8dbfb5.

Worker67.234s/family75.829s<420s cap/30s reserve. GPUallocated5,580,204,032B/
reserved6,905,921,536B,heldOS3,740,504,064B;37 outputs46,442,771B. All resource/
input gatesPASS,no firstfault/overlap,exit0/session30528closed;launcher6664/
worker23696.0 optimizer/native/T4/RESERVED calls. No physical VRAM or full
DLL-tree certificate. Original source-active cost remains incompatible with
thefinal compact engine target;original LUT/ternary/SSM goal stays incomplete.
