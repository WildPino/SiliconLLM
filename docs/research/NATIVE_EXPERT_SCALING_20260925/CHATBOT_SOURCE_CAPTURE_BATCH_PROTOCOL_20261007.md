# Source calibration acquisition after original300s resource failure

Original qualified-runtime source freeze588e125, actual executor920951/session8530,
final360f7f exit1. Sixteen full requested conversations and14 complete forward
frames of code_debug_1 are retained. Worker708/create1791383690.2328684,
launcher31380/create1791383678.8718083. Worker guard fails at300.094s, actualexit1,
OS peak1680101376B through exit; GPU allocated1020826112/reserved1071644672B.
Whole launcher312.297s,parent last32096256B. Original ALL48/300s capture remains
FAILED; never retroactively change that result or call partial generation complete.

## Changed acquisition variable, before more source responses

Keep ALL original48 case messages/roles/fit32/development16, pure Qwen BF16 eager
source, both EOS/max96, x/y format and source correctness/OS/GPU/size gates.
Adopt previously saved COMPLETE FORWARD operands, including the failed
conversation's complete written prefix, as CALIBRATION source information.
A full requested96/EOS generation is not necessary for a valid original
function response at a particular x. This is a new explicit calibration-data
eligibility rule, NOT a pass of original full-generation/resource criteria or
fresh whole model quality. No fitting/response-error criteria have been observed.

Byte-only [adopter](../../../benchmarks/native_expert_scaling/chatbot_capture_adopt.py)
requires original pure-source/Arrow-absent/input gates, all saved frame SHA,
header/size/C-order/dtype/offset, exact journal equality and input/next-ID alignment.
It preserves original source receipt SHA, all original case partitions and the
partial flag/termination. Does not compute source/model/cell responses again.
Full-generation status remains false for code_debug_1; its KV cache was not
saved, so DO NOT resume/replay that conversation. Qualify existing complete
written forwards only; no invented missing frames or cached-state reconstruction.

Each new worker selects at mostEIGHT previously UNENTERED cases in frozen
manifest order, skipping every adopted case including that partial one.
Fixed worker300s/family1200s/resource gates remain. Four planned batches8/8/8/7
cover the31 missing cases. Original labels/generation before the timeout are
not repeated. Every newly collected requested generation must finish its own
EOS/max96 within its batch; a further fault retains prefix and forbids replay.

Change durable I/O: append+flush each forward and journal; fsync every16 forwards
AND at conversation completion. First producer fsynced both files every forward.
This is a motivated potential overhead reduction, not proven causation from
old aggregate clocks. Correctness is unchanged after byte verification; hard
power-loss durability granularity is now16 forwards, explicit. Completed-file
receipts/final launcher SHA are still mandatory. No intrinsic GPU/DRAM speed
claim. Original acquisition costs including failed startups/time remain charged.

## New data eligibility and next fit

After all four batches, require ALL48 original conversation CASES represented
by at least one byte-qualified original forward prefix, exact32-fit/16-development
partition, unchanged original messages/source/format, no source replay and all
new batch gates. Report original16 complete +one partial +31 new complete;
never claim ALL48 full generations completed or original300s resources passed.
This dataset is CALIBRATION/RESERVED DEVELOPMENT ONLY, not final fresh quality.
Finite region exposure/initialization/joint fit/response/count-utility/stop
criteria still must freeze before training. The compact converter and SAME-
artifact fresh chatbot quality/accepted50 remain missing.
