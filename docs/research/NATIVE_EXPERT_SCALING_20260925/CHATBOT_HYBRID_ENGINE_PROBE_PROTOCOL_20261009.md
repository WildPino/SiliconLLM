# Original-engine entry, incremental chatbot apparatus and complete decode cost

9 October 2026. Preregistered before execution. The human asks to retain the
original LUT/ternary/SSM destination and permit lengthy offline conversion.
Prioritize a measured deployment envelope before another training variant.
Common/private implementation b936b60 is frozen but UNEXECUTED and deferred.

## Decision and reused evidence

Reuse the actual 425,210,736-byte 8-update packed artifact, not the unexported
checkpoint286 or common/private model. Old source quality and 19/32 numerical
RMS failures remain. Question: can the actual compact target execute through
phase60/engine.c, preserve incremental state and charge full chatbot readout?
The result selects the deployment/cost work needed before longer adaptation.
It cannot admit quality or accepted50 from this unqualified artifact.

Add a compile-time SILICON_FALCON_TERNARY_CHAT branch to engine.c. Default E4
code and all original 11 extracted kernel bodies are preserved. Reuse the
existing compact native operators/loader exactly via inclusion. Entry dispatch
alone does not demonstrate architecture fidelity; actual operator provenance,
packed-only fields and the state/readout execution supply the scoped evidence.

Target D512/L12/SSM10/SWA2/72 ternary H128 functions/top8/AQ63/F32 controls/full
V65537 head, single core/thread, nearest-even/FTZ-off/no fast math. Activation
SiLU, source-informed scan and norms are existing costed extensions, not an
identity with original D256/L6/V1024 E4. Router remains flat O(n).

## Persistent stream and canonical chat

Binary pipes use uint32 IDs, versioned headers, full-history prompts and both
EOS IDs11/228. Consume every emitted ID including final/EOS. Every generated
ID also executes its full next head; the extra unused final head is charged.
Store only token IDs and bounded recurrent/SWA state, not unbounded KV attention.
Reuse state only if the entire consumed prefix is an exact prefix of the new
canonical input. Otherwise reset and charge full replay. Token decode/re-encode
is not presumed lossless. Source Rust tokenizer and original Jinja template are
used without loading Torch/Transformers or donor weights. Plain roles supported;
tool-use capability, context quality and user-facing streaming are outside scope.

ONE native process, FIVE requests, fixed consumed FIT cases (no fresh quality):

1. fit_history_00 complete prompt, force reset, max64 generation, normal EOS.
2. First half of that prompt, force reset, zero generation.
3. Append its remaining prefix, max64 generation. Require generated IDs and
   ALL65537 final next-logit bits exactly equal request1. This paired execution
   changes segmentation to verify state continuity; it is not an old-run replay.
4. Original messages plus the actual generated answer and fixed user followup
   `Repeat your last answer in one short sentence.`, max32. Require exact reuse
   if canonical roundtrip preserves cached IDs, otherwise safe reset/replay.
5. fit_instruction_00 canonical prompt, max32. Require history-change reset.

Retain every emitted ID/reply including truncation/EOS, final full rows, request
counts and timing; no unfavorable-output filtering. No teacher/student calls,
optimizer/GPU use, no old C fixed-prefix or model/capture replay. Original source
template/IDs are checked against the two retained FIT packets. Require finite
full rows, exact core/head/cache counters, no earlier EOS in emitted sequence,
unchanged original body hashes, successful compile/process/resource receipts.

Report full readout/core/MLP/forward/prefill/decode/request wall times, model load,
tokens actually emitted, one-core raw rates separately for every request. First
request includes its prefill; report load separately. Raw decode rate is not
quality-qualified accepted throughput. No physical cache flush or DRAM counters;
packed footprint and logical coefficient work are not measured physical traffic.

## Budget, binding and stops

ONE local family <=150s/4GiB OS/8MiB output; worker15s reserve, compile30s,
native session90s checked between forwards, pipe95s maximum. No concurrent
model/compiler timing. Bind model/export terminal, original body/header hashes,
engine/native/entry/client/probe/launcher/protocol/Python/compiler extents,
source tokenizer/template, consumed corpus and three foreign tracked hashes.
Selected runtime scope is explicit, not a complete compiler/DLL attestation.
Keep first fault/completed request packets; never restart due observation expiry.

ENGINE_CHAT_APPARATUS_PASS is an engineering gate only. Quality, numerical
transport, useful large n/structured winner+mass/physical DRAM/multiple families
and same-artifact accepted50 remain open. A slow result changes the deployment
budget before T4; a fast result does not justify month-scale training on128 tiny
templates. Long offline work needs broader data, controlled conversion stages,
measured T4 feasibility and communicated GPU-hour budget/stops.
