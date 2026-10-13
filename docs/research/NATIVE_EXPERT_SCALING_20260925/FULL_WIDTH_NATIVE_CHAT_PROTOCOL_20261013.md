# Full-width native recurrent chatbot: frozen experimental protocol

13 October 2026. Implementation available; no observations/admission yet.

## Question and decision

Do the admitted 3072-channel, state96, own gate-RMS and p6/I16 branches
compose across all24 trained Falcon-H1-1.5B layers while preserving useful
chatbot behavior? Reuse all24 source-only packets/qualified row ABI, first
source-only token generator and the24 existing FIT biorthogonal bases.
No further component tuning precedes this whole-model observation.

If quality passes, this is a transfer backbone for replacing expensive source
functions. If it fails, use the stored causal trajectories to locate accumulated
state/readout error before conditional-capacity work. Complete all audits under
either quality outcome. Fail-Fast is revoked. No retries with relaxed gates.

## Construction and arithmetic

Unchanged411 canonical BF16 source parameters verified before/after. Actual
hidden2048, source FFN/SwiGLU, attention/RoPE, final RMSNorm, head and tokenizer.
Patch ONLY the24 mixer forward methods. Initial mixer uses actual token ID to
select the complete source-only F0 table. Later23 mixers use their actual
candidate hidden states through original source in_proj/effective BF16 MuP.
Their inputs are NOT teacher-state/latent injection. Every mixer owns its
native causal conv ring and3072x96 recurrent state, reset at new sequences only.
Native BF16 output precedes the original decoder's SSMout multiplier, applied
once. No training/optimizer/RESERVED/longalign queries. No D256 hidden projection.

Batch1, no padding, one token per source-operator call, inference/eval,
TF32 disabled, deterministic algorithms. Preserve original attention cache;
each layer's pre-update attention position must equal its native step index.
Source mamba cache is unused. The existing full-prefix exactness gate maps to
the declared serial prefill wrapper versus independent serial continuation
calls; it does NOT assert GPU parallel-prefill equivalence. Compare all16 known
prompts, every native trace/state/ring, attention KV digest, and final logits.
No cache/state reuse across independent cases. Exact site IDs; one function
per site, mass1. This trivial routing is not an expert-scaling demonstration.

## Data and quality gates fixed before results

Known FIT24/DEV24 repaired broad histories:22547 positions,8808 labels.
Teacher BF16 logit custody inherits the original failed family; reusing the
repaired corpus does not retroactively admit its source generation exit.
No fresh donor answer generation needed. Cases and tokenizer from admitted
native-chat binding, donor14/16. Greedy generation max64, EOS11/228, full
scores/argmax/stop/special leak checked. Followup contains candidate's actual
answer. Its serialized prompt hard cap2048, no silent truncation.

Reuse task>=12/16, each category>=2, blank<=1, no special leak, valid stopping,
DEV case-mean KL<=1 and disagreement<=.20, each domain case-mean KL<=2 and
disagreement<=.35. Require own-answer followup correct as an additional gate.
FIT descriptive only. All16 prefix controls bit exact; every actual integer
readout independently exact, mass defect0. Numeric field/projection1e-6,
scan/state/norm1e-4 inherited from row binding. Contextual source generator
F64/BF reference versus original CUDA BF16 rows relative RMS<=.01, consistent
with prior source-generator1% gate. This is an approximation check, not a
claim of CUDA bit equivalence. Preserve per-label and per-case metrics.

## Trace, independent audit and limits

For EVERY consumed token: initial BF16 native output2048; later23 BF16
H2048 after SSMin, preconv row6704 after MuP, native output2048. Row-major
records per site. Record all24 final state/ring/seen/last and attention KV
digests, actual consumed IDs and full generated scores. No terminal predicted
token is consumed after EOS/64; seen = prompt length + generated length -1.
Store all48 label logits as F32 exact decodes of BF16. Full source parameter
canonical hashes and complete input/output file hashes before/after.

Separate CPU-only audit reconstructs EVERY contextual generator from canonical
source G, all native causal fields, projections, F64 recurrent trajectory,
own gate-RMS, exact activation quantizer, F64-exact integer products under
the36.64B bound and BF16 readout. Full native block replay must equal worker
single-token traces and final states/rings. Recalculate every categorical
label, generation argmax, tokenizer result, followup lineage and decision.
Numeric failures accumulate; all sequences still audited. Mechanical faults
retain terminal/partial bytes and do not create a quality verdict.
GPU attention/FFN/head arithmetic remains a producer/source-identity witness,
not an independent CPU reproduction; report this scope explicitly.

Local RTX3060 only, no T4. Producer cap7200s with180s reserved, OS8GiB,
GPU allocated6GiB/reserved10GiB, new data24GiB in the unique E family.
Binder calculates trace/logit/final-state upper bound before launch, leaving
>=1GiB metadata margin; at least26GiB free required. CPU audit cap7200s/OS8GiB.
Both are potentially hours-long: announce, bounded launch, pause goal under
the user's standing instruction. No delegated monitor. Holder DETACHED8
worker, actual PID/create-time custody and watchdog. Deadline/resource/
unexpected child/source mutation are mechanical stops. Observe exit before
any audit/admission; user wake resumes work. No useful TPS estimate from this
hybrid GPU/CPU bridge; final engine, useful conditional n and same-artifact
>=50 accepted token/s remain open.

## Programs

`benchmarks/native_expert_scaling/full_width_native_chat.py`: CPU bind, owned
bounded launch, source-model worker and persistent native adapter.
`benchmarks/native_expert_scaling/full_width_native_chat_audit.py`: complete
stored numeric/behavior audit. Binding/result/terminal paths and exact hashes
are recorded in this directory, payloads in the unique E family.
