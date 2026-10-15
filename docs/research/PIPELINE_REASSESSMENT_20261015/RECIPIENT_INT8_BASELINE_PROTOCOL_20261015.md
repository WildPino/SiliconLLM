# Complete original-recipient int8 baseline: prospective protocol

15 October2026. Prepared before export, compilation or numerical execution.
This is a cost and decoded-wire screen for a possible pretrained recipient
in the [donor-memory program](CARVING_DECISION_20261015.md), not a new attempt
to promote a failed conditional student or establish the final large-model goal.

## Fixed artifact and complete computation

Use the original Qwen2.5-0.5B-Instruct BF16 snapshot at revision
`7ae557604adf67be50417f59c2c2f167def9a775`, source SHA256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
All24 layers, D896, FFN4864,14 query/2 KV heads,151936 vocabulary rows remain.
No learned bank, removed layer, artificial context vector or donor-state
substitution is involved. No network or GPU calls.

Reuse `qwen_export.py` with int8/R8, BF16 source loading, fold none and explicit
head untying via its historical `--head-ternary` flag. In this int8 file the
explicit head is int8, not ternary. Embeddings, norms and biases are F32;
all168 layer matrices and the head are signed-I8 codes plus F32 row scales.
The original embedding remains the input lookup table. R8 scale is
`max(maxabs(row)/127,1e-12)`; codes are nearest-even, clipped[-127,127].

Compile `engine.c` with `SILICON_QWEN_RECIPIENT_I8`. Its small wrapper reuses
the existing complete `donor_engine.c` loader and forward function. Fix four
matvec accumulators, AVX4 attention, original libm exponentials, no LUT/fusion/
head shortlist or optional banks, and six workers on six distinct physical
cores (logical0,2,4,6,8,10). Start/end thread affinity readback must pass.
The separately prepared W4 recipient profile is not selected or measured.

## Why the old evidence is insufficient

METH127's original Instruct F32 C arm includes attention/KV/FFN/head and has
64/64 scoped Torch top1 parity but only18.244 decode tokens/s. The historical
base-model one-byte68.52 measurement is not an Instruct result. METH59's
Instruct R8 prompt-ranking gate failed; its result is preserved. A future
jointly recovered recipient may differ, but this run grants no fresh chatbot
quality, semantic or donor-transfer admission. Do not interpret decoded-codec
parity as preservation of the BF16 donor.

## Numerical gate before cost

The preparation binds original source/config/tokenizer, scripts, compiler,
OpenMP runtime and relevant CPU Python implementation/native libraries.
No new user corpus or sealed evaluation cohort is opened. Three public
engineering prompts (English explanation, Italian planning and Python coding)
are encoded with the pinned canonical chat template. Their exact IDs/hashes
are retained before execution. The first32 IDs of the first prompt form the
native probe; all32 complete vocabulary logits are retained.

The independent auditor parses all290 source tensor names/shapes/extents,
every original F32 widening and all169 emitted code/scale matrices. It checks
every stored value against independently stated R8 derivation, exhausts the
archive and reconstructs an untied FP32 eager reference from the **decoded
artifact only**. All32 native logit rows must be finite, have identical top1
and relative L2<=1e-3. Different association/FMA is allowed only within that
predeclared numerical tolerance. Source/model execution shares Torch/NumPy
operators; this is not independent verification of those libraries. Full
input hashes are checked before and after. Any failure stops before timing.

## Fixed cost observations and decision

Run three independent repetitions, each using a new process for every case,
in fixed order: the three canonical prompts, then two context-only fixtures
containing512 and2048 real tokenizer IDs from repeated library prose. All
cases request up to64 own-history greedy tokens, stopping at either canonical
EOS151645/151643. Context fixtures are cost probes, not useful-chat evidence.
No schedule, thread count, waiting policy, context or quantizer grid follows.

Load/initialization, prefill and decode are separate. Each non-EOS emitted
token is actually forwarded, including the final token, to conservatively
charge a complete native step; an EOS scan is timed but has no native forward.
Report all emitted IDs, per-step time/context, EOS, prefill, native-forward
count and process wall/resources. The prospective cost allowance is worst
process average<=18ms per charged native forward, with per-case max/min rate
spread<=1.10. This leaves a tentative2ms for fusion/frontend within20ms;
it is not a measured allowance or final useful50 admission. Report every
process and tail even if that screen fails. Scalar/physical addressed-byte
counts are not PMU-measured DRAM.

A cost PASS licenses a bounded memory-fusion/extraction protocol and further
whole-artifact quality work. A cost FAIL rejects this fixed recipient encoding/
execution envelope; it does not prove that another encoding, runtime or
learned recipient cannot work. Old quality failures remain failures.

## Resources and closure

Preparation hashes/tokenizes only. Freeze code/protocol before preparing the
binding and running. Export, compile, numerical audit and cost sequence are
each capped at900 seconds, entire sequence2400 seconds,8GiB observed host RSS
per child and2GiB outputs. All child commands have owned PID/create-time,
exclusive logs/receipts and recorded exits; stop only owned processes on cap
or failure. The controller keeps the first fault and all partial outputs;
no automatic rerun. Python and native outputs, compiler/executable/source
identity, all observations and final seals remain reviewable. No T4 job is
launched and no long-job pause is required for this bounded CPU inquiry.
