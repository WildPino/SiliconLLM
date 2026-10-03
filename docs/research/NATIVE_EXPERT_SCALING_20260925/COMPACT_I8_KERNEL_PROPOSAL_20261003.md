# Exact signed-I8 dot with biased bytes and two-part activation: proposal

**IMPLEMENTED/MEASURED as305: exact outputs/controls PASS,22ms cost FAIL.** No teacher
collection/training or quality promotion is licensed. This prospective
implementation changes integer instructions/storage representation, keeping
the SAME signed-I8 weights/input quantization/scales, model geometry and math.
It must earn its own frozen native14ms cost gate and exact303 output controls.

## Independently derived integer identity and saturation bounds

Current303 row dot widens signed bytes toI16 before AVX2 multiply/add. For
each original signed weight w in[-127,127], store u=w+128 in unsigned[1,255].
This is a reversible representation, not new quantization or an extra byte.
Input quantization remains x in[-127,127], original max/127/rounding/scales.
Prepare h=trunc(x/2) in[-63,63], l=x-2h in[-1,1], and sum_x=sum(x).
Then exactly, using integer operations:

`sum(w*x) = 2*sum(u*h) + sum(u*l) - 128*sum_x`.

AVX2 unsigned-byte×signed-byte adjacent-pair add has I16 saturation. Direct
full u*x can overflow a pair; it MUST NOT be used. With the two prepared
parts, pair magnitude is at most2*255*63=32,130 for h and2*255*1=510 for l,
both<=32,767. Widen each pair sum toI32 using multiply/add with I16 ones,
accumulate independently, then combine and subtract128*sum_x. Even the
unbiased intermediate full dot is bounded by1536*255*127=49,743,360, safely
inside signedI32. Exact final signed dot equals original303 scalarI64.
Prove every intermediate bound before coding; use compiler intrinsic source
definitions to verify instruction signedness/saturation rather than assuming.

This replaces repeated signed-I16 expansion with unsigned/signed byte
products. It adds two input buffers/one sum and row correction. Whether the
new instructions, input conversion and memory pattern are cheaper on this
CPU is UNKNOWN. No throughput prediction or compiler speedup is claimed.
Do not borrow304's22ms as the original303 cost or assume a14ms pass.

## Historical prospective implementation and frozen controls

[305 measured result](METH_305_BIASED_I8_PREFLIGHT_RESULT_20261003.md) now
closes this unchanged exact-I8 kernel before teacher fit. The requirements
below describe its pre-observation proposal, not an unrun next step.306/307
packed-I4/worker-wait evidence and current next action are in INDEX.

Add a NEW helper/controller/protocol/phase60 selector; preserve303/304.
Physically populate the complete640-function bank with biased representations
of the identical seeded303 signed codes; controls independently regenerate
and decode every bank edge. No reduced bank, quality subset or extra capacity.
Keep all308 matrices, source macro routing/norms/Q6 head, activation scales/
rounding, all nonlinear/residual fixture work and original operation order.
Charge input splitting/correction/all routing/dispatch inside operator timing.

Before native observation, freeze numeric/extreme/fault/finite/address controls,
all30 original303 output/route hashes, same10 fixtures/3 repetitions/2warmups,
repeatability<=1.10/EACH median<=14ms, source/component provenance and600s/
24GiB/six-thread resource gates. Exhaustively check representable weight/input
products and worst adjacent pairs against scalarI64; test signed extrema and
row corrections. Include actual source Q6 head, no free head/cache/shortlist.

If stable cost still fails, stop unchanged geometry training and record the
negative result before another scientific variable. A pass licenses only
separately frozen teacher-function fit using existing298 calibration where
appropriate, followed by independent composed donor-relative quality,
useful learned child diversity/exposure, causal native cache/routing/head,
actual DRAM costs and SAMEartifact accepted>=50 batch1token/s. Multiple actual
families/scales and useful RAM-dependent n remain final requirements.
