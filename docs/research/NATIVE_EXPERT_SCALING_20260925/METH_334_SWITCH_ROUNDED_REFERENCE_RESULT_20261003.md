# METH-334 result: complete prescribed-arithmetic correctness PASS

Freeze5b1e15e, rawfd1fabf. Original333 primitive failure777d97b preserved;
unchanged333 native source/engine macro used. Authoritative exec98383 exit0.
Command/protocol: [METH-334](METH_334_SWITCH_ROUNDED_REFERENCE_PROTOCOL_20261003.md).
Raw SHA256 `155fd1bc3f5894e9450a6ba2e22f4d08e1e974582e3d27bff62e52ee3f8eafaa`.
Native binary SHA256 `40148eea90e5d826100cc1a877a76ef3418003b8c4eec480039ff9e969f00c3a`;
compiler/runtime exact pinned328. All raw commands/stdout/stderr and matched
reference arrays under `results/native_expert_scaling/meth334_switch_full_source`.

## Fixed controls

Independent NumPy primitive oracles: F64 matrix/attention rounded F32 EXACT;
softmax maxabs1.86264514923e-9; norm EXACT after explicit root/reciprocal
rounding. On SAME operands standard Torch F32 sqrt differs from F64-root
rounded F32 by1.19209289551e-7; standard versus rounded reciprocal is0.
This isolates a root implementation difference on these consumed controls;
no universal primitive-error or quality claim. Installed packages unchanged.

Both exact328 Tiny weights/caps1/64: every state/logit/probability has zero
measured difference against matched reference; same discrete choices/capacity/greedy. All nine native
semantic faults detected (relative full-logit errors0.038579..1.0). Independent
Torch vectorized mm/bmm/softmax calls recorded:91/36/31 per Tiny capacity.
No copied native reductions; norm instance methods restored after reference.

Fresh whole original archive/sidefile hashes and ALL6392 native actual mapped
tensor hashes EXACT327. All original14,664,154,368 architectural unique
parameters retained. Native source manifest matches329; zero copied weights.

| Consumed engineering case | Encoder / decoder / full-logit relative L2 | Selected probability maxabs | Exact route/capacity/greedy | Decision |
| --- | --- | --- | --- | --- |
|0|9.9770e-8 /5.1385e-9 /9.2088e-8|1.6391277313e-7|PASS|PASS|
|1|3.7210e-8 /1.0765e-8 /2.8345e-7|1.4901161194e-7|PASS|PASS|

Maximum individual source state relative error `7.50328016762e-07` <1e-4;
probability <=1e-6 unchanged against prescribed arithmetic. Full zero-head
fault error1.0 both. Reference intercepted mm/bmm/softmax604/216/146 case0,
709/264/178 case1. Complete encoder/decoder/logits/routes saved in matched NPZ
with raw-record hashes; cache used autoregressively. No source cases reselected.

## Cost and decision

MAIN125.282s AFTER imports, maximum checked combined6,793,158,656B,
end controller4,807,856,128B, within20min/32GiB. CPU1thread, no GPU/download/
training. Subprocess times are apparatus records, not accepted token rate.

**PASS for implementation correctness of specified target arithmetic only.**
This is a NEW explicitly matched numerical estimand. Original329/330/332
probability1e-6 versus originalCPU1 donor remain FAIL under immutable protocols.
It does not substitute target arithmetic for original pretrained quality.
Two consumed engineering inputs and Tiny tests do not establish heldout
reconstruction/prediction/generation/tasks, useful extra n or50 accepted/s.
F64 apparatus is not the final fast compact LUT artifact.

Next implement a separately frozen all-bank compact representation/exact
integer-and-scale reference and real CPU cost preflight. Keep every real256
expert per bank. ORIGINAL unmodified official donor remains PRIMARY for a NEW
prospectively specified untouched quality cohort. Real smaller-n128 payload,
useful larger RAM-driven n, routing/LUT/DRAM cost, same-artifact quality/rate,
multiple-family/~100B remain unverified. Goal active.
