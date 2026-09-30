# METH-189: rank-64 FFN correction on the stored compact core

METH-188 isolates 4.361 of 5.318 pooled prompt-ranking points to the
grouped-Q6 FFNs alone. Train exactly one rank-64 low-rank correction
for each of the 72 stored Q6 FFN matrices while keeping the exact
METH-186 R8 tied head, Q6 codes/scales, BF16 attention, FP32 controls,
and centered E1280 bank frozen. The teacher is BF16 donor+the same
centered E1280 bank. No new quality source is consumed by training.

## Representation and training

For each gate/up/down linear map with input width `i` and output width
`o`, use trainable FP32 factors `A[64,i]` and `B[o,64]`; the forward map
is the dequantized stored Q6 base plus `B(Ax)`. Initialize `A` from a
fixed seed with standard deviation `1/sqrt(i)` and `B=0`, so the
initial student is exactly the METH-187 Q6+E1280 arm. Do not update
the shared bank, router, head, base codes or scales. Export only the
72 pairs in BF16, verify exact safetensors readback, and charge both
factors to every token. The fixed payload is 26,542,080 elements or
53,084,160 BF16 bytes; combined ideal active total is 534,619,136
bytes/token, 25,380,864 under 560 MB. The full E1280 bank remains
resident in RAM. This is a cost screen, not proof of actual DRAM
traffic or C speed.

Use the existing hash-checked METH-136 train-only raw and long-chat
corpus construction: 256 seeded updates, one 128-token raw sequence
and one teacher-continuation chat sequence per update, with chat prompt
positions masked out. Keep its raw/chat draw ordering and source
exclusions. Distill BF16+E1280 logits with the METH-88 KL plus
confident teacher top-1 hinge (threshold and margin 0.2, hinge weight
0.1). Average raw and chat losses. AdamW, learning rate 3e-4, no
weight decay, clip global norm to 1.0. Train with activation gradient
checkpointing on the local RTX 3060. Record per-update loss and
draws. No tuning on the 24 viewed METH-121 sources.

## Stops and evaluation boundary

Cap training at 15 minutes wall, 10.5 GiB allocated GPU memory,
20 GiB RSS and 1 GB new checkpoint/report disk. At update 16, stop
early if projected 256-update wall time exceeds 15 minutes; stop on
nonfinite loss/gradient or binding/parity failure. Write a diagnostic
progress report at updates 1, 16, 64, 128, 192 and 256. The trained
checkpoint is a candidate only. Next, replay the exact METH-187
already viewed development arms with the stored BF16 correction,
requiring pooled BPB delta <=0.01, category delta <=0.02, pooled
ranking delta >=-1 point and each category >=-2 points versus
BF16+E1280. If that fails, do not spend fresh evaluation sources or
native full-rate work on this artifact. If it passes, freeze a fresh
source-disjoint quality protocol before looking at any new sources.
Native C correctness, LUT scaling and >=50 accepted tok/s on the same
artifact remain mandatory. No T4 is authorized by this protocol.
