# METH-65: one real Instruct-donor update with CPU-resident E1280 factors

**Decision question.** METH-64's offload gradients are correct on a
synthetic residual. Test whether the same path survives a complete
24-layer Qwen2.5-0.5B-Instruct forward/backward with frozen pretrained
core and independently allocated E1280 top-four residual factors.
This one-update apparatus screen does not establish learned capacity.

Use source model SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
the exact tokenizer fingerprint and training corpus IDs already bound
by METH-55. Start with rank-8 A factors and zero B factors, rank-64
32×40 product keys, all independent per pair and layer. Freeze donor
weights. Require max initial donor/student logit error ≤1e-3 on one
16-token raw prefix and one 16-token chat prefix. Use two fixed
microbatches, one 128-token raw and one existing donor-continuation
chat training item, with the METH-55 masked CE+donor-KL objective.
Backpropagate both through the full donor, accumulating CPU selected
factor gradients. Require finite nonzero B gradients on all layers,
finite router gradients (zero at the initial B=0 state is expected),
and no dense E1280 gradient tensor. Apply one sparse row-wise Adam
factor update (LR 3e-4, β1=.9, β2=.999, ε=1e-8, no decay) and one
GPU AdamW router update (LR 3e-5, no decay). Require at least one B
slot changed per layer and finite subsequent student logits. Save
the exact selected row counts, loss, peak GPU/RSS and update time;
keep the result and source hashes. No checkpoint promotion or quality
claim follows from one update.

Stop above 5 GiB allocated GPU, 12 GiB RSS or 15 minutes. Use only
the local RTX 3060 and do not disturb other processes using the GPU.
If this passes, a separate frozen E128/E1280 controlled training and
external audit can evaluate learned distinct expert quality and route
load. The BF16 donor wrapper used here is not the rejected METH-62
compact core or a native `engine.c` result.
