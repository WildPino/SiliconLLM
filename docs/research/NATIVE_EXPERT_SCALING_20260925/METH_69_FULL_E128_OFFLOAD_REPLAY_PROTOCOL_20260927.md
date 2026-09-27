# METH-69: replay the validated E128 donor training under CPU offload

**Question.** METH-68 proves three synthetic steps can match dense
AdamW. Reproduce the original METH-55 complete 16-update E128
Qwen2.5-0.5B-Instruct train with factors in CPU RAM and selected rows
on GPU. This is a numerical implementation gate before 10× training.

Bind the original METH-55 model, raw corpus, teacher chat rows,
development manifest, RNG seed 555556 and terminal checkpoint SHA-256
`2d652709c6730e2b4f3a1a543834d7eec7409ef0f374738ae24d57ae2a5b5bd3`.
Use its exact A/B/router initialization and its two raw/two chat
microbatches per update, masked CE+KL weights, factor/router learning
rates, no weight decay and global-norm clipping. Keep the donor frozen.
The CPU factor optimizer must reproduce dense AdamW's moment decay and
inactive-row updates as in METH-68. Do not change the training draw
schedule or retune on the viewed METH-55 development set.

After 16 updates compare every factor and router tensor against the
frozen METH-55 checkpoint. Require maximum absolute parameter error
≤2e-5, identical donor-relative prompt top-1 counts on METH-55's
24 prompts, and held-out raw BPB difference ≤1e-5 versus the
original result. Record per-step objective and gradient norms,
factor/route maximum errors, memory, time, source/checkpoint hashes.
The comparison is a replay diagnostic on known data, not a new quality
promotion. A pass permits a new E128/E1280 training protocol with
disjoint development and external semantic material; it does not
establish useful large-E capacity or native throughput.

Run on local RTX 3060 only. Stop above 5 GiB allocated GPU, 8 GiB
RSS or 12 minutes. No T4.
