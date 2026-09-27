# METH-66: controlled E128 versus E1280 joint donor-transfer smoke

**Motive.** METH-65 proves one real E1280 update fits, but the added
slots may stay unused or damage the donor. Compare two independently
initialized rank-64 product-key geometries, 8×16=E128 and
32×40=E1280, with otherwise identical rank-8 top-four residuals,
training examples, objective, step count and sparse CPU optimizer.
This 16-update screen can qualify a longer independent audit; it
cannot prove final semantic quality or 10B/100B transfer.

**Bound inputs.** Use Qwen2.5-0.5B-Instruct source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
the METH-55 donor-continuation chat training manifest and fixed
31,250×512 raw token corpus. Start both arms at exact donor parity
with independent A rows and zero B; no copied expert slots. Freeze a
new seed-666666 24-prompt development manifest before either arm
trains, excluding the 256 chat-training rows and all METH-44/47/55/56
development rows. Exclude those and the new rows from raw sampling.
The 24 development prompts measure donor-relative top-1 only; they
cannot serve as a later external semantic audit.

**Training.** Run 16 updates/arm, two raw and two full-chat
microbatches/update, using the same NumPy draw schedule for both
arms. Raw and chat masked CE plus donor KL use METH-55 weights 0.5
and 1.0. Keep donor core frozen. Use CPU-resident FP32 A/B factors,
selected-row GPU transfers, row-wise sparse Adam at LR 3e-4,
β1=.9/β2=.999/ε=1e-8, and GPU AdamW product keys at LR 3e-5,
no weight decay. Clip aggregate trainable gradient norm to 1.0.
Save source, corpus, manifest, RNG and checkpoint hashes. Factor
states and optimizer moments are distinct for every expert row.

**Terminal checks and stop.** Verify exact donor parity at initialization.
On the new 24 prompts, require each arm ≥95% donor-position top-1;
on the fixed held-out raw 4×256 slice, require ΔBPB≤+0.05.
Require at least 64 changed B slots/layer for E128 and 640 for
E1280, nonzero finite B/router gradients after their expected initial
zero-router step, exact product-key versus exhaustive top-four on
development hidden states, and report selected-slot counts plus
maximum/mean route load by layer. Record donor and each arm's losses,
memory, elapsed time and checkpoint bytes. If a bound fails, stop that
arm and preserve the result; do not tune on the viewed prompts.

Use only local RTX 3060. Stop above 5 GiB peak allocated GPU,
12 GiB process RSS or 30 minutes per arm. No T4. Passing both arms
would justify a frozen longer continuation and source-grounded blind
semantic/task audit at E1280. It would not establish a compact LUT
bank, full `engine.c` model or ≥50 accepted tok/s.
