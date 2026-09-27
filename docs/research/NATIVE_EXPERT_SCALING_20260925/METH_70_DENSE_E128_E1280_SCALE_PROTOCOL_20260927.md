# METH-70: dense METH-55 recipe at E128 versus E1280

**Question.** METH-66's CPU-offloaded comparison changes both expert
count and the original METH-55 training recipe. The local RTX 3060
currently has sufficient free VRAM to test the original dense
AdamW recipe with 10× experts directly. Determine whether increasing
independent learned expert slots from 128 to 1,280 preserves donor
retention on new development prompts, and measure route concentration.
This is still Qwen0.5B geometry, not a 10B/100B or native-rate proof.

**Frozen arms.** Use the bound Qwen2.5-0.5B-Instruct donor, width 896,
24 layers, rank-8 FP32 A/B residual factors cast to BF16 in forward,
rank-64 FP32 product-key router, exact top-four pair selection and
zero-B donor parity. E128 uses 8×16 product keys; E1280 uses 32×40.
All expert tensors have independent storage and initialization from
the METH-55 seed rule; no replicated learned rows. Use dense GPU
AdamW for both arms, same two raw/two full-chat microbatches per
update, METH-55 CE+KL weights, factor LR 3e-4, router LR 3e-5,
global-norm clipping 1.0, no weight decay, and 16 fixed updates.
Use identical NumPy training draw seeds/schedule for the two arms.

**Fresh development set.** Before inference, select a seed-707070
24-prompt manifest from the bound raw corpus. Exclude the 256 chat
training rows and METH-44/47/55/56/66 development rows (five prior
sets of 24); exclude all of them and the new rows from raw sampling.
Commit the manifest before either arm trains. These prompts test
donor-relative position top-1; a later external semantic audit needs
new source-disjoint content.

**Gates and telemetry.** Require exact donor/student logits at initial
16-token raw/chat prefixes and raw BPB parity. After 16 updates,
require each arm ≥95% donor top-1 on the new 24 prompts, raw held-out
ΔBPB≤+0.05, ≥64 changed B slots/layer at E128 or ≥640 at E1280,
and exact product-key top-four versus exhaustive pair scores on all
evaluated hidden states. Record selected slots/layer, maximum/mean
route load, per-step B/router gradient norms, task memory, full
checkpoint/optimizer/RNG hashes and runtime. Do not turn a favorable
BPB or changed-slot count into a quality pass when top-1 fails.

Stop an arm on nonfinite values, >10.5 GiB peak GPU allocation,
>20 GiB RSS or >15 minutes including checkpoint save/hash. Save
partial evidence on stop. Local RTX 3060 only; no T4. Passing this
smoke would permit a longer retention run and independent generation,
task and blind semantic audit. It would not establish CPU LUT speed,
full `engine.c` integration or ≥50 accepted tok/s on one artifact.
