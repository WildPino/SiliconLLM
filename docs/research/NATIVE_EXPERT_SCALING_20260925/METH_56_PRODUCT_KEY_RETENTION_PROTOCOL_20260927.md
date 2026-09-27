# METH-56: longer donor retention for the joint product-key E128 model

**Motive.** METH-55's 16-update E128 product-key smoke passes initial
retention, but METH-44 also passed at 16 updates before METH-45 lost
chat agreement at update 512. Continue exactly the METH-55 checkpoint
under the stronger METH-47 donor objective to test whether this new
router geometry avoids that failure. This does not test distinct
large-E quality or native throughput.

Bind the METH-55 checkpoint SHA-256
`2d652709c6730e2b4f3a1a543834d7eec7409ef0f374738ae24d57ae2a5b5bd3`,
Qwen0.5B-Instruct donor and METH-44 raw/chat assets. Resume its
optimizer and NumPy/PyTorch/CUDA RNG state at update 16. Use two raw
and two full-chat microbatches per update, assistant CE and donor KL
on all raw/chat positions with weights 2.0/4.0. Factor LR 1e-4,
product-router LR 1e-5, AdamW no decay, clip 1.0. Target update 512;
evaluate at updates 256 and 512. Save both checkpoint states and
all hashes. Stop at 256 if held-out raw ΔBPB>+0.05 or chat-prompt
top-1<95%; do not run the terminal audit after a stop.

Freeze a **new** seed-565656 24-prompt development manifest before
training. Exclude the 256 chat-training rows and all 72 prior
METH-44/47/55 development rows. Exclude all these and the new 24 rows
from raw sampling. Recheck donor BPB and METH-55 student BPB at resume
within 1e-5. At each evaluation, report raw ΔBPB, donor-relative
top-1 over all 24 prompts, changed B slots, product-route selected
slots and worst maximum/mean load. Verify exact 16-pair vs exhaustive
E128 route choices on actual evaluation hidden states. The update-512
gate is ΔBPB≤+0.05, chat top-1≥95%, at least 64 changed B slots per
layer and exact-route oracle. Passing makes the checkpoint eligible
for a separately frozen external, source-grounded generation and task
audit; it is not itself a semantic-conservation claim.

Local RTX 3060 only, ≤10.5 GiB peak allocated GPU, ≤20 GiB RSS,
≤20 minutes wall time. No T4. Preserve the outcome even if a gate
fails; do not tune this same development set after viewing it.
