# METH-55: jointly train product keys and E128 residual experts

**Question.** METH-54's rank-64 additive product keys satisfy the
synthetic CPU 10× cost gate, but no keys have ever selected trained
experts. Test whether this geometry supports an actual donor-parity
start and useful joint updates at E128 before considering a larger
distinct expert ladder. This is a local 16-update smoke, not a
100B-quality or native-rate claim.

Use the exact Qwen0.5B-Instruct donor and validated raw/chat training
assets from METH-44. E128=8×16, 24 layers, top-4, rank-8 A/B
experts. Fresh BF16 donor core, frozen. Per layer trainable FP32
`P[64,896]`, `U[8,64]`, `V[16,64]`; compute additive axis scores,
their top-four each, 16 pair scores, exact top-four pair IDs and
softmax gate. Factor A initialized as METH-15; B=0 so both raw and
chat logits start at donor parity. Keep every expert's A/B distinct.
Apply the METH-44 full-prompt objective: two raw and two chat
microbatches per update, CE on assistant response for chat and donor
KL over the entire prompt/response; raw/chat KL weights 0.5/1.0.
Optimizer AdamW, factor LR 3e-4, router LR 3e-5, no weight decay,
grad norm clip 1.0. Fixed 16 updates and fixed RNG seeds.

Freeze a **new** 24-prompt development manifest from the training
corpus with seed 555555, disjoint by row ID from the 256 chat-training
rows and METH-44/47's 48 development rows. Exclude these rows from
raw sampling. Before training, verify donor parity maximum absolute
logit error ≤1e-3 on one raw and one chat prefix, and held-out raw
BPB equality within 1e-5. Verify top-four candidate search exactly
against the full 128-pair score matrix on deterministic random inputs
before training and on actual hidden states during evaluation.

At update 16 report donor/student raw BPB, donor-relative top-1
agreement on all new chat prompts, changed expert slots and routing
load per layer, including min/max selected and worst maximum/mean
load. Require ≥95% top-1, ΔBPB≤+0.05, ≥16 changed B slots per
layer and nonzero finite B/router gradients (router after update 1)
to allow a later precommitted longer continuation. These automatic
gates do not imply semantic conservation; a source-grounded external
generation set and manual adjudication are required later.

Run only on the local RTX 3060, with ≤10.5 GiB peak allocated GPU,
≤20 GiB process RSS and ≤10 minutes wall time. Save checkpoint,
code/data hashes, result JSON and failure reason even if a gate
fails. No T4 run follows automatically from this smoke.
