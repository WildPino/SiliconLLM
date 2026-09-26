# METH-43: Instruct donor, zero-output E128 experts and chat retention smoke

**Uncertainty.** METH-42's same-geometry Instruct donor passes the
chat-format repetition screen, but grafting the base-trained
adapter changes 15.723% of prompt top-1 IDs and fails its ranking
gate. Test whether newly initialized E128 residual experts start
at exactly the Instruct donor and can receive useful gradients
under a bounded raw-text objective plus explicit teacher-chat
retention. The 16-update run is an apparatus/early quality gate
for a longer training continuation, not full conversion.

Bind Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`,
weight SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
tokenizer fingerprint `4efeeb9382a77a06` and original METH-15
training file/ID hashes
`0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9` /
`9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da`.
The [chat-training prompt manifest](meth43_instruct_chat_train_manifest.json),
SHA-256 `17921cb83c1d10793e2a56df6aa899039bd794757eeff43921068b8de2096055`,
was frozen before teacher generation. Its
[builder](../../../benchmarks/donor_adaptation/s1/meth43_instruct_chat_train_manifest.py)
chooses 256 distinct rows of the 31,250×512 Qwen training IDs
using NumPy seed 434343, decodes each row's first 96 IDs and
applies a fixed Instruct chat summary template. Prompt lengths
are 157–159 IDs. METH-42's 24 diagnostic prompt sources are
not sampled for chat training.

**Teacher data preparation.** On the local RTX 3060, load the
bound BF16 Instruct donor. For each of the 256 frozen prompts,
generate up to 64 greedy tokens with KV cache and EOS. Keep
every continuation, including short or non-EOS output, and
report lengths/EOS/repetition. Make a 129-ID trailing window
from prompt plus continuation, with assistant-target mask on
continuation IDs only; this yields 128 next-token positions
per row for subsequent student training. Save the entire raw
teacher data and window IDs/masks with manifest and donor hashes.
Do not select responses after observing their content. Stop if
a response has zero tokens, any window/mask has an invalid
shape, or source/model hash mismatches. Preparation budget:
≤10 minutes after model load, ≤10.5 GiB allocated GPU, ≤20 GiB
RSS, local GPU only, no T4.

**16-update smoke recipe, fixed before teacher outputs.** Freeze
all Instruct donor weights; wrap all 24 FFNs with E128/top-4
rank-8 residual experts. Initialize output factors to zero,
random input factors/router with seed 5151 as METH-15, and
verify disabled/enabled logits on bound train/chat inputs
have maximum absolute difference ≤1e-3 at step zero. AdamW
factor LR 1e-3/router LR 1e-4, weight decay 0, clip 1.0.
Each update has three independently sampled raw Qwen training
windows (128 IDs) plus one sampled frozen teacher-chat window
(128 input and next-token targets). Raw microbatch objective:
student CE + 0.1 teacher→student KL. Chat microbatch objective:
student CE + 0.5 teacher→student KL, both masked to assistant
target positions. Divide the sum of four microbatch objectives
by four. Donor runs with experts disabled under `no_grad`;
student runs enabled. No donor parameter may update.

The smoke passes only if step-zero identity, first-update
minimum output-factor gradient >0, second-update minimum
router gradient >0, all 16 updates finite, at least 16 changed
expert output slots per layer, heldout raw student-minus-donor
BPB ≤+0.05 and ≥95% prompt-position top-1 agreement on the
24 fixed METH-42 chat inputs. The latter is a **development**
retention screen on already viewed prompts, not an independent
promotion gate. Preserve optimizer and RNG state in a local
checkpoint for a prospective longer continuation. Smoke budget:
≤10 minutes, ≤10.5 GiB allocated GPU, ≤20 GiB RSS. On any
failed gate, stop rather than treating a raw BPB improvement
as sufficient. No native speed or large-E claim follows.

If the smoke passes, preregister a longer continuation and new
held-out document, task and chat-generation sets before using
that checkpoint for promotion. The final target still needs
packed export, `engine.c` parity and ≥50 accepted tok/s on
the same quality-valid artifact, plus learned expert-count
scaling and family transfer.
