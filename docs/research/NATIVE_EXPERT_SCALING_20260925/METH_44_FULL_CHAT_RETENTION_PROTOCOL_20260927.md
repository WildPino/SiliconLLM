# METH-44: full-chat retention correction, prospectively fixed

**Observed uncertainty.** METH-43 has exact initial donor logits and
trainable E128 experts, yet its student retains only 89.716% of donor top-1
IDs on chat prompt positions. Its chat loss masked both cross-entropy and KL
to assistant response positions within a trailing 129-ID window. This leaves
most user prompt positions without an explicit retention loss. Test one
prospectively specified change to that objective; do not tune on the viewed
METH-42 prompts or promote from this small smoke alone.

Bind Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, weight SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
the METH-43 teacher result SHA-256
`f42f678280a9086a0fc2ec6e57a712e33fdea8f5cb62bdac7f53c9f28a5f1dd7`,
and the original METH-15 raw train file/IDs hashes. The new
[24-prompt development manifest](meth44_instruct_fresh_chat_manifest.json),
SHA-256 `703abca3f83a983d9380ba84c1a79744d15881d638174b064cf38a47453f47ba`,
was frozen before METH-44 training. Its [builder](../../../benchmarks/donor_adaptation/s1/meth44_instruct_fresh_chat_manifest.py)
uses NumPy seed 444444, selects 24 distinct h0 corpus rows excluding all
256 METH-43 chat-training source rows, and applies the same Instruct
summary template. Exclude these 24 rows from all METH-44 raw-window samples.
This same-corpus set is only a fresh development retention screen, not
an independent final chat-quality promotion set. The 24 viewed METH-42
prompts may be logged as secondary diagnostics, never as the gate.

**Single recipe.** Start from a fresh, frozen Instruct donor, not the failed
METH-43 checkpoint. Wrap all 24 FFNs with E128/top-4/rank-8 residual experts,
output factors zero and METH-15 seed 5151. Verify disabled/enabled logits
have maximum absolute difference ≤1e-3 on raw and chat inputs. For 16
updates use AdamW with factor LR `3e-4`, router LR `3e-5`, weight decay zero
and clip norm 1.0. Each update uses two independently sampled raw training
windows of 128 IDs and two independently sampled saved METH-43 teacher-chat
rows. Seed NumPy sampling 544444. Raw objective is student next-token CE
plus `0.5 ×` frozen donor→student KL over all 127 target positions. For
chat, concatenate the frozen full prompt and saved donor continuation;
student CE is averaged over assistant response targets only, while
`1.0 ×` donor→student KL is averaged over **all** full-sequence next-token
positions, including user prompt positions. Average the four microbatch
objectives. Do not regenerate, filter or select teacher responses.

Stop on source/mask/hash mismatch, nonfinite update, missing positive minimum
output-factor gradient at update 1, missing positive minimum router gradient
from update 2, or budget breach. The smoke passes only if all 16 updates
finish, every layer changes at least 16 expert output slots, heldout raw
student-minus-donor BPB ≤+0.05, and ≥95% top-1 agreement on all positions
of the new 24-prompt development set. Preserve optimizer and RNG state
locally. Time cap ≤10 minutes, peak allocated GPU ≤10.5 GiB, process RSS
≤20 GiB, local RTX 3060 only. No T4 job.

Even a pass would authorize only a separately preregistered longer
continuation and fresh external document/task/generation evaluation. It
would not prove large-E quality, CPU LUT scalability, packed export or
≥50 accepted batch-1 tok/s in `benchmarks/phase60/engine.c`.
