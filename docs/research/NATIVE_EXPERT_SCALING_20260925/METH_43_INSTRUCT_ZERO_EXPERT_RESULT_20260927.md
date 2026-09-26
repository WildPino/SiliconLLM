# METH-43: Instruct zero-output E128 smoke result

The [frozen protocol](METH_43_INSTRUCT_ZERO_EXPERT_PROTOCOL_20260927.md)
and [chat-training manifest](meth43_instruct_chat_train_manifest.json)
were committed before teacher generation. The BF16 Qwen2.5-0.5B-Instruct
teacher generated [256 saved responses](meth43_instruct_teacher_chat_result.json)
on the local RTX 3060. Mean continuation length was 49.684 tokens; 188/256
ended with EOS and 1/256 repeated an 8-gram three times. All 256 training
windows were retained without selecting responses by content. An independent
readback found 256 unique source rows, zero invalid 129-ID window hashes,
zero invalid 128-position assistant masks and 12,719 assistant target IDs.
Teacher generation took 503.328 s, used at most 1.014 GB allocated GPU and
2.543 GB process RSS, within its declared caps. The teacher-result SHA-256 is
`f42f678280a9086a0fc2ec6e57a712e33fdea8f5cb62bdac7f53c9f28a5f1dd7`.

The [bound smoke runner](../../../benchmarks/donor_adaptation/s1/meth43_instruct_zero_expert_smoke.py)
completed 16 updates and saved the [full measured result](meth43_instruct_zero_expert_smoke_result.json),
SHA-256 `0c6c185f2f9e58b05caa4cac19911565aab99745551bb9a911fe08ecd7c04bcd`.
Step-zero raw and chat logits were identical to the frozen donor (both
maximum absolute difference 0). Every layer received a positive output-factor
gradient from update 1 and a positive router gradient from update 2. The
smallest changed output-slot count among 24 layers was 125/128. The donor
heldout raw BPB remained 0.971255; the student's BPB was 0.958693, a
−0.012562 difference. The smoke took 27.141 s, with peak allocated GPU
2.327 GB and final process RSS 3.145 GB. The local optimizer/RNG checkpoint
is `benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth43_instruct_e128_smoke.pt`,
SHA-256 `ef87d2b3112c98e081f07de052186cff068bed34095a485908e3df8c5cfd1bf7`.

**Decision: stop this checkpoint.** Its prompt-position top-1 agreement on
the 24 fixed METH-42 chat inputs is 3629/4045 = **89.716%**, below the
predeclared 95% development threshold. Category breakdown: code 1292/1428
(90.476%), prose 1135/1289 (88.053%), and technical-general 1202/1328
(90.512%). A raw-text BPB improvement is insufficient to promote it.
This is a 0.5B/E128 training-apparatus result; it provides no evidence yet
for larger learned expert banks, CPU LUT rate, or native `engine.c` quality.
The METH-42 prompt set has now been viewed by a failed gate and is not a
target for tuning or an independent gate in the next recipe.
