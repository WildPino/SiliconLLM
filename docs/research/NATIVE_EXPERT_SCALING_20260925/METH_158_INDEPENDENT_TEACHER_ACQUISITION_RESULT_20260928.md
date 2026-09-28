# METH-158: 2,560 independent teacher continuations acquired

**Decision: input acquisition complete; route/training eligibility is a separate question.** The [frozen protocol](METH_158_INDEPENDENT_TENFOLD_TRAINING_DATA_PROTOCOL_20260928.md) selected 2,560 new chat prompts and 2,560 raw 128-token windows from 5,120 distinct H0 training rows. All raw windows were reread against the pinned source ID matrix with zero mismatches. Prior training/development rows and external excerpt fragments were excluded. The [manifest](meth158_independent_training_manifest.json) SHA-256 is `09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69`; its rejected candidate counts are 315 excluded rows and eight excerpt overlaps. This is a new training input pool, not an external quality set.

The pinned BF16 `Qwen/Qwen2.5-0.5B-Instruct` donor revision `7ae557604adf67be50417f59c2c2f167def9a775` greedily generated one response of at most 128 tokens per frozen prompt. The [sharded generator](../../../benchmarks/donor_adaptation/s1/meth158_independent_teacher_shard.py) was committed at `dd62e11`; the [resumable verifier](../../../benchmarks/donor_adaptation/s1/meth158_run_teacher_shards.py) at `c0a3b3a`, and the [merge](../../../benchmarks/donor_adaptation/s1/meth158_merge_teacher_shards.py) at `492eada`. A 64-prompt pilot preceded the remaining 38-shard run. Every one of the 40 shard files is committed and was verified against its prompt index, train row, prompt hash and response-ID hash. The [progress ledger](meth158_teacher_progress.json), SHA-256 `a777c1e09f1671ec3485c8645287a6fa7a489dd901d3715e47942f691f8b3b5e`, retains each shard hash and runtime. The [merged response file](meth158_independent_teacher_merged.json), SHA-256 `cdcb22a4273148dede97a17eac81345c4fc5f8e40f09cb18ecac4e35115e437c`, passed full 40-shard readback and ordered-row checks.

| Acquisition measure | Observation |
|---|---:|
| New chat responses / source-distinct raw windows | 2,560 / 2,560 |
| Response tokens, total / mean / range | 158,381 / 61.868 / 8–128 |
| EOS-terminated responses | 2,495 / 2,560 |
| Responses with an eight-gram repeated three times | 20 / 2,560 |
| Total greedy-generation runtime across 40 shards | 7,025.237 s (1.951 h) |
| Per-shard generation runtime | 147.594–192.468 s |
| Maximum per-shard allocated RTX 3060 memory / RSS | 1.014 GB / 2.354 GB |

The executed resumed command was `.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth158_run_teacher_shards.py --out-dir docs/research/NATIVE_EXPERT_SCALING_20260925/meth158_teacher_shards --progress docs/research/NATIVE_EXPERT_SCALING_20260925/meth158_teacher_progress.json --max-new-shards 40`. The merge command used `meth158_merge_teacher_shards.py --shard-dir <same shard directory> --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth158_independent_teacher_merged.json`. The [stdout log](meth158_teacher_runner.stdout.log) records all shard completions. No T4 was used.

The [matched training-input loader](../../../benchmarks/native_expert_scaling/meth158_training_inputs.py), committed at `418cce4`, was executed against the final merged SHA. It returned the pinned `(31250, 512)` source-ID matrix, 2,560 distinct chat rows and 2,560 ordered raw/chat draws, with updates exactly 1–2,560 and all source/prompt/continuation hashes verified. The loader is ready for a later matched protocol, but METH-159's route-support failure blocks use of the current route for that training run.

This artifact licenses only the separately frozen METH-159 support screen. It does not establish that the hash route balances the larger data, that tenfold experts learn useful capacity, or that the model retains donor quality.
