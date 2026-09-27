# METH-106–109: long-chat child retention passes fresh development

**Decision.** METH-107 keeps 1,084–1,152 meaningfully distinct child slots per layer and passes all six frozen METH-109 development gates against the BF16+E128 teacher. It is eligible for the separately frozen METH-110–112 external audit, but is not yet quality promoted. The METH-105 semantic failure remains binding until a new blind external review passes. No child-factor LUT or full native inference result exists for this checkpoint.

The [protocol](METH_106_LONG_CHAT_RETENTION_PROTOCOL_20260927.md) was fixed before the experiment. [METH-106](meth106_long_e128_teacher_chat_result.json) re-generated the pinned 256 METH-43 training prompts with the frozen BF16+METH-56 E128 teacher, up to 128 greedy continuation tokens. Four independently saved 64-row shards were validated and merged in exact order. Sixty-eight responses exceed 64 tokens; 255/256 terminate with EOS. Mean response length is 53.55 tokens, maximum 128. This extends assistant-target supervision beyond the earlier 64-token teacher replies without using external prompts.

The [METH-107 trainer](../../../benchmarks/donor_adaptation/s1/meth107_long_chat_child_retention.py) resumes the rejected METH-99 E1280 candidate and makes 256 B-only updates, pairing each long chat row once with an excluded-source raw draw. The local checkpoint SHA-256 is `15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`; its [training ledger](meth107_long_chat_child_retention_result.json) records 1,084–1,152 sibling-distinct slots per layer and 1,183–1,277 visited slots. The run took 267.14 s, peaked at 7.779 GB allocated GPU memory and used 5.373 GB RSS, within the frozen limits. The 1.774 GB checkpoint remains a local reproducible artifact; the runner, seed, hashes and draw ledger are committed.

The [METH-108 manifest](meth108_long_chat_dev_manifest.json) contains 8 code, 8 prose and 8 technical documents with exact source, text and token hashes. Existing PG19 test/validation candidates were exhausted after prior exclusions, and the sampled PG19 train fragments overlapped the calibration base. Prose candidates therefore came from disjoint 1-MiB windows of the pinned Simple English Wikipedia concatenation. They are distinct byte windows of one corpus, not independently curated articles. [METH-109](meth109_long_chat_development_result.json) scores intact BF16 donor, BF16+E128 and BF16+E1280 on identical IDs:

| Fresh development gate | E128 | E1280 | E1280−E128 |
|---|---:|---:|---:|
| Pooled document BPB | 1.075889 | 1.075398 | −0.000490 |
| Donor prompt top-1 agreement | 96.409% | 95.851% | −0.558 point |
| Greedy EOS / 24 | 23 | 24 | +1 |

Document BPB improves in all three categories (deltas −0.000123 to −0.000967), and all category prompt agreement losses stay below one point. All six predeclared gates pass. The 255.66 s evaluation peaked at 4.218 GB allocated GPU memory and 3.328 GB RSS. These are development observations on 24 excerpts, not a semantic or native performance verdict. The separately [frozen external protocol](METH_110_112_LONG_CHAT_EXTERNAL_PROTOCOL_20260927.md) uses new prose provenance and a blind semantic gate.
