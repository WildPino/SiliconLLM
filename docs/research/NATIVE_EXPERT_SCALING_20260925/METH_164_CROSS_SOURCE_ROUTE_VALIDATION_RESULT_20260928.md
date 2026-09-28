# METH-164: 138-tuple route passes a different source and two lengths

**Decision:** the METH-162 table passes the frozen [cross-source route protocol](METH_164_CROSS_SOURCE_ROUTE_VALIDATION_PROTOCOL_20260928.md), so the conditional METH-165 independent training-support expansion may proceed. This is an untrained routing screen. It does not repair the METH-159 median-support failure, establish specialist usefulness or supply native C cost.

The [manifest](meth164_cross_source_route_manifest.json), SHA-256 `d8420488104990ab63338721837c4f100135c45a07e787f2708eb741dfb63c9b`, selected 128 chat, 128 raw and 24 document rows without row overlap from PG19 train shard 10, SHA-256 `ed08834a0a5d4b63af11f94595561fb6988693ff3a75eacbf6e1a934cec39f78`. The [teacher runner](../../../benchmarks/donor_adaptation/s1/meth164_cross_source_teacher.py) produced four verified 32-prompt shards and a [merge](meth164_cross_source_teacher_merged.json), SHA-256 `73bd039595682a268244bfc1d3df8c0e9bb005f5afe0cdc3c43315b90e131125`. Its 128 responses contain 8,001 tokens, 125 EOS terminations and one triple repeated 8-gram. Generation took 318.532 seconds on the RTX 3060, with 1,014,440,448 peak allocated GPU bytes and 2,293,280,768 final RSS bytes.

The [route runner](../../../benchmarks/native_expert_scaling/meth164_cross_source_route_screen.py) checked eight-prompt BF16 teacher/control parity, used the same actual E1,280 source-child choices for both tables and compared 75 versus 138 shared tuples. Its [machine result](meth164_cross_source_route_result.json) SHA-256 is `1459921ea90fc77395353d4bafcc261e0413d7b34dfd5049acd35e7bbaf2ad63`. The fixed table added shared handling to 112 chat, 16 raw, 30 document-width-128 and 8 document-width-512 input positions. Every cell/layer has hot parents, four selections/token and exact grandchild/source-child identity.

| Cell | Input tokens | Candidate minimum content coverage | Candidate maximum load ratio | Candidate worst hot-parent share |
|---|---:|---:|---:|---:|
| Chat | 28,193 | 5,523 | 1.2063 | 19.03% |
| Raw | 16,256 | 5,529 | 1.2462 | 18.18% |
| Documents, 128-token windows | 24,576 | 6,193 | 1.2167 | 18.56% |
| Documents, 512-token windows | 24,576 | 6,156 | 1.1619 | 20.07% |

All preregistered candidate gates pass: minimum content coverage >=4,000, maximum ratio <=1.25, worst hot-parent share <=25%, hot parents present, nonzero chat shared traffic and BF16 parity. The original 75-tuple route also passed these same new-source cells (worst hot shares 19.03%, 18.18%, 21.18%, 20.07% respectively). Thus this test supports **no obvious regression** from the larger table on a separate text source and two lengths; it does not independently recreate and cure METH-159's training-template hotspot. Route forward took 120.812 seconds, 2,945,090,560 peak allocated GPU bytes and 4,634,300,416 final RSS bytes. PG19 shard 9 was not read and remains available for untouched quality.

Exact commands:

```powershell
.venv\Scripts\python.exe benchmarks\donor_adaptation\s1\meth164_cross_source_route_manifest.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth164_cross_source_route_manifest.json
.venv\Scripts\python.exe benchmarks\donor_adaptation\s1\meth164_cross_source_teacher.py --shards docs\research\NATIVE_EXPERT_SCALING_20260925\meth164_teacher_shards --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth164_cross_source_teacher_merged.json
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth164_cross_source_route_screen.py --teacher-sha 73bd039595682a268244bfc1d3df8c0e9bb005f5afe0cdc3c43315b90e131125 --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth164_cross_source_route_result.json
```
