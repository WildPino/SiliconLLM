# METH-165: data scale repairs median support but exposes another hot parent

**Decision: reject the fixed 138-tuple route for matched E1,280/E12,800 training.** The [prospective protocol](METH_165_EXPANDED_INDEPENDENT_SUPPORT_PROTOCOL_20260928.md) required every combined and new-only route gate. The additional independent data raise the minimum active content-row median from METH-159's 39 to **56**, passing >=50, but the combined worst hot-parent grandchild share is **32.32%**, failing <=25%. Do not train this route or claim useful 10× expert capacity.

The [manifest](meth165_expanded_training_manifest.json), SHA-256 `2875ac50a4de005b56ad6403c6bd30e358b87eb21d7dbb1918738957855068ba`, selected 1,280 new raw and 1,280 new chat rows from the pinned H0 ID matrix, excluding all 5,120 METH-158 selected rows and earlier train/dev/external fragments. The [teacher runner](../../../benchmarks/donor_adaptation/s1/meth165_expanded_teacher.py) completed 20 independently checked 64-prompt shards. The [merge](meth165_expanded_teacher_merged.json), SHA-256 `9843b8d98d0bbe322cbbf2567f72d5c3b09a34bfa774b90785c22a9079076ed9`, contains 78,291 continuation tokens, 1,250/1,280 EOS terminations and three triple repeated 8-grams. Teacher generation on the local RTX 3060 took 3,148.391 seconds, peak allocated GPU memory 1,014,466,560 bytes and final RSS 2,306,527,232 bytes, within the 70-minute/10.5-GiB/20-GiB stops.

The [route runner](../../../benchmarks/native_expert_scaling/meth165_expanded_route_support.py) required BF16 teacher/control exact parity, the fixed METH-162 table SHA-256 `dca94f6ea833be5446b497132595ff5dc80d94a414f18a8a2bad116764fac4e1` and METH-164 pass SHA-256 `1459921ea90fc77395353d4bafcc261e0413d7b34dfd5049acd35e7bbaf2ad63`. It forwarded the new 1,280 raw/chat pairs, then summed each new 12,800-slot all/content histogram with the saved exact METH-162 histograms for the original 2,560 pairs. The [machine result](meth165_expanded_route_support_result.json), SHA-256 `b1c9bf6d0c3d74c7a282c657f069735299cb7cfaa0f92e52175669d475e5058c`, records all 24 layers for new raw, new chat, new combined and all 3,840 pairs. Input count is 162,560 new raw tokens plus 280,206 new chat tokens, **1,330,096 combined**. Route forward took 482.531 seconds, peak allocated GPU memory 2,945,090,560 bytes and final RSS 4,682,764,288 bytes.

| Gate group | Minimum occupied content rows | Minimum active median | Maximum under-32 fraction | Maximum load ratio | Worst hot-parent share |
|---|---:|---:|---:|---:|---:|
| New raw only | 9,661 | 11 | 73.13% | 1.083 | 18.60% |
| New chat only | 9,710 | 14 | 69.88% | 1.080 | 23.55% |
| New pairs together | 10,417 | 22 | 61.38% | 1.053 | 19.60% |
| All 3,840 pairs | **10,999** | **56** | **41.03%** | **1.026** | **32.32% — fail** |

The first four combined support gates pass, as do both new-only raw/chat load gates. The failing layer is 20, source child 656, content local 4: **85 of 263** content selections. The original METH-158 portion supplies 56 of 175, new raw 0 of 23 and new chat 29 of 65. Each portion stays below the frozen 250-selection threshold for a "hot parent"; the pooled parent crosses it. This is why passing the new-only and original in-sample hot-parent screens did not guarantee a pooled pass. A postfailure trace inspection attributed 38 of the original 56 selected occurrences to two Qwen chat role delimiters at positions 152 and 154, each seen 19 times. Those exact causal tuples recur 12 times each in the new chat inputs, for 31 combined occurrences each, still below the METH-149/162 recurrence threshold 32. This attribution is exploratory and cannot retroactively change the frozen result.

No matched long training or new external quality audit followed. PG19 shard 9 remains untouched for the future quality test. The next route design should handle chat-template structure as a semantic class and be checked on exact stored source-child traces, independently sourced prompts and native C cost before training. Simply adding more independent contexts has proven enough for the median support gate at this 0.5B donor scale, but not for the fixed route's load gate.

Exact commands:

```powershell
.venv\Scripts\python.exe benchmarks\donor_adaptation\s1\meth165_expanded_training_manifest.py --route-sha 1459921ea90fc77395353d4bafcc261e0413d7b34dfd5049acd35e7bbaf2ad63 --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth165_expanded_training_manifest.json
.venv\Scripts\python.exe benchmarks\donor_adaptation\s1\meth165_expanded_teacher.py --shards docs\research\NATIVE_EXPERT_SCALING_20260925\meth165_teacher_shards --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth165_expanded_teacher_merged.json
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth165_expanded_route_support.py --teacher-sha 9843b8d98d0bbe322cbbf2567f72d5c3b09a34bfa774b90785c22a9079076ed9 --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth165_expanded_route_support_result.json
```
