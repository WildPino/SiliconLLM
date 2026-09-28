# METH-152: matched shared-structure E12800 training pass

**Decision: all frozen training gates pass; fresh quality remains untested.** A fresh E12800 B bank with one shared structural and nine hashed content slots per E1280 source child completed the exact METH-136 256 raw/chat updates. The comparator is the already frozen METH-136 continued E1280 arm on the same draws and objective. Source B rows were cloned exactly in BF16 at initialization, and the eight bound prompts had exact initial teacher/candidate logits. Parent/child routing, gates, shared A, the 75-tuple structural table and hash constants remained frozen; only selected CPU-master B rows trained.

The [protocol](METH_152_MATCHED_SHARED_E12800_TRAINING_PROTOCOL_20260928.md) was committed at `1950d4c` and the [training/accounting code](../../../benchmarks/native_expert_scaling/meth152_matched_shared_train.py) at `b2a5224`, before this run. The [result](meth152_matched_shared_train_result.json) is SHA-256 `6beb41e673f70ca9cbd496b59ba9572d7ef646d7950ec93d88609d007ac6901c`. The 4,404,019,224-byte BF16 [bank](../../../benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth152_shared_candidate_bf16.bin) is SHA-256 `17196e7a320a6887ba4a04283e673f3b53b961a0062cb15b05e75d558df58a9e` and passed exact readback. It remains a local untracked large artifact; preserve it for the quality audit.

| Frozen gate or diagnostic | Observation | Limit |
|---|---:|---:|
| Completed updates and initial BF16 parity | 256; exact | 256; exact |
| Content slot coverage, minimum/layer | 8,673 | >=6,000 |
| Nine-way content maximum load / own parent's maximum | 1.104 | <=1.25 |
| Worst grandchild share for a parent with >=250 content selections | 19.17% | <=25% |
| BF16-distinct B rows, minimum/layer | 11,820 | >=3,200 |
| Structural selections/layer | 66,516, exact table-derived count | Exact |
| Total selections/layer | 346,428 | Exact |
| All-token maximum-to-mean load, across layers | 55.31–234.14 | Descriptive |
| All-token load relative to continued E1280 control | 1.57–8.57× | Descriptive |

The structural route accounts for 19.20% of all selections per layer and deliberately concentrates recurring chat template contexts in slot zero. The content gates use the 279,912 remaining selections per layer and were captured after each changing-weight forward, before checkpoint backward recomputation. Complete all-token, structural and content histograms are in the result; their sums equal the backward collector counts exactly. This split does not make the all-token skew or CPU traffic disappear. The METH-151 native result measured route-only hit and miss costs, not training-time factor traffic or full generation cost.

The run finished in 1,107.859 seconds on the local RTX 3060; peak allocated GPU memory was 5.041 GB, with process RSS about 34.2 GB during training. No T4 was used. The candidate is **not yet a useful-expert or quality pass**: distinct B bytes and balanced training routes do not prove better held-out prediction, donor retention or grounding. The separately frozen [METH-153 protocol](METH_153_SHARED_E12800_FRESH_QUALITY_PROTOCOL_20260928.md), committed before this result at `2e38326`, must now select source/fragment-disjoint documents, run the fresh prediction gate, and only then proceed to generation and blind grounding if it passes. This remains a 0.5B-donor 10×-expert rung, not evidence for 10B/100B transfer or full native throughput.
