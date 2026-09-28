# METH-145: hash-routed E12800 trains, but repeated-template traffic fails load gates

**Decision: reject this trained E12800 bank for quality promotion.** The candidate completes all 256 B-only updates on the identical METH-136 draw sequence and exports a readback-verified 4,404,019,224-byte BF16 bank. Exact initial logits pass. Every layer selects at least 9,449 grandchildren and has at least 11,830 BF16-distinct rows after mean centering. Yet 20 of 24 layers exceed the frozen 1.25× maximum/mean load ratio relative to the already trained E1280 control, reaching **2.875×**. All 24 layers exceed the 25% hot-parent share limit; their worst shares range from **93.14% to 100%**. The METH-146 new-source quality test must remain unopened.

The [protocol](METH_145_MATCHED_HASH_E12800_TRAINING_PROTOCOL_20260928.md) was committed at `339304f` and the [training implementation](../../../benchmarks/native_expert_scaling/meth145_matched_hash_train.py) at `f05f4e8` before execution. It reuses the completed METH-136 E1280 control result SHA-256 `068c19a2fa34b95cca2a357674a1f7e1185ea04095593d54eec1e892ecb29be6` and control bank SHA-256 `d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299`, with the same 256 deterministic raw/chat draws, objective and selected-row optimizer. The unchanged METH-142 token hash chooses the grandchild inside each full-model forward. The [result](meth145_matched_hash_train_result.json), SHA-256 `14687c03bf16b552afbdfef2e402e3acc54ff4eea8dd50e3a2140f3e8f2800e7`, stores every training-time grandchild count and both failed gates. The candidate bank SHA-256 is `881a43fed3b1132a75bb4a4cfce2ada64cc13c40993248412bbadf52adf64c4f` and stays a rejected diagnostic artifact.

| Frozen gate | Result |
|---|---|
| Initial eight-prompt BF16 logit parity | Exact pass |
| Selected grandchildren per layer >=6,400 | 9,449–12,092; pass |
| BF16-distinct grandchildren per layer >=3,200 | 11,830–12,780; pass |
| Candidate/control slot skew <=1.25× in every layer | 20/24 fail; worst 2.875× |
| Hot-parent grandchild share <=25% in every layer | 24/24 fail; worst 100% |
| Candidate bank byte readback | Exact pass |

The candidate arm took 1,115.203 seconds, with 33.925 GB reported process RSS at arm completion and 5.041 GB peak allocated GPU memory on the local RTX 3060. Including binding and export, the run took 1,129.625 seconds. No T4 was used. Training used 476.774 GB cumulative forward selected-B transfer; this is transfer volume across all forwards, not resident memory or native serving traffic.

A direct check of the bound METH-136 training loader shows that **all 256 chat sequences begin with the same 55 token IDs**; their first varying token is at position 55. In the saved candidate counts, multiple hot grandchild slots receive exactly 256 or 512 selections, consistent with these repeated causal prefix states. This is a mechanistic lead, not a source-attributed route trace. A deterministic token/history hash cannot split identical prefixes by content available at those positions. METH-142/143 passed on more varied raw and document inputs, so their fresh-source traffic pass did not cover this template-heavy training mixture. Next: measure load by raw, fixed-template and variable-chat positions with the same frozen hash and then test a template-aware allocation rule or more varied training mixture under a new protocol. Do not waive the failed METH-145 gate or infer quality from distinct bytes.
