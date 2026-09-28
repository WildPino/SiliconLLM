# METH-155: matched learned-base E12800 training pass

**Decision: all frozen training gates pass; untouched quality pending.** A new candidate trained 1,280 shared base B rows and 12,800 routed residual rows on the exact METH-136 256 raw/chat draws. The comparator remains the frozen continued E1280 control. The base learning rate was `1e-5`, residual rate `2e-6`; all routes, shared A, gates, donor and objective were frozen. Initial BF16 logits exactly matched the centered E1280 teacher on eight bound prompts. The candidate kept one CPU combined B bank equal to base plus residual after every step, so each selected grandchild used one B gather. Its optimizer aggregated selected gradients by source child for the base and updated individual residuals separately.

The [training protocol](METH_155_MATCHED_SHARED_BASE_RESIDUAL_TRAINING_PROTOCOL_20260928.md) and [fresh quality protocol](METH_156_SHARED_BASE_FRESH_QUALITY_PROTOCOL_20260928.md) were committed at `fc32153` before this run. The [factorized expert/optimizer](../../../benchmarks/native_expert_scaling/meth155_factorized_shared_experts.py) and [trainer](../../../benchmarks/native_expert_scaling/meth155_matched_factorized_train.py) were committed at `e4c138f`; exact combined-bank step audits were committed at `ee52518`. An initial run without the protocol's per-step combined-bank audit was stopped before outcome and is excluded; its local progress file is preserved separately. The reported run started from the exact original BF16 source and used the corrected code.

The [result](meth155_matched_factorized_train_result.json) is SHA-256 `759d9044c7187c9bfd8ed3fbd6cc8b6d6f3be663c81bf9549b62200668dd4e2a`. The 4,404,019,224-byte local [combined BF16 bank](../../../benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth155_factorized_candidate_bf16.bin) is SHA-256 `c335d425c97fac5ec7284eea97ce3f3dc9462264ec79e7f46ce8598465d3e01d` and passed exact readback. Preserve this untracked large artifact for METH-156.

| Frozen gate or diagnostic | Observation | Limit |
|---|---:|---:|
| Updates; initial BF16 parity | 256; exact | 256; exact |
| Content grandchild coverage, minimum/layer | 8,673 | >=6,000 |
| Content maximum slot load / own-parent maximum | 1.127 | <=1.25 |
| Worst content grandchild share in hot parent | 17.94% | <=25% |
| BF16-distinct combined rows, minimum/layer | 11,820 | >=3,200 |
| Learned base rows and BF16-changed base rows, minimum/layer | 1,182 each | >=1,100; >0 |
| Exact combined-bank selected/untouched audits | 256/layer | 256/layer |
| Structural selections/layer | 66,516 | Exact table count |
| Combined ten-grandchild mean minus original B, RMS/layer | 8.38–10.59 × 10⁻⁵ | Descriptive |
| All-token maximum-to-mean load | 55.16–234.25 | Descriptive |

The learned common shift now has similar scale to the METH-136 continued control's `8.26–10.53 × 10⁻⁵` source-relative RMS. That checks the intended repair of METH-152's near-zero exported mean; it does **not** show a held-out benefit. The shared structural slot still takes 66,516 of 346,428 selections/layer (19.20%). Full all-token skew and CPU traffic remain reported rather than being excluded from cost. The same fixed METH-151 native route supports one combined B gather, but cold factor access, compact LUT arithmetic and accepted-token throughput in `engine.c` remain unmeasured.

The corrected run took 1,929.547 seconds on the local RTX 3060, within the 35-minute stop. Peak allocated GPU memory was 5.041 GB; RSS near the final update was 45.95 GB, below the 50 GiB cap. No T4 was used. METH-155 establishes a trainable, balanced 10×-expert construction with a learned common base, **not** useful additional capacity. Only the precommitted METH-156 source/fragment-disjoint prediction, generation, task and grounding gates can make that claim for this 0.5B rung; 10B/100B and full native speed remain open.
