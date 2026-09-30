# METH-179: trained content-child routing has no robust functional alignment

**Decision: the mechanism diagnostic fails both preregistered conditions.** On the 24 previously consumed METH-173 shard-12 documents, the exact trained content route has pooled BPB **1.06766915**. The mean of its eight nonzero cyclic content-child rotations is 0.00001519 BPB *better* (mean rotation minus exact = -0.00001519). The exact route beats only three of the eight rotations by pooled BPB, one by just 0.00000003 BPB, and the paired 10,000-draw source bootstrap 5th percentile for mean-rotation-minus-exact is **-0.00014455**, below the required positive value. Only 9/24 individual documents favor the exact route over the mean rotation. These results offer no robust evidence that the learned child-specific B rows are aligned with the fixed content route.

The [protocol](METH_179_CONTENT_ROUTE_FUNCTION_PROTOCOL_20260930.md) was committed at `fe7c7f9` before inference. The [ablation scorer](../../../benchmarks/native_expert_scaling/meth179_content_route_function.py) was committed at `d17529a` before execution. The [machine result](meth179_content_route_function_result.json) has SHA-256 `32bc1760dbdb4ce3efcaa98fa8d5f266fa773cc22d9f434369bc75ae772fef4a`. It binds the METH-173 manifest SHA-256 `6ebfc36fd452f2a20cfcc9e8210505e36b725ef95992949655c19f3c94eaee20`, the METH-175 candidate result and exact 4.404 GB BF16 bank, and the pinned tokenizer. Eight-prompt initial BF16 parity was exact; the trained bank's readback was exact. The route intervention keeps the model, B bank, source-child IDs, four selections and local structural zero unchanged. All nine arms used the same 24×1,024 token IDs, causal windows and absolute positions.

| Content route | Pooled BPB | Exact minus rotation BPB |
| --- | ---: | ---: |
| Exact, shift 0 | 1.06766915 | — |
| Shift 1 | 1.06756728 | +0.00010187 |
| Shift 2 | 1.06763308 | +0.00003606 |
| Shift 3 | 1.06766917 | -0.00000003 |
| Shift 4 | 1.06755866 | +0.00011049 |
| Shift 5 | 1.06776180 | -0.00009265 |
| Shift 6 | 1.06762605 | +0.00004310 |
| Shift 7 | 1.06778773 | -0.00011858 |
| Shift 8 | 1.06762786 | +0.00004129 |

An independent vectorized bootstrap from the saved per-document nats and decoded byte counts reproduced the exact 5th percentile. The local RTX 3060 run took 281.578 seconds, with 4,673,313,792 bytes peak allocated GPU memory and 11,276,984,320 bytes final RSS, below the frozen caps. No T4 was used.

This is a postfailure mechanism test on a cohort used for router validation, so it cannot upgrade or replace METH-176's independent quality verdict. Combined with METH-178's 2.318% route-weighted within-parent B variation, it gives no reason to spend another long run merely extending the same fixed hash-route/shared-base training recipe. The next candidate needs a different training signal or learned route that makes child identity predictive of distinct corrections, selected using training and development data alone and judged on genuinely untouched sources. The CPU LUT relative scaling failure at E128,000 remains a separate constraint.
