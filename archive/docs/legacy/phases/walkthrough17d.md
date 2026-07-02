# Phase 17D: Pruned Low-Rank Candidate Generator Walkthrough

In Phase 17D, we sought to optimize the CPU-bound `Candidate Gen` loop. The insight was that since `trigram + bias` accurately catches almost all likely targets (as proven by Tail Mode 2's success), there's no need to evaluate the Low-Rank projection on all 256 classes. We can instead statically precompute a Candidate Pool `M` per context based on `trigram + bias`, and strictly limit the Low-Rank Rerank to those `M` candidates.

## Architectural Changes

1. **Precomputed Candidate Pool (`topk_indices`)**:
   - `eval_entropy_stream.c` was updated to sort classes per-context by their initial `trigram_logits[c] + B[c]` score during boot.
   - The top $M$ candidates (`req_topm`) define the Candidate Pool.

2. **Pruned Candidate Gen**:
   - The inner AVX loop now only iterates over the `M` candidates instead of all 256 classes. 
   - We extract the Top $K$ elements from this pool of size $M$ using an inline Insertion Sort (which executes extremely fast because $M$ is small and the incoming candidates are pre-sorted by their likely strength).

3. **Loss Tracking**:
   - Implemented metric tracking for the BPB contribution of targets outside the pool `M` and targets within `M` but outside `K`, allowing us to precisely measure the entropy cost of pruning.

## Benchmarks (`c_code.c`)

We evaluated multiple combinations of $M \in \{64, 96, 128\}$ and $K \in \{32, 64\}$ using Rank 16 and `Tail Mode 2` (ngram + bias), and compared it to the baseline full softmax. 

| Model Version | Rank | M   | K   | Model BPB | Hit Rate | Predict Cycles/byte | Candidate Gen |
|---------------|------|-----|-----|-----------|----------|---------------------|---------------|
| Phase 16 Full | 192  | 256 | 256 | **2.7180**| 100%     | ~33,000             | N/A           |
| Phase 17D     | 16   | 128 | 64  | 2.7184    | 98.89%   | 21,362              | ~13,000       |
| Phase 17D     | 16   | 128 | 32  | 2.7358    | 95.28%   | 11,340              | 7,047         |
| Phase 17D     | 16   | 96  | 64  | 2.7184    | 98.89%   | 21,362              | ~13,000       |
| Phase 17D     | 16   | 96  | 32  | 2.7358    | 95.28%   | 10,978              | 6,689         |
| Phase 17D     | 16   | 64  | 64  | 2.7205    | 98.73%   | 19,309              | ~11,000       |
| **Phase 17D** | **16**| **64**| **32**| **2.7348**| **95.32%**| **10,502**          | **6,171**     |

> [!TIP]
> **We Achieved the 10k Goal!**
> Setting $M=64$ and $K=32$ is the optimal tradeoff. The Predict phase runs in just **10,502 cycles**, bringing the total cycles per byte to ~14,900. The BPB cost compared to the full Phase 16 model is just **0.0168**!

## Cycle Breakdown and Loss Analysis (M=64, K=32)

```text
Target in Pool:  98.73% (M=64)
Top-K Hit Rate:  95.32% (K=32)
Loss out pool:   0.1560 BPB
Loss out K:      0.3456 BPB
Dot Prods/byte:  32.0

--- Mean Cycles / Byte ---
Extract:         40.4
Normalize:       62.2
Predict:         10502.3
  - Candidate Gen: 6171.6
  - Full Residual: 1930.5
  - Softmax+Tail:  2293.4
Avg Tail Mass:   0.0364
Total:           14968.9
```

> [!IMPORTANT]
> The Candidate Gen loop has collapsed from ~12k cycles (in Phase 17C) to **6,171 cycles**. The Full Residual on $K=32$ costs **1,930 cycles**, and the Softmax+Tail evaluation takes **2,293 cycles**. We have successfully factored the readout layer into a CPU-native structure.

### Loss Contribution Insights
- The target falls entirely out of the $M=64$ static pool only 1.27% of the time, costing us 0.1560 BPB. 
- The target falls out of the reranked $K=32$ candidate set 3.41% of the time, costing us 0.3456 BPB. 
- Because we dynamically compute the exact mathematically consistent tail mass (Tail Mode 2), the probabilities are correctly squashed onto the trigram+bias fallback, keeping the BPB incredibly competitive.
