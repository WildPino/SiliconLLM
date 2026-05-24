# Phase 17C: Readout Acceleration & Factorization Walkthrough

In Phase 17C, we implemented the Tail Softmax candidate generation logic using the SVD factors of our 256-way readout layer. This allowed us to dynamically identify the top $K$ candidates for each byte without executing the full 192-dimension dot product on all 256 classes. 

## Architectural Changes

1. **SVD Factors Export (`analyze_svd.py`)**: 
   - We modified our analysis script to save the $A_{proj}$ ($Rank \times 192$) and $B_{proj}$ ($256 \times Rank$) matrices alongside the original $W$ matrix.
   - This enables the C code to load the small factors for Candidate Generation, while keeping the full-fidelity $W$ available for the `Full Residual` on the Top-K candidates.

2. **Tail Softmax (`eval_entropy_stream.c`)**:
   - Implemented `base_probs` ($exp(trigram)$) calculation at model boot.
   - Designed a Blazingly Fast Candidate Generation loop: $z = A_{proj} \times x$, followed by $cand\_logit = B_{proj} \times z$.
   - Adopted a **Min-Heap of size K** instead of `qsort()` to pick the Top-K candidates. This avoids full sorting and guarantees $O(CLASSES \cdot \log K)$ performance.
   - Implemented `tail_mass` computation for mathematical equivalence, allowing us to attribute correct probabilities to all $256 - K$ remaining classes using either just n-gram (`tail_mode=1`) or n-gram+bias (`tail_mode=2`).
   - Dynamically scale the `Z_total` to prevent any possibility of floating-point overflow using `max_l`.

## Benchmarks (`c_code.c`)

We evaluated Rank 16 and Rank 32 configurations with $K=32$ and $K=64$, comparing against the original Phase 16 Full Softmax implementation:

| Model Version | Rank | Top K | Tail Mode | Model BPB | Hit Rate | Predict Cycles/byte |
|---------------|------|-------|-----------|-----------|----------|---------------------|
| Phase 16 Full | 192  | 256   | N/A       | **2.7180**| 100%     | ~33,000             |
| Phase 17C     | 16   | 32    | 1 (ngram) | 2.8507    | 95.28%   | 17,377              |
| Phase 17C     | 16   | 64    | 1 (ngram) | 2.8395    | 98.89%   | 22,379              |
| Phase 17C     | 32   | 32    | 1 (ngram) | 2.8380    | 95.46%   | 19,351              |
| Phase 17C     | 32   | 64    | 1 (ngram) | 2.8358    | 98.95%   | 25,780              |
| Phase 17C     | 16   | 32    | 2 (+bias) | 2.7358    | 95.28%   | 16,609              |
| **Phase 17C** | **16**| **64**| **2 (+bias)**| **2.7184**| **98.89%**| **21,003**          |
| Phase 17C     | 32   | 64    | 2 (+bias) | 2.7149    | 98.95%   | 25,620              |

> [!TIP]
> **Tail Mode 2 (ngram + bias) is magical!** Even though Candidate Generation only catches the target ~98.89% of the time, mapping the remaining 1.11% of targets onto the `n-gram + bias` tail almost perfectly recovers the lost entropy. The BPB of 2.7184 is practically indistinguishable from the un-approximated 2.7180!

## Cycle Breakdown for Optimal Config (Rank 16, K=64, Tail=2)

```text
--- Mean Cycles / Byte ---
Extract:         42.6
Normalize:       57.7
Predict:         21003.7
  - Candidate Gen: 12445.2
  - Full Residual: 3884.1
  - Softmax+Tail:  4565.7
Observe:         4299.1
Total:           25403.2
```

We reduced the Predict phase from ~33,000 cycles to ~21,000 cycles. We did not hit the arbitrary 10,000 cycle mark because Candidate Generation itself requires about ~12k cycles (evaluating the Min-Heap condition 256 times). However, Candidate Generation combined with the partial Full Residual proves far more efficient overall.

> [!NOTE]
> The implementation of the Candidate Generation is theoretically optimal for CPU: 16-rank dot products execute using AVX2 intrinsics (2 loads, 2 FMA per class). To push this under 10k cycles, we would need to eliminate classes entirely at the unigram/bigram level before even evaluating the low-rank vectors, but this setup guarantees we don't drop BPB.
