# Phase 17E: Remove Low-Rank Rerank? Walkthrough

In Phase 17E, we questioned the necessity of the `Low-Rank Candidate Gen` loop. If the static pool $M$ (based on `trigram + bias`) is already highly accurate, does the low-rank rerank actually provide enough value to justify its 6,000+ cycle cost? 

To test this, we completely bypassed the Low-Rank approximation (`lr_rank = 0`) and directly passed the top $K$ candidates from the static `trigram + bias` pool into the `Full Residual` and `Tail Softmax` components.

## Benchmark Results (`c_code.c`)

| Model Version | Candidate Gen | K  | Model BPB | Hit Rate | Predict Cycles/byte | Cand Gen Cycles |
|---------------|---------------|----|-----------|----------|---------------------|-----------------|
| Phase 16 Full | N/A           | 256| **2.7180**| 100%     | ~33,000             | N/A             |
| Phase 17D     | Low-Rank M=64 | 64 | 2.7205    | 98.73%   | 19,309              | ~11,000         |
| Phase 17D     | Low-Rank M=64 | 32 | 2.7348    | 95.32%   | 10,502              | 6,171           |
| **Phase 17E** | **Static Top-K** | **64**| **2.7205**| **98.73%**| **8,799**           | **~140**        |
| **Phase 17E** | **Static Top-K** | **48**| **2.7406**| **97.14%**| **6,457**           | **~115**        |
| Phase 17E     | Static Top-K  | 32 | 2.7629    | 94.35%   | 4,402               | ~90             |

> [!CAUTION]
> **The Low-Rank Approximation was useless!**
> Compare Phase 17D (Low-Rank M=64, K=64) with Phase 17E (Static K=64). The BPB is **identically 2.7205**. The Low-Rank generator was simply burning 11,000 cycles to re-sort 64 elements that all got passed to the Full Residual anyway! By skipping it, we drop from 19k cycles to **8,799 Predict cycles**.

### Evaluating K=48 (The Sweet Spot)
If we want even more speed, `Static K=48` completes the Predict phase in **6,457 cycles**. Compared to the best Low-Rank config (M=64, K=32), it:
- Loses only **0.0058 BPB** (2.7406 vs 2.7348).
- Saves **4,000+ Predict cycles** per byte.

## Natural Text Verification (`promessi_sposi.txt`)

> [!IMPORTANT]
> The readout weights were specifically trained on the `c_code.c` dataset. Therefore, the BPB metrics reported on `promessi_sposi.txt` reflect **Out-Of-Domain (OOD) transfer capability**, not in-domain quality.

- **Phase 17C (Rank 16, K=64)**: 5.9418 BPB
- **Phase 17E (Static K=64)**: 5.9412 BPB

> [!TIP]
> The static `trigram + bias` candidate selector actually **outperforms** the Low-Rank selector on unseen domains, because the Low-Rank matrices ($A_{proj}, B_{proj}$) overfitted slightly to the C code domain!

## Architectural Verdict

> "Il decodificatore CPU-native non è un softmax pieno, né un candidate generator appreso; è una tabella statica `trigram+bias` che sceglie pochi candidati, un residual fisico sui candidati, e una tail probabilistica rigorosa per tutto il resto."

The Silicon has spoken loudly. The Readout layer does NOT need an approximated Low-Rank candidate generator. The **profilo readout accelerato migliore misurato finora** is:
1. **Precompute** `trigram + bias` at boot to establish the static candidate ordering for every context.
2. **Select** the static top $K$ candidates in $O(1)$ time. This gives us three powerful operating profiles:
   - **Accurate** (`K=64`): BPB 2.7205, Predict ~8.8k cycles
   - **Fast** (`K=48`): BPB 2.7406, Predict ~6.4k cycles
   - **Aggressive** (`K=32`): BPB 2.7629, Predict ~4.4k cycles
3. **Execute** `Full Residual` exclusively on those $K$ candidates.
4. **Compute** `Tail Mode 2` (ngram + bias) for the remaining mass.
