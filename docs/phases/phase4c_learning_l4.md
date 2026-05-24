# Phase 4.C: L4 Evolutionary Population (SIMD)

We implemented an aggressive genetic algorithm natively inside the CPU pipeline. By transposing the grid memory (`__m256i state[GRID_SIZE]`), we successfully calculated the wave dynamics for 32 independent engines simultaneously using AVX2 bitwise logic (`_mm256_and_si256`, `_mm256_blendv_epi8`, etc.).

## 1. Throughput Benchmarks (The Power of L1)
We measured the cycle cost for 1 tick (4 waves + 1 bridge + injection) across 32 engines.

- **1024 Cells (L2 Cache Bound)**: 37,540 cycles/tick total $\approx$ **1173 cycles** per virtual engine.
- **256 Cells (L1 Cache Bound)**: 9,398 cycles/tick total $\approx$ **293 cycles** per virtual engine.

**Analysis**: The user's prediction was perfectly accurate. Spilling into L2 cache slows the pipeline significantly (from ~300 to ~1100 cycles). However, if we restrict the grid to 256 cells (fitting entirely within the 32KB L1 Data Cache), the throughput is astounding. 293 cycles per engine means the wave propagates at **~0.28 cycles/elem** — identical to the pure scalar baseline from Phase 1, despite the fact that we are now computing 4 complex rules and 3 masked blends simultaneously. The AVX2 vector units absorb all the arithmetic overhead perfectly.

## 2. Learning Results (Grid = 256 cells)
Every 1000 ticks, we ranked the 32 engines. The bottom 8 were overwritten by the top 8, with 2 random rule mutations applied to maintain diversity.

| Task | Final Accuracy | Conclusion |
| :--- | :--- | :--- |
| **XOR-2** | 53.6% | Slight signal above noise, but failed to converge |
| **Period-7** | 85.7% | Hit optimal baseline (always predicting 0) |
| **Echo-5** | 53.9% | Slight signal above noise, but failed to converge |

### L4 Analysis
Despite running 32 engines in parallel and applying an evolutionary algorithm, the system still failed to converge on the target sequences. 

The physical substrate is blazingly fast and processes data perfectly, but the "programming language" of the substrate (selecting between 4 discrete boolean rules per zone) creates a highly deceptive, non-convex fitness landscape. 
Evolutionary algorithms struggle here because there are no "building blocks" of partial success: flipping a rule in zone $Z$ drastically alters the chaotic wave interference pattern, destroying any previously learned structure. The system lacks the "smoothness" required for learning.
