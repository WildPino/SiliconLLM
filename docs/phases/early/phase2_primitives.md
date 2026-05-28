# Phase 2: CPU-Native Cognitive Primitives

In Phase 2, we moved away from measuring raw memory latency and investigated the optimal "algebra" that maximizes computational throughput on a modern superscalar CPU pipeline (AMD Ryzen 5 3600X, Zen 2). We tested several candidates for the fundamental cognitive operations of the SiliconLLM model.

## 1. AVX2 XNOR + POPCOUNT
**Goal**: Test the viability of a discretized dot product. Since Zen 2 lacks `VPOPCNTDQ`, we used the highly optimized `vpshufb` 4-bit nibble lookup method across 256-bit vectors.
* **Throughput**: ~9.60 cycles per 256-bit operation.
* **Analysis**: While 9.6 cycles for 256 binary multiplications-and-accumulations is vastly superior to sequential scalar operations, the `vpshufb` sequence requires several instructions (shifts, masks, shuffles, adds, and sad). It is viable, but it's not a "free" single-cycle operation on Zen 2.

## 2. Hadamard / Butterfly Mixing
**Goal**: Test how quickly we can mix states within local memory tiles without external memory lookups, using permutations and arithmetic.
* **Throughput**: ~2.34 cycles per 256-bit vector (9.38 cycles for a block of four 256-bit vectors).
* **Analysis**: Butterfly mixing is incredibly fast. Mixing 1024 bits of state takes under 10 cycles. This strongly suggests that mixing information spatially through arithmetic butterfly networks is far cheaper than looking up keys in a sparse memory cache.

## 3. Local Cellular Propagation (1D Grid)
**Goal**: Measure the throughput of contiguous local state updates (Cellular Automata style).
* **Step 1 (XOR Baseline)**: 0.27 cycles / element
* **Step 2 (Mix +/-)**: 0.33 cycles / element
* **Step 3 (Quasi-Rule 30)**: 0.28 cycles / element
* **Step 4 (SIMD Tile 8-el)**: 0.50 cycles / element
* **Analysis**: This is the most profound discovery. The superscalar CPU is so good at Instruction-Level Parallelism (ILP) and prefetching contiguous memory that a simple scalar loop computing `new[i] = (state[i-1] & state[i+1]) ^ state[i]` executes at ~3.5 elements per clock cycle. The naive SIMD version with complex permutations was actually slower per-element (0.50 cycles) than the unrolled scalar execution. The CPU strongly prefers contiguous wave-like propagation.

## 4. Tile 8x1 Branchless Compute
**Goal**: Simulate accumulate-and-mask logic without branches.
* **Throughput**: ~8.31 cycles per 256-bit operation.
* **Analysis**: Updating a state conditionally using bitwise masking (`state += input & mask`) works predictably without the massive 5-15 cycle stalls of branch misprediction. 

## 5. Packing (8-bit to 4-bit)
**Goal**: Measure the latency of shifting and compressing vectors.
* **Throughput**: ~2.15 cycles per 256-bit operation.
* **Analysis**: Very fast. The CPU handles shifts and masks almost frictionlessly, meaning that compressing states into 4-bit or 2-bit representations to fit more "history" into the L1 cache is highly practical.

## Conclusions for SiliconLLM Architecture
1. **Wave Propagation over Matmul**: The CPU can update hundreds of MBs of local state using cellular-style rules (`AND/XOR`) at multiple elements per clock cycle. A model based on local propagation rather than massive dense matrix multiplications leverages the CPU's true strength (ILP on contiguous streams).
2. **Butterfly Mixing**: For global-ish mixing, permutation networks (like Hadamard/Butterfly) are the optimal choice.
3. **Beware Naive SIMD**: Complex SIMD shuffles (like `vpermps`) can sometimes bottleneck the pipeline more than simple, heavily unrolled scalar operations. Vectorization must be aligned with the CPU's natural lane structures.
