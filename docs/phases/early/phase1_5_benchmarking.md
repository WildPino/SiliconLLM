# Phase 1.5 Benchmarking: Active Computation

After analyzing "passive memory" routing entropy in Phase 1, we proceeded to test "active computation" in Phase 1.5 to understand how to keep the CPU under optimal stress and exploit its Out-of-Order (OoO) and SIMD capabilities.

The benchmarks were run on an AMD Ryzen 5 3600X (Zen 2) with a background-free environment.

## 1. Dependency Chains (Serial vs OoO)
We tested the difference between a purely serial computational chain (where each operation depends on the previous one) and an independent set of chains that the CPU can execute out-of-order.
- **Serial Dependency Chain**: 9.85 cycles/op
- **OoO Independent Chains (4x)**: 2.43 cycles/op

**Conclusion**: Instruction-Level Parallelism (ILP) is crucial. A single sequential thread of logic leaves most ALU ports idle. By unrolling and keeping at least 4 independent accumulators/chains, we achieve a ~4x speedup.

## 2 & 3. SIMD Occupancy & Popcount Scalability
We evaluated bitwise operations using scalar 64-bit instructions versus 256-bit AVX2 vector instructions.
- **Scalar XNOR+Popcnt (64-bit)**: 7.20 cycles/op
- **AVX2 XOR+Add (256-bit)**: 1.37 cycles/op
- **AVX2 Add + Extract (to Scalar)**: 1.10 cycles/op

**Conclusion**: Native 64-bit popcount has noticeable latency when done sequentially. AVX2 handles 256-bit operations (equivalent to 4x 64-bit ops) in just over 1 cycle, making it vastly superior for bulk bitwise processing. Furthermore, extracting vector results back to scalar registers is heavily optimized (~1.1 cycles).

## 4. Tiny Tiles & Branching (Predicted vs Branchless)
We compared traditional `if/else` logic over unpredictable data versus a mathematical/bit-masking branchless approach.
- **Branching (if/else unpredictable)**: 9.15 cycles/op
- **Branchless (math/bit-masking)**: 3.95 cycles/op

**Conclusion**: Branch mispredictions are highly detrimental to the pipeline, causing stalls of ~5 extra cycles per operation compared to branchless logic. The model must rely heavily on bit-masking and arithmetic routing rather than conditional jumps.

## Overall Architecture Implications
1. Our minimum computing blocks (e.g., 2x2 / 4x4 tiles) must be evaluated using **AVX2 SIMD**, avoiding scalar popcounts where possible.
2. The sequence of operations must contain **independent streams** to fully saturate the CPU's superscalar ALU ports (OoO execution).
3. The routing and gating mechanisms (e.g., ReLU or routing decisions) must be strictly **branchless**.
