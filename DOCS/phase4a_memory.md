# Phase 4.A: Memory Mechanisms

We tested four memory mechanisms to understand how state can persist in the SiliconLLM architecture without bottlenecking the CPU's Instruction-Level Parallelism (ILP). The tests measured both pure isolated cycle cost and the cost when interleaved with the baseline wave propagation. We also tested the physical persistence of a 32-byte pattern over T ticks.

## Benchmark Results (Ryzen 5 3600X)

### Cycle Costs (Throughput)
- **Baseline Wave (Quasi-Rule 30)**: `0.343` cycles/elem
- **[M4] Circular Buffer**: `0.007` cycles/elem (Isolated) — *The absolute winner. Writing to a modulo index is virtually free.*
- **[M6] Leaky Integrator**: `0.093` isolated / `0.480` interleaved — *Extremely fast, adding only ~0.14 cycles of overhead to the wave. Very viable for short-term trace decay.*
- **[M3] Shift Register**: `0.096` isolated / `0.507` interleaved — *`memmove` is highly optimized but starts to bottleneck the ILP slightly more than simple math.*
- **[M2] Bistable Attractors**: `0.124` isolated / `0.623` interleaved — *The branchless mask/latch logic introduces the heaviest overhead, nearly doubling the wave cost.*

### Fidelity / Persistence (Hamming Distance / 256 bits)
- **[M3] Shift Register & [M4] Circular Buffer**: **0 bits distance** after 50, 100, and 200 ticks. Perfect mathematical persistence. The data survives identically until overwritten.
- **[M2] Bistable Cells**: Settles at **~61 bits distance** regardless of T. The attractor captures the pattern loosely but allows degradation due to wave noise.
- **[M6] Leaky Integrator**: Settles at **~135 bits distance**. The `>> 1` decay mechanism quickly washes out the initial pattern without a sustaining input. It is true "short-term" memory.

## Architectural Implications
1. **Explicit Storage is Free**: If we want long-term perfect persistence, a Circular Buffer (M4) is the way to go. It costs essentially 0 cycles and guarantees perfect reconstruction.
2. **Biological Decay is Cheap**: The Leaky Integrator (M6) merges seamlessly with the wave propagation loop, costing just ~0.14 extra cycles/elem. This is excellent for short-term memory (eligibility traces).
3. **Avoid Latching**: Branchless bistable latches (M2) introduce too much masking overhead and still degrade.

**Conclusion**: The optimal memory architecture pairs the dissipative wave with an explicit **M4 Circular Buffer** for long-term discrete token/history storage, and uses **M6 Leaky Integrators** locally for short-term continuous state tracking.
