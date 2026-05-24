# SiliconLLM — Complete Research Memory

## Your Role (The Architect)

You are the **Architect** of the SiliconLLM project. Your mission is to design and build a **CPU-native language model** — not by porting a GPU Transformer to the CPU, but by discovering what computational form emerges naturally from modern silicon.

The guiding principle of this project is:

> **"Tirare fuori dal silicio un LLM, non imporre al silicio un LLM."**
> ("Extract an LLM from the silicon, don't impose an LLM onto the silicon.")

This means: every architectural decision must be backed by measured hardware data. You do NOT assume. You benchmark. You do NOT import paradigms from GPU-land. You discover what the CPU wants to do, and you build the model in that shape.

**Target hardware**: AMD Ryzen 5 3600X (Zen 2 architecture).
- 6 Cores / 12 Threads
- L1 Data Cache: 32 KB per core
- L2 Cache: 512 KB per core
- L3 Cache: 32 MB shared
- SIMD: AVX2 (256-bit width), NO AVX-512
- Compiler: GCC with `-O3 -march=native`

**Language**: Pure C with AVX2 intrinsics. No external libraries. No frameworks.

---

## Repository Structure

```
d:\_THINGS\Progetti\SiliconLLM\
├── DOCS/                          # All experimental results and analysis
│   ├── phase1_benchmarking.md     # Phase 1 results (passive memory)
│   ├── phase1_5_benchmarking.md   # Phase 1.5 results (active compute)
│   ├── phase2_primitives.md       # Phase 2 results (cognitive primitives)
│   ├── phase3_wave_engine.md      # Phase 3 results (first wave prototype)
│   ├── phase4a_memory.md          # Phase 4A results (memory mechanisms)
│   ├── phase4b_readout.md         # Phase 4B results (readout mechanisms)
│   ├── phase4c_learning_l2.md     # Phase 4C results (perturbation learning)
│   ├── phase4c_learning_l4.md     # Phase 4C results (SIMD evolutionary)
│   ├── phase4rc_results.md        # Phase 4RC results (reservoir computing)
│   ├── architecture_decisions.md  # Current architectural decisions log
│   └── project_summary.md         # High-level project summary
├── RESEARCH/                      # Background research documents
│   ├── derivative_problem.md
│   ├── modern_CPU_architecture.md
│   └── modern_CPU_architecture_with_estimation.md
├── PLAN/
│   └── plan.md                    # Original 6-phase research roadmap
├── src/                           # Benchmark source code (historical)
│   ├── benchmark.c                # Phase 1 benchmark
│   ├── benchmark1_5.c             # Phase 1.5 benchmark
│   ├── benchmark2_primitives.c    # Phase 2 benchmark
│   ├── benchmark4_memory.c        # Phase 4A benchmark
│   ├── benchmark4_readout.c       # Phase 4B benchmark
│   └── wave_engine.c              # Phase 3 prototype
├── benchmark4_l2.c                # Phase 4C L2 learning benchmark
├── benchmark4_l4_simd.c           # Phase 4C L4 SIMD evolutionary benchmark
├── benchmark4_rc.c                # Phase 4RC reservoir computing benchmark
├── benchmark5_arithmetic_wave.c   # Phase 5 arithmetic wave + injection tests
├── test_harness.h                 # Shared test infrastructure (tasks, metrics)
├── bin/                           # Compiled executables (historical)
└── build.bat                      # Build script
```

---

## The Complete Research History

What follows is a chronological account of every significant experiment, discovery, failure, and decision in the project. Read it ALL before making any architectural decision.

---

## PHASE 1: Hardware Archaeology (Passive Memory)

**File**: [benchmark.c](file:///d:/_THINGS/Progetti/SiliconLLM/src/benchmark.c)
**Results**: [phase1_benchmarking.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase1_benchmarking.md)

### What We Did
We measured the raw throughput of the CPU for different memory access patterns using `__rdtsc()` cycle-counting. Tests:
1. Sequential memory read (contiguous array traversal)
2. Random memory read (pointer chasing / random indices)
3. Scalar XOR + Popcount
4. Tiny 2×2 matrix multiplications

### The Discovery: Routing Entropy

| Access Pattern | Cost |
|---------------|------|
| Sequential (L1) | ~2-3 cycles/element |
| Random (L3 miss) | **~56 cycles/element** |

**The lesson**: The CPU processes sequential/contiguous data at extreme speed (hardware prefetcher, L1 locality). Random access patterns cause pipeline stalls of ~14 nanoseconds per element. The performance gap is **~20x**.

**Architectural implication**: Any model we build MUST use contiguous, sequential, wave-like data access. Sparse attention, random KV-cache lookups, hash tables — all of these destroy the CPU pipeline. The model must be **streaming and local**.

---

## PHASE 1.5: Active Computation

**File**: [benchmark1_5.c](file:///d:/_THINGS/Progetti/SiliconLLM/src/benchmark1_5.c)
**Results**: [phase1_5_benchmarking.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase1_5_benchmarking.md)

### What We Did
Tested "active" compute patterns: dependency chains, SIMD occupancy, branch prediction.

### Key Results

| Test | Result | Lesson |
|------|--------|--------|
| Serial dependency chain | 9.85 cycles/op | Single-threaded serial logic wastes most ALU ports |
| 4× independent chains (OoO) | 2.43 cycles/op | **4x speedup from ILP** — keep 4+ independent streams |
| Scalar XNOR+Popcnt (64-bit) | 7.20 cycles/op | Not free — has latency |
| AVX2 XOR+Add (256-bit) | 1.37 cycles/op | **~5x faster than scalar** for bulk bitwise |
| Branching (unpredictable if/else) | 9.15 cycles/op | Branch misprediction is devastating |
| Branchless (bit-masking) | 3.95 cycles/op | **2.3x faster** — all logic must be branchless |

**Architectural implications**:
- Unroll loops. Keep 4+ independent data streams to saturate OoO execution.
- Use AVX2 for bulk operations, not scalar.
- NEVER use if/else for model logic. Use bitwise masks, `cmov`, saturating arithmetic.

---

## PHASE 2: CPU-Native Cognitive Primitives

**File**: [benchmark2_primitives.c](file:///d:/_THINGS/Progetti/SiliconLLM/src/benchmark2_primitives.c)
**Results**: [phase2_primitives.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase2_primitives.md)

### What We Did
Tested candidate operations for the model's core algebra.

### Key Results

| Primitive | Cost | Notes |
|-----------|------|-------|
| **Local cellular propagation** (`new[i] = (L&R)^C`) | **0.28 cycles/elem** | THE discovery. ~3.5 elements per clock cycle. |
| Hadamard/Butterfly mixing (4×256-bit) | 2.34 cycles/block | Cheap global mixing via permutations |
| AVX2 XNOR+POPCOUNT (vpshufb) | 9.60 cycles/op | Viable but not free on Zen 2 |
| Packing 8→4 bit | 2.15 cycles/vector | Compressing state is cheap |

### THE Discovery: Local Propagation Is Essentially Free

The CPU's superscalar ILP engine devours contiguous wave-like updates. A simple scalar loop computing `new[i] = (state[i-1] & state[i+1]) ^ state[i]` runs at **~3.5 elements per clock cycle**. This is faster than a naive SIMD version with complex permutations (which was 0.50 cycles/elem due to shuffle overhead).

**Architectural implication**: The model's core substrate should be a **1D cellular automaton / wave grid** using contiguous memory updates. This is what the silicon wants to do.

> [!IMPORTANT]
> The SIMD version with `vpermps` cross-lane shuffles was SLOWER than unrolled scalar. SIMD must align with the CPU's natural lane structure. Don't force complex permutations.

---

## PHASE 3: First Wave-Butterfly Prototype

**File**: [wave_engine.c](file:///d:/_THINGS/Progetti/SiliconLLM/src/wave_engine.c)
**Results**: [phase3_wave_engine.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase3_wave_engine.md)

### What We Built
A 1024-cell 1D grid with:
- 4 selectable boolean rules per zone (16 zones of 64 cells)
- Butterfly bridges every 64 cells for global mixing
- Left-edge input injection (write symbol into cells [0..7])
- Branchless accumulator readout (sum last 64 cells, compare to threshold)
- "Learning" via random rule mutation on error

### Results

| Metric | Value |
|--------|-------|
| Throughput | 5,466 cycles/tick (~730K ticks/sec) |
| XOR-2 accuracy | ~50% (random) |
| Period-7 accuracy | 85.7% (learned to always predict 0) |
| Echo-5 accuracy | ~50% (random) |

### Why It Failed

The "learning" was completely blind. Randomly mutating one of 4^16 ≈ 4 billion rule combinations provides zero gradient information. The fitness landscape of cellular automata rules is fractal — changing one rule doesn't smoothly improve the output; it catastrophically changes the entire wave dynamics.

> [!WARNING]
> **This failure taught us the most important lesson of the project**: the wave dynamics are too chaotic to train directly. Any attempt to train the rules (L2 perturbation, L4 evolutionary, L3 Hebbian, etc.) will fail because the landscape is non-convex and discontinuous. This realization led directly to the Reservoir Computing paradigm.

---

## PHASE 4.A: Memory Mechanisms

**File**: [benchmark4_memory.c](file:///d:/_THINGS/Progetti/SiliconLLM/src/benchmark4_memory.c)
**Results**: [phase4a_memory.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase4a_memory.md)

### What We Tested
Four mechanisms for state persistence, measured both in isolation AND interleaved with wave propagation:

| Mechanism | Isolated (cycles/elem) | Interleaved (cycles/elem) | Fidelity (Hamming @ 200 ticks) |
|-----------|----------------------|--------------------------|-------------------------------|
| **M4 — Circular Buffer** | **0.007** | — | **0 bits** (perfect) |
| M6 — Leaky Integrator | 0.093 | 0.480 | 135 bits (~50% = noise) |
| M3 — Shift Register | 0.096 | 0.507 | 0 bits (perfect) |
| M2 — Bistable Attractors | 0.124 | 0.623 | 61 bits (degraded) |

### Decisions

1. **M4 (Circular Buffer)** for long-term memory: perfect fidelity, essentially zero cost. It's just a pointer with modular indexing (`index & (SIZE-1)` = one AND instruction).
2. **M6 (Leaky Integrator)** for short-term traces: `cell = (cell >> 1) + input`. It's NOT memory — it's a **trend tracker**. The signal decays to noise in ~8 ticks. Use it for eligibility traces, not for storing tokens.
3. **M2 (Bistable) rejected**: too expensive, still degrades.
4. **M3 (Shift Register) rejected**: viable but M4 is cheaper and equally perfect.

> [!IMPORTANT]
> M6 (leaky integrator) has a fidelity of ~135 bits / 256 — that's nearly 50%, which is pure noise. Do NOT confuse M6 with a memory mechanism. It's a temporal low-pass filter.

### Problem Solved: Interleaved Throughput Test
We measured memory mechanisms INTERLEAVED with wave propagation, not just in isolation. This was critical because isolated benchmarks don't capture cache interference, pipeline contention, or memory bandwidth competition. The interleaved cost of M6 (0.48 cycles/elem vs 0.28 for bare wave) showed that the memory doesn't destroy the wave's ILP.

---

## PHASE 4.B: Readout Mechanisms

**File**: [benchmark4_readout.c](file:///d:/_THINGS/Progetti/SiliconLLM/src/benchmark4_readout.c)
**Results**: [phase4b_readout.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase4b_readout.md)

### What We Tested
Mechanisms to extract decisions from the wave state (64-cell readout zone):

| Mechanism | Cycles/op | Discriminative? |
|-----------|-----------|-----------------|
| R1 — Flat Sum | 6.518 | ❌ Fails (same sum for inverse patterns) |
| **R2 — Multi-Channel K=4** | **6.512** | ✅ Perfect separation |
| R2 — Multi-Channel K=8 | 6.634 | ✅ Perfect separation |
| R3 — Pattern Match (AVX2 XNOR+POPCOUNT) | 10.599 | ✅ But 52.3% blind to 1-byte flips |
| R4 — Hierarchical Diff Tree | 30.647 | ✅ Robust to noise |

### The ILP Miracle
R2 at K=4 and K=8 costs **exactly the same** as R1 (flat sum). How? Because the CPU's superscalar engine runs the 4-8 independent accumulators in parallel on different ALU ports. You get **spatial sensitivity for free** thanks to Instruction-Level Parallelism. This is the hardware telling us: "give me independent streams."

### R3 Is Mathematically Broken for Sensitivity
R3 (XNOR+POPCOUNT distance) failed 523/1000 times to detect a 1-byte flip in the pattern. Why? If byte `v` has Hamming distance `d` from template byte `t`, then the flipped byte `v ^ 0xFF` has distance `8 - d`. When `d = 4` (the most probable case for random data, P ≈ 27%), the distance doesn't change: `8 - 4 = 4`. The metric is structurally blind to these flips.

### Extended Capacity Test
We tested 1000 random 64-byte patterns. R2 (K=4, K=8) and R3 (16 templates) all had **zero collisions**. Channel capacity is enormous for our purposes.

### Decision
**R2 Multi-Channel (K=8 or higher)** is the winner. Free ILP, perfect sensitivity, massive capacity. Use K=32 for V1 (still free due to ILP).

---

## PHASE 4.C: Learning — L2 (Perturbation / Hill Climbing)

**File**: [benchmark4_l2.c](file:///d:/_THINGS/Progetti/SiliconLLM/benchmark4_l2.c)
**Results**: [phase4c_learning_l2.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase4c_learning_l2.md)

### Integrated Throughput Validation
Before testing learning, we measured the FULL integrated pipeline (wave + M4 + M6 + R2-K8):
- **12,011 cycles/tick** → ~330,000 ticks/second on one core.
- This is ~2x the bare wave (5,400 cycles). The memory and readout components add overhead but do not destroy the pipeline.

### L2 Learning Results

| Task | Accuracy | Result |
|------|----------|--------|
| XOR-2 | 50.4% | Random — complete failure |
| Period-7 | 85.7% | Learned to always predict 0 (baseline trick) |
| Echo-5 | 49.7% | Random — complete failure |

**Why**: 1D perturbation (wiggle one rule selector, measure improvement) is blind in a 4^16 space with a fractal fitness landscape. Every mutation destroys the entire wave interference pattern. There are no "partial solutions" that hill climbing can exploit.

---

## PHASE 4.C: Learning — L4 (SIMD Evolutionary)

**File**: [benchmark4_l4_simd.c](file:///d:/_THINGS/Progetti/SiliconLLM/benchmark4_l4_simd.c)
**Results**: [phase4c_learning_l4.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase4c_learning_l4.md)

### The SIMD Breakthrough (Throughput)
We transposed the memory layout to Structure-of-Arrays: `__m256i state[GRID_SIZE]`, where each `__m256i` holds cell `i` for all 32 engines. This allows one AVX2 instruction to process the same cell across all 32 engines simultaneously.

To handle per-engine rule selection in SIMD (each engine has its own rule selectors), we compute ALL 4 rules for every cell, then use `_mm256_blendv_epi8` (3 masked blends) to select the correct rule per lane.

| Grid Size | Cycles/tick (32 engines) | Cycles/engine | In Cache? |
|-----------|------------------------|---------------|-----------|
| 1024 cells | 37,540 | **1,173** | L2 (64 KB state) |
| 256 cells | 9,398 | **293** | ✅ L1 (8 KB state) |

At 256 cells: 293 cycles / (4 steps × 256 cells) = **~0.28 cycles/elem** — identical to the Phase 2 scalar baseline, despite computing 4 rules + 3 blends! The AVX2 vector units absorb ALL the extra arithmetic.

> [!IMPORTANT]
> **L1 vs L2 matters enormously**: 1024 cells (64 KB for state+new_state) spills into L2, costing 4x more per engine than 256 cells (16 KB total, fits in L1). This 4x penalty is partially offset by 32x parallelism, but the net throughput per engine is dramatically worse in L2.

### L4 Learning Results
Despite 32 parallel engines with evolutionary selection (clone top 8 over bottom 8, mutate):

| Task | Accuracy |
|------|----------|
| XOR-2 | 53.6% |
| Period-7 | 85.7% |
| Echo-5 | 53.9% |

**Total failure.** The problem is NOT computational power. The problem is that **discrete boolean CA rules create a fractal fitness landscape with no building blocks**. Switching one rule in one zone doesn't incrementally improve the output — it catastrophically changes the entire wave dynamics. Evolutionary search cannot find structure in this landscape.

### The Paradigm Shift
This failure, combined with the L2 failure, led to the most important realization of the project:

> **Stop trying to train the wave. The wave is a physical reservoir. Train only the readout.**

This is the Reservoir Computing / Echo State Network (ESN) paradigm. The chaotic wave dynamics are not a bug — they're a feature. The chaos projects the input into a high-dimensional nonlinear feature space. A simple linear readout (trained via Least Mean Squares) extracts the prediction.

---

## PHASE 4.RC: Reservoir Computing

**File**: [benchmark4_rc.c](file:///d:/_THINGS/Progetti/SiliconLLM/benchmark4_rc.c)
**Results**: [phase4rc_results.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/phase4rc_results.md)

### Problem 1: Echo State Property (ESP) — The Wave Ignores Input

**The problem**: For a reservoir to be trainable, its state must be determined by the INPUT history, not by its own autonomous dynamics. We tested this by running two reservoirs with identical input but a 1-bit difference in initial state, and measuring the Hamming distance over time.

**Result without damping**: The Hamming distance DIVERGED to ~700 bits and never converged. The wave's chaotic self-dynamics completely overwhelm the input signal. The reservoir is deaf.

**Solution**: Global exponential damping — `state[i] >>= 1` every tick. This halves the energy every tick, forcing the autonomous chaos to decay to zero in 8 ticks (since uint8_t has 8 bits). The reservoir's state becomes entirely determined by the recent input history.

**Result with `>>= 1` damping**: Hamming distance drops to **0 bits** in <50 ticks. ESP is perfect.

> [!WARNING]
> The `>>= 1` damping creates a hard constraint: the wave's "memory" of its own state is exactly **8 ticks** (one bit-position per tick). After 8 ticks, ALL information from the initial state (or any previous autonomous dynamics) is gone. The wave becomes effectively **stateless** — it's a pure function of the most recently injected inputs. All temporal memory comes from the M4 circular buffer, NOT from the wave itself.

### Problem 2: Bit-Plane Degeneracy — Why Boolean Rules Can't Do XOR

With boolean rules (`(L & R) ^ C`, `L ^ C ^ R`, etc.), the 8 bits of each `uint8_t` cell evolve **independently**. Bit 0 of cell[i] never interacts with bit 3 of cell[i] — boolean logic operates on each bit-plane separately. Combined with the `>>= 1` damping (which shifts bits right), the 8-bit cell degenerates into a **1-bit wave delayed 8 times** (a pure shift-register delay line). This produces no nonlinear cross-bit features.

A 1-bit delay line cannot provide features to linearly separate XOR, which is fundamentally nonlinear.

**Solution**: Replace boolean rules with **arithmetic rules** (saturating add, sub, average). Addition propagates carries across bit positions, creating genuine cross-bit mixing. See Phase 5 below.

### RC Benchmark Results (Boolean Rules)

| Task | RC-1 (K=8) | RC-2 (K=32) | RC-3 (K=8 + M4) |
|------|-----------|------------|----------------|
| XOR-2 | 50.0% | 60.0% | 50.4% |
| Period-7 | 85.7% | 85.7% | 85.7% |
| Echo-5 | 49.8% | 51.1% | **100.0%** |

**Echo-5 solved at 100%** with RC-3 (M4 skip connections to the readout). The LMS learns to route M4[t-5] directly to the output. This proves the LMS training algorithm works perfectly when the features are available.

**Period-7 at 85.7%** is structurally impossible to solve: without input, the >>1 damping silences the reservoir in 8 ticks. A silent reservoir can't count to 7. The 85.7% = 6/7 = always predicting 0 (the majority class).

---

## PHASE 5: Arithmetic Wave + Injection Topology

**File**: [benchmark5_arithmetic_wave.c](file:///d:/_THINGS/Progetti/SiliconLLM/benchmark5_arithmetic_wave.c)
**Results**: [architecture_decisions.md](file:///d:/_THINGS/Progetti/SiliconLLM/DOCS/architecture_decisions.md) (section "The Final Unresolved Tension")

### Arithmetic Rules: Cross-Bit Mixing

We replaced boolean rules with AVX2 saturating arithmetic:
```c
// Example arithmetic rules:
_mm256_adds_epu8(avg(L, R), subs(C, 128))   // average + ReLU offset
_mm256_adds_epu8(subs(L, C), R)              // difference + sum
_mm256_subs_epu8(adds(L, R), C)              // Laplacian (edge detector)
```

**Throughput**: ~370 cycles per virtual engine per tick — **identical to boolean rules**. `vpaddusb` and `vpsubusb` are single-cycle instructions on Zen 2, just like `vpand`/`vpxor`.

**ESP**: Still holds perfectly with >>1 damping.

### Problem 3: Damping Kills Non-Linearity

The `>>1` damping halves the signal every tick. An input of 255 becomes 127 after 1 tick, 63 after 2 ticks. The arithmetic non-linearities (saturation at 255, or `subs(C, 128)` which acts as ReLU) trigger ONLY when the signal is near the extremes (0 or 255). But the damping pulls all values toward ~0-127, far from these thresholds. Result: the arithmetic wave operates in a **strictly linear regime** where no saturations occur.

**A strictly linear reservoir cannot solve XOR** (which is linearly inseparable by definition).

XOR-2 accuracy with arithmetic rules + >>1 damping: **52.5%** (random).

### Failed Fix Attempts

| Attempt | ESP? | XOR-2 | Why it failed |
|---------|------|-------|---------------|
| Soft damping `(3*state + wave)/4` (25% decay) | ❌ (44 bits) | 50.4% | Decay too weak → ESP violated → reservoir deaf |
| Soft damping + Hardware ReLU `subs(state, 32)` | ✅ (0 bits) | 58.5% | ESP restored but signal still too diffuse spatially |

### Problem 4: The Input Injection Topology

Even with ESP fixed and arithmetic non-linearities present, the system couldn't learn XOR-2. Root cause analysis:

The input was injected as a **uniform block of 32 cells** at the left edge of the grid, all set to either 0x00 or 0xFF. Problems:
1. **No spatial gradient**: 32 identical cells = flat region, no wave dynamics inside the block.
2. **No temporal collision**: Input at t-1 was at cells [0..31]; input at t-2 had propagated to cells [32..63] but was already heavily dampened. By the time t-1 and t-2 signals physically met in the grid, the t-2 signal was too weak for the arithmetic saturation to trigger.

### The Solution: Multi-Temporal Interleaved Injection (T3)

Instead of injecting only the current input at the left edge, we use the M4 circular buffer to inject BOTH current AND historical tokens at distributed, interleaved positions throughout the grid:

```
Grid layout (T3 topology, spacing = 4):
Cell 0:  input(t)      ← current symbol, full intensity (255)
Cell 1:  wave cell      ← free to compute
Cell 2:  M4[t-2]       ← historical symbol, full intensity (255)
Cell 3:  wave cell      ← free to compute
Cell 4:  input(t)       ← repeated
Cell 5:  wave cell
Cell 6:  M4[t-1]        ← historical symbol, full intensity (255)
Cell 7:  wave cell
...repeating...
```

This guarantees that:
- Current and historical signals are at **full intensity** (255, direct from M4, no decay)
- They are **physically adjacent** (separated by 1-2 cells)
- The arithmetic wave update on the cell between them immediately computes `sat_add(255, 255) = 255` (saturated when both inputs are 1) or `sat_add(255, 0) = 255` (not saturated when only one is 1)
- This creates the **AND(a, b) cross-term** that the linear readout needs to reconstruct XOR

### Injection Topology Test Results

| Topology | XOR-2 Accuracy | Feature Variance |
|----------|---------------|------------------|
| T0 (Left-edge baseline) | 52.2% | 4,800 |
| T1 (Distributed current-only) | 65.4% | — |
| T2 (Sparse interleaved t, t-1, t-2) | 71.3% | — |
| **T3 (Dense interleaved, spacing=4)** | **74.4% (rising)** | **20,646** |

T3 exploded the feature variance by **5x** compared to baseline. The wave is now "alive" — creating rich, diverse, input-dependent features.

---

## CRITICAL OPEN PROBLEMS (Unsolved at End of Research Phase)

These are problems that were identified but NOT resolved. The Architect MUST address them.

### 🔴 OPEN PROBLEM 1: LMS Never Converged Above 80% on XOR-2

Despite the feature separability diagnostic showing deltas >100 between classes (confirming that the wave creates discriminative features), the LMS readout never achieved >78.5% accuracy on XOR-2. Three separate numerical issues were identified:

**Issue A: Feature scale mismatch.** Wave channels (sums of 32 cells) produce values ~4000-8000. M4 channels produce values 0-255. LMS weight updates are proportional to channel magnitude, so wave weights oscillate violently while M4 weights barely move.

**Issue B: Missing bias term.** Without a constant "bias" channel, the LMS decision hyperplane is forced to pass through the origin (0,0,...,0). But the data cloud is centered at (4000, 4000, ..., 128, ...), far from the origin. The weights waste all their capacity compensating for this offset instead of detecting features.

**Issue C: Weight clipping.** The test harness clipped weights to `MAX_W = 10000`. With wave channels at ~4000 and 32 channels, the dot-product sum reaches tens of millions. A bias weight of 10,000 * 255 = 2,550,000 is insufficient to compensate.

**Failed fix attempt**: Normalizing wave channels by dividing by 32 (`channel >>= 5`). This brought the scale to 0-255 (matching M4) but **killed the variance** (divided by 32² = 1024). Feature variance collapsed from 21,000 to ~20. The channels became a flat wall of ~127, indistinguishable from noise. Accuracy collapsed to **45%** (below random, due to weight instability).

**Lesson**: Dividing by N preserves the mean but divides the variance by N². Never normalize features by dividing — center them by subtracting (which preserves variance).

**Recommended solutions for V1** (untested):
1. Add a bias channel: `channel[48] = 255` (constant). The LMS learns `w_bias * 255` to absorb the mean offset. One extra weight.
2. Mean-center channels: `wave_channel[k] -= expected_mean` (e.g., subtract 32*127 = 4064). Preserves variance, removes offset.
3. Use **Ridge Regression** (offline) instead of LMS (online). Ridge regression computes optimal weights in one shot: `w = (X^T X + λI)^{-1} X^T y`. Requires inverting a 49×49 matrix — trivial. Immune to scale, bias, and learning rate issues.
4. Use **NLMS** (Normalized LMS) for online adaptation: `w[k] += error * channel[k] / (||channel||² + ε)`. Automatically handles scale differences.

### 🔴 OPEN PROBLEM 2: XOR-2 Test Is Methodologically Flawed

The XOR-2 task generates a deterministic sequence: `seq[t] = seq[t-1] ^ seq[t-2]`. With fixed initial conditions, this sequence is **periodic with period 3** (e.g., 0, 1, 1, 0, 1, 1, ...).

The M4 buffer stores perfect history. With 3+ slots of history, M4 can memorize the periodic pattern WITHOUT computing XOR. The system may be learning to recognize its position in the cycle (a linear/lookup task), not computing the nonlinear XOR function.

**Evidence**: The feature separability diagnostic showed M4 channels with delta >100 for XOR-2. But for TRUE XOR on i.i.d. random inputs, the M4 channel deltas should be ~0 (because XOR is symmetric: the mean of M4[t-1] is 127.5 for both class 0 and class 1). Large M4 deltas indicate the system is exploiting the periodicity, not computing XOR.

**Implication**: We do NOT have definitive proof that the arithmetic wave creates genuinely nonlinear features. The wave MIGHT be contributing, or the M4 might be doing all the work via periodicity exploitation.

**How to test properly**: Run XOR on two **independent random bit streams** (not a self-generated sequence). Feed bit `a` at time t and bit `b` at time t-1 (from a separate random generator). Target = `a XOR b`. This eliminates periodicity and forces genuine nonlinear computation.

**However**: For V1 (real text prediction), input IS autocorrelated. Exploiting temporal patterns via M4 IS useful. The wave's value may lie in spatial mixing (adding capacity) rather than strict nonlinear computation. The definitive test of the architecture is **performance on real text**, not performance on synthetic XOR.

### 🟡 OPEN PROBLEM 3: The Wave Is Effectively Stateless

With `>>= 1` damping, the wave forgets its own state in 8 ticks. Combined with T3 injection (which overwrites injection cells every tick), the wave state at time t is a **deterministic function of the last ~2-3 injected values** (since the wave only runs for 4 ticks per macro-step before readout).

Mathematically: `features(t) = Φ(input(t), M4[t-1], M4[t-2])` — a fixed nonlinear function.

This means the wave is a **stateless nonlinear feature extractor** (a kernel / random feature map), not a dynamical system with memory. All temporal memory comes from M4 injection, not from the wave itself.

**Context window** = number of M4 slots injected into the grid. With T3 (spacing 4) on a 256-cell grid: 256/4 = **64 tokens** maximum. On 1024 cells (L2): 256 tokens.

For language modeling, this may be sufficient for local patterns but insufficient for long-range dependencies.

### 🟡 OPEN PROBLEM 4: Only Binary Inputs Tested

All experiments used binary inputs (0x00 or 0xFF). Real text is byte-level (0-255). The saturation-based nonlinearity (`sat_add(255, 255) = 255` vs `sat_add(255, 0) = 255`) works cleanly for binary inputs. With arbitrary byte values, the saturation behavior is more complex and untested.

### 🟡 OPEN PROBLEM 5: Multi-Class Output

All experiments used binary output (predict 0 or 1). Real text prediction requires 256-class output (next byte). Options:
- 256 separate readout heads (one per class, each with ~49 weights → 12,544 total)
- 8 binary readout heads (one per bit of the output byte → 392 weights, but loses inter-bit correlation)

Neither has been tested.

---

## CONFIRMED ARCHITECTURAL DECISIONS (Backed by Data)

These are PROVEN by benchmarks and should NOT be changed without new data:

| # | Decision | Measured Cost | Supporting Data |
|---|----------|--------------|-----------------|
| 1 | Wave substrate: AVX2 saturating arithmetic on contiguous 1D grid | 0.28 cycles/elem (L1) | Phase 2, Phase 4C-L4 |
| 2 | Memory (long-term): M4 Circular Buffer | 0.007 cycles/elem | Phase 4A |
| 3 | Memory (short-term): M6 Leaky Integrator | 0.093 cycles/elem | Phase 4A |
| 4 | Readout: R2 Multi-Channel (K=32) | ~6.5 cycles (free via ILP) | Phase 4B |
| 5 | Mixing: Butterfly/Hadamard bridges | 2.34 cycles/block | Phase 2 |
| 6 | Paradigm: Reservoir Computing (ESN) — fixed wave, trained readout | — | Phase 4C (L2, L4 failures) |
| 7 | ESP: enforced via `>>= 1` global damping | 0 extra cycles (merged with wave) | Phase 4RC |
| 8 | Injection: T3 multi-temporal interleaved from M4 | Same as left-edge | Phase 5 |
| 9 | All logic must be branchless | — | Phase 1.5 |
| 10 | Grid in L1 (256 cells) is 4x faster than L2 (1024 cells) | 293 vs 1173 cycles/engine | Phase 4C-L4 |

---

## THE PATH TO V1

V1 is a **byte-level next-character predictor**. NOT a language model (yet). The goal is to demonstrate that the wave-reservoir architecture can achieve non-trivial perplexity on real text data.

### V1 Minimum Requirements

1. **Input**: byte (0-255), not binary
2. **Grid**: 256 cells (L1) initially, scale to 1024 (L2) if needed
3. **Injection**: T3 multi-temporal from M4 (64 token context at 256 cells)
4. **Output**: 256-class (start with 8×binary heads, upgrade to 256 heads if needed)
5. **Readout weights**: int32_t, WITH bias channel, NO artificial clipping
6. **Training**: Offline Ridge Regression for initial weights (optimal, immune to numerical issues), then NLMS for online adaptation
7. **Test data**: Real text file (Shakespeare, source code, etc.)
8. **Metric**: Bits-per-byte (lower = better; random baseline = 8.0 bits/byte)

### Critical Questions for V1 Design

1. How to encode the 256 possible byte values for injection? (Currently injecting 0x00 or 0xFF. For bytes, inject the raw value? Or one-hot encode into 8 cells?)
2. How does the readout architecture change for 256 classes? (256 separate weight vectors? Softmax? Or 8 independent binary predictions for each bit of the output byte?)
3. How to handle the context window limit (64 tokens at 256 cells)? Is this enough for meaningful text prediction?

---

## APPENDIX: Lessons Learned (Traps to Avoid)

1. **Never normalize features by dividing.** Dividing by N preserves the mean but divides variance by N². Use mean-subtraction instead.
2. **Never assume — always benchmark.** We nearly built V1 on three separate unverified assumptions (arithmetic rules fix XOR, soft damping enables nonlinearity, feature normalization fixes LMS). All three were wrong. The 5-minute benchmark saved weeks each time.
3. **LMS is fragile.** It's sensitive to feature scale, missing bias, learning rate, and weight clipping. For definitive evaluation, use offline ridge regression first to establish the accuracy ceiling.
4. **SIMD is not always faster.** Complex cross-lane shuffles (`vpermps`) can be slower than unrolled scalar on contiguous data (Phase 2). Use SIMD for parallelism across independent streams (like 32 engines), not for complex permutations within a stream.
5. **The CPU loves waves.** Contiguous sequential updates at 0.28 cycles/elem. Random access at 56 cycles/elem. The model must be wave-shaped.
6. **Discrete CA rules are untrainable.** The fitness landscape is fractal. Don't try gradient descent, hill climbing, or evolutionary search on CA rule selectors. Use the rules as a fixed reservoir and train only the readout.
7. **Damping creates a tradeoff.** Strong damping (>>1) → good ESP, but kills all wave memory and prevents nonlinear saturation. Weak damping → wave has memory and nonlinearity, but ESP fails. The solution is to enforce damping AND inject historical values at full intensity via M4.
8. **The wave with >>1 damping is stateless.** It's a fixed nonlinear function of the injected inputs, not a dynamical system with memory. This is fine architecturally (it's a kernel/feature map), but the designer must understand this.
9. **Self-referential sequences (like XOR-2 = seq[t-1] ^ seq[t-2]) are periodic and contaminate benchmarks.** Use truly random i.i.d. inputs for nonlinearity testing.
