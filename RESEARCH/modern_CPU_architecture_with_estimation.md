# Learning Architectures Designed for Modern CPUs

## Towards 1-bit and 2x2 models built around clock cycles, throughput, and execution unit granularity

### Abstract

This paper proposes a radical reformulation of the neural model design problem, starting not from the architecture of an LLM, but from the operations that can actually be executed efficiently by modern CPUs. The starting point is not the massive parallelization typical of GPUs, but rather the internal structure of the CPU: pipeline, decode, micro-op cache, branch prediction, load/store, SIMD, FMA, bit-manipulation, vector multiplications, fused instructions, and in some architectures, specialized matrix extensions.

The central idea is simple but powerful: if the model is designed to naturally "fall" onto the functional units of the CPU, the bottleneck is no longer backpropagation as an abstract concept, but the alignment between the model's algebra and the per-cycle capabilities of the processor. From this perspective, a 1-bit or 2x2 block network is not a quantization curiosity, but a potential native form of computation.

The purpose of this paper is to build a mathematical language for this approach: first describing the machine, then deriving the model's structure.

---

## 1. Introduction

The most common mistake in designing models for CPUs is treating the CPU as a slower GPU. This is not the case. The CPU is a different machine, optimized for:

* Low single-thread latency
* Fine-grained flow control
* Branch prediction
* Out-of-order execution
* Complex and fused instructions
* Memory accesses with a strong cache hierarchy
* Moderate but extremely efficient SIMD
* Selective vectorization
* High-density bitwise and integer operations
* Some specialized extensions for matrices or blocks

A GPU, by contrast, takes advantage of a massive amount of threads and massive throughput on uniform operations. The CPU does not compete on the same level, but it can be leveraged much better if the problem is formulated in its natural space.

The question therefore is not: *"How do I run an LLM on a CPU?"*
The right question is: *"What shape must a model take so that the CPU executes it with high occupancy of its functional units per clock cycle?"*

This changes everything.

---

## 2. The CPU as a Cycle-by-Cycle Transformation System

### 2.1 Clock Cycle, Throughput, and Latency

A CPU does not execute "one operation at a time" in a naive sense. Each clock cycle passes through multiple stages: fetch, decode, rename, dispatch, issue, execute, retire.

For model design, two quantities are relevant:

* **Latency:** how many cycles are needed before the result of a dependency is available.
* **Throughput:** how many independent operations can be sustained per cycle.

The key point is that a modern CPU can have many parallel units, but only if the dependency graph allows it. Therefore, the model must not only be "small"; it must be decomposed well.

We define the useful throughput of an operation as:


$$T_{\mathrm{eff}} = \frac{\text{useful operations completed}}{\text{clock cycles}}$$

The goal is not to maximize per-token complexity, but to maximize the density of useful work per cycle with minimal dependencies.

### 2.2 The CPU as a Graph of Functional Units

We can model the CPU as a bipartite graph between:

* Issued instructions
* Available functional units

Each instruction is a vector of requirements:


$$I_k = (a_k, b_k, m_k, s_k, v_k, f_k)$$

Where, for example:

* $a_k$: Integer arithmetic type
* $b_k$: Bitwise operation
* $m_k$: Memory access
* $s_k$: Shift/rotate
* $v_k$: SIMD vector operation
* $f_k$: Fused multiply-add or composite operation

An efficient core is one that can map the highest possible number of instructions onto different units in the same time interval.

---

## 3. Operations Modern CPUs Do Well

This section does not list "every possible ISA feature," but rather the conceptual building blocks that truly matter for model design.

### 3.1 Scalar Integer Operations

Modern CPUs excel at:

* Addition and subtraction
* Integer multiplication
* Comparisons
* Min/max
* Saturating arithmetic (on some ISAs)
* Logical and arithmetic shifts
* Rotations
* Bit test, mask, set, clear
* Popcount, leading/trailing zero count
* Bit scan
* Conditional branches and setcc-like operations

For a 1-bit or 2-bit model, this is crucial: bitwise is not a limitation; it is a primary resource.

### 3.2 Vector SIMD

Modern CPUs support wide register vectors with various families of SIMD instructions. The principle is that a single operation acts on multiple elements in parallel.

Mathematically, instead of:


$$y_i = f(x_i)$$

You execute:


$$\mathbf{y} = f(\mathbf{x})$$

over blocks of components. This makes the following structures natural:

* Block matrices
* Grouped tensors
* Convolutions or blockwise products
* Group-applied activations
* Vectorial accumulation of errors or scores

### 3.3 FMA and Fused Operations

Fused multiply-add operations are central:


$$z = a \cdot b + c$$

in a single conceptual instruction. The fact that multiplication and addition are fused is incredibly important. A well-constructed model should produce many expressions reducible to this form because FMA reduces temporary variable traffic and better exploits datapaths.

### 3.4 Load/Store and Cache Hierarchy

No CPU model can ignore memory. The cost is often not the computation itself, but data movement. Modern CPUs have:

* Registers
* L1 cache
* L2 cache
* Often a shared L3 cache
* Hardware prefetchers
* Out-of-order load/store

An efficient model must minimize erratic accesses and prioritize:

* Linear streaming
* Local reuse
* Small, repeated blocks
* A cache-compatible working set

### 3.5 Branch Prediction and Flow Control

The CPU can execute branches well, but only if they are predictable. A model with too many divergent conditions or chaotic routing degrades throughput. Flow control must be made:

* Regular
* Stable
* Locally predictable
* Ideally replaced by masks and arithmetic selections

### 3.6 Specialized Matrix Extensions

Some modern CPUs also feature extensions closer to block multiplications, including forms of matrix instructions or tile-based execution. This suggests an important direction: the model doesn't necessarily have to be vectorial in the classical sense; it can be **tile-algebraic**.

Instead of thinking of a weight as a single scalar, it can be viewed as a 2x2 block, a 4x4 block, or a small tile perfectly suited to the execution unit.

---

## 4. The Problem Isn't Just the Model: It's the Computation Unit

Most neural architectures are designed around the wrong granularity. Modules are defined in terms of "layers," but the CPU reasons in terms of:

* Instructions
* Micro-operations
* Port mapping
* Dependencies
* Hidden latency
* Pipeline filling

Therefore, the true object of design is the **computation unit compatible with the clock cycle**.

We define a computational block $B$ such that it:

* Can be executed with minimal dependencies
* Well-occupies one or more functional units
* Produces enough useful work per cycle
* Requires local and regular memory
* Repeats many times with a similar structure

This definition is far more useful than the classical concept of a layer.

---

## 5. 1-bit and 2x2 Models as Native CPU Forms

### 5.1 Why 1-bit?

If the system is designed for bitwise logic, the CPU is already in its natural domain. A binary weight is a discrete variable:


$$w_i \in \{-1, +1\}$$

or in boolean form:


$$w_i \in \{0, 1\}$$

This enables:

* Masks
* AND/OR/XOR
* Popcount
* Bit packing
* Fast comparisons
* Accumulation via sums of discrete contributions

Multiplying by a bit can be reinterpreted as a selection or a negation, which is often much cheaper than floating-point multiplication.

### 5.2 Why 2x2?

A 2x2 matrix is the first non-trivial object where the following emerge:

* Composition
* Linear transformation
* Local rotation
* 2D mixing
* Accumulation of minimal correlations

The fact that many CPUs can handle small vector blocks well suggests a hierarchical architecture made of micro-matrices, not massive dense tensors.

Instead of a large linear projection:


$$Y = XW$$

You build a system of blocks:


$$Y_{ab} = \sum_{cd} X_{ac} W_{cd}$$

with $W$ decomposed into small, regular tiles.

---

## 6. Throughput Theory Per Model

### 6.1 Definition of Algorithmic Throughput

Let $C$ be the number of available clock cycles and $U$ the number of useful operations completed. The algorithmic throughput is:


$$\tau = \frac{U}{C}$$

The model is good if it maximizes $\tau$ while maintaining adequate accuracy or expressive capacity. However, this is not enough: hidden memory and control costs must also be penalized.

Thus, we define:


$$\tau_{\mathrm{net}} = \frac{U}{C + \alpha M + \beta B + \gamma D}$$

Where:

* $M$ is memory cost
* $B$ is branch/divergence cost
* $D$ is dependency and stall cost
* $\alpha, \beta, \gamma$ are architectural weights

The ideal architecture minimizes the penalizing terms.

### 6.2 Maximizing Work Per Cycle

A single cycle should execute the highest possible number of independent micro-operations. To achieve this, the model must be decomposable into a set of parallel instructions:


$$\mathcal{I}(t) = \{i_1, i_2, \dots, i_k\}$$

with a low degree of interference. The structure must therefore be:

* Locally dense
* Globally regular
* Arithmetic-friendly
* Cache-friendly
* Branch-light

---

## 7. New Paradigm: From Layer to "Clock Cell"

I propose the concept of the **clock cell**: a computational cell designed to be executed within a clock window or a small number of cycles with high occupancy of available units.

A clock cell should:

* Receive small, contiguous inputs
* Produce compressible output
* Use CPU primitive operations
* Avoid long dependencies
* Be replicable in blocks

An entire network is thus a network of clock cells, not a stack of abstract layers. This aligns much closer to the logic of a real CPU.

---

## 8. Compositional Operations Suited for a CPU-Native Model

### 8.1 XOR-like and Mask-Based Operations

For binary models, the most natural transformations are often similar to:

* **XOR** for state composition
* **AND** for gating
* **OR** for fusion
* **NOT** for inversion
* **Popcount** for similarity measurement

The semantic representation can be built as a combination of these primitives. For example, a binary similarity between two vectors can be seen as:


$$s(x,y) = \mathrm{popcount}(\mathrm{XNOR}(x,y))$$

or an equivalent form.

### 8.2 2x2 Block Operations

A 2x2 transformation can represent a local mixing cell:

$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

If $a, b, c, d$ are binary or quasi-binary, the execution cost can be kept extremely low, and the network is composed of many local micro-mixers instead of a few massive multipliers.

### 8.3 Accumulative Operations

Many CPU-friendly structures are accumulative:


$$z_{t+1} = z_t + \Delta_t$$

This is perfect for:

* Progressive scoring
* Voting
* Binary counts
* Evidence integration
* Short-term memory logic

A model that uses local accumulation instead of fully dense transformations can better exploit the scalar data path.

---

## 9. Mathematical Structure of the Proposed Model

### 9.1 State

The internal state is not a gigantic float vector, but a set of discrete variables and small, continuous auxiliary registers:


$$S = (b, u, a, m)$$

Where:

* $b$: Binary state
* $u$: Continuous support latents
* $a$: Accumulators
* $m$: Short-term memory or discrete momentum

### 9.2 Evolution

The evolution per layer/cell is:


$$S_{t+1} = \Phi(S_t, x_t)$$

with $\Phi$ designed to be implementable using CPU primitives. The function $\Phi$ does not necessarily need to be smooth; it just needs to execute well.

### 9.3 Update Decision

A cell can make decisions using a threshold rule:


$$b_i^{t+1} = \operatorname{sign}(\mu_i^t + \rho_i^t)$$

Where:

* $\mu_i^t$ is a signal accumulator
* $\rho_i^t$ is a local corrective term

This structure is highly compatible with integer arithmetic, bitwise logic, and small comparisons.

---

## 10. Adapting LLM Logic to this Machine

Herein lies the most important point: you must not take a standard transformer and hope it "adapts" to the CPU. You must rebuild it from the ground up.

Classical LLM structures use large weights, dense matrices, costly softmax operations, global attention, and heavy floating-point math. Conversely, a CPU-native architecture should favor:

* Local competition instead of global softmax
* Routing via blocks instead of dense attention
* Compressed memory instead of enormous states
* Discrete updates instead of continuous transformations everywhere
* 1-bit/2x2 compositions instead of massive GEMMs

In practice, the semantics of the model must be built from many small operators, each perfectly aligned with the clock cycle.

---

## 11. A Structural Proposal: The Discrete Tile Network

Imagine a network composed of tiles:


$$T_{ij} \in \{-1, +1\}^{2\times2}$$

Each tile operates on local micro-states and produces a compressed output. The layer becomes a grid of tiles featuring:

* Local connectivity
* Sum of contributions
* Binary gating
* Context accumulation

This makes the following possible:

* High data-flow regularity
* Low memory pressure
* Strong prefetchability
* Intensive use of SIMD and bit tricks
* Potential mapping to future matrix extensions

---

## 12. Design Metrics for the CPU-Native Network

Such an architecture should be evaluated using different metrics than standard ones.

### 12.1 IPC and Useful Work

$$\mathrm{IPC} = \frac{\text{retired instructions}}{\text{cycles}}$$

However, IPC alone is not enough: we need *useful* instructions, not just *many* instructions.

### 12.2 Arithmetic Intensity vs. Memory Traffic

$$I = \frac{\text{arithmetic operations}}{\text{bytes transferred}}$$

The higher $I$ is, the more the model leans toward being compute-bound rather than memory-bound.

### 12.3 Flow Regularity

A CPU-native network must have low structural entropy in its execution paths. In other words:

* Fewer unpredictable branches
* Fewer sparse accesses
* Fewer long dependencies
* More structured repetition

---

## 13. Strong Hypothesis: The CPU as a Discrete Model Accelerator

The most radical thesis is this:
**The CPU is not just a compromise for running large models; it can become a natural accelerator for models built on discrete algebra, 2x2 blocks, bitwise logic, accumulators, and regular micro-routines.**

In this vision, a well-designed model could leverage the following much better than currently assumed:

* Internal per-cycle parallelism
* Flow prediction
* The ability to perform multiple micro-operations per architectural instruction
* Cache locality
* Bit packing
* Processing of small blocks

The comparison with the GPU then shifts: it is no longer about competing with massive throughput, but about using a model where the per-token cost drops drastically because the computation is already compact and hardware-adjacent.

---

## 14. Conclusion

If we start from pure mathematics, the question is not how to "shrink" an LLM for a CPU.
The question is: *What model theory emerges directly from a modern CPU?*

The answer proposed by this paper is:

* Use small discrete units, 1-bit or 2x2.
* Prioritize operations the CPU can issue and sustain well per cycle.
* Maximize useful throughput, not just model size.
* Reduce dependencies, unpredictable branches, and memory traffic.
* Structure the network as a collection of clock cells and local tiles.
* Replace the grammar of the classical transformer with a grammar of discrete algebra, accumulation, and regular blocks.

The final perspective is clear: the CPU must not chase the GPU. It must impose its own shape onto the model.

---

## Appendix A — Design Outline

A possible conceptual pipeline is:

1. Discrete token or input
2. Compact binary encoding
3. Propagation via 2x2 tiles
4. Local accumulation
5. Threshold-based decision
6. Block compression/expansion
7. Discrete short-term memory
8. Final output via voting or scoring

## Appendix B — Open Research Questions

* What is the optimal tile granularity to maximize effective IPC?
* Can a 1-bit model maintain enough expressivity using only accumulators and 2x2 blocks?
* How much do branch prediction and cache locality matter compared to the complexity of a single operator?
* Is it possible to define a training theory directly on micro-operations rather than continuous gradients?
* What compromise between discretion and stability produces the best per-token throughput?

## Appendix C — Central Thesis in a Single Sentence

The optimal shape of a CPU model is not a miniaturized version of a GPU-style transformer, but a discrete system built to firmly occupy the clock cycle, the cache, and the functional units of the CPU.

## Appendix D — Numerical Case Study: AMD Ryzen 7 9700X

This section takes a real consumer model and uses it as a test bench to understand where the CPU excels and where it breaks down.

### D.1 Basic Architectural Parameters

Let's take a Ryzen 7 9700X as a reference:

* 8 cores / 16 threads
* Boost up to 5.5 GHz
* DDR5 up to 5600 MT/s over 2 channels
* Total L2: 8 MB
* Total L3: 32 MB

In its Zen 5 form, architectural documentation indicates for each core:

* 32 KB of L1 I-cache
* 32 KB of L1 D-cache
* 1 MB private L2
* 64 B cache line

### D.2 Conversion to Clock Cycles

At 5.5 GHz, one cycle lasts:


$$T_{\text{cycle}} = \frac{1}{5.5 \text{ GHz}} \approx 0.1818 \text{ ns}$$

Thus, using latencies measured in public benchmarks for the 9700X:

* **L1 latency** $\approx 0.7$ ns $\rightarrow$ about 3.85 cycles
* **L2 latency** $\approx 2.5$ ns $\rightarrow$ about 13.75 cycles
* **L3 latency** $\approx 10.6$ ns $\rightarrow$ about 58.3 cycles
* **RAM latency** $\approx 63$ ns $\rightarrow$ about 346.5 cycles

This is the crucial point: there is a jump of nearly two orders of magnitude between L1 and RAM.

### D.3 How Many Cache Lines Each Level Holds

With a 64 B cache line:

* 32 KB L1 D-cache holds 512 lines
* 1 MB L2 holds 16,384 lines
* 32 MB L3 holds 524,288 lines

The strategic takeaway is this: if your model can stay within L1 or L2, you are playing in an entirely different league compared to spilling over into L3 or RAM.

### D.4 Theoretical Memory Bandwidth

With dual-channel DDR5-5600, the maximum theoretical bandwidth is:


$$5600 \text{ MT/s} \times 8 \text{ B} \times 2 = 89.6 \text{ GB/s}$$

But bandwidth should not be confused with latency. RAM can be fast in bandwidth yet slow to respond. A cache is often useful precisely because it reduces waiting time, not just because of bandwidth.

### D.5 What Happens When Data Misses L1

If an access goes to RAM, the core can stall for roughly 346 cycles.
If the code executes one useful operation per cycle, you need about 346 independent operations of useful work to hide a single miss.
If instead, the model uses an 8-way SIMD, the number of micro-ops or elementary blocks needed to cover the wait drops by about a factor of 8.

This is one of the reasons why CPU-native models must be built to:

* Maximize independence between micro-operations
* Compress the working set
* Avoid sparse accesses
* Favor local reuse
* Use regular and repeatable blocks

### D.6 Branch Prediction: The Cost of Guessing Wrong

Modern CPUs don't wait passively: they try to guess the next control flow using BTBs, history, and return stacks. AMD documentation describes the BTB as a structure read at the beginning of the pipeline while instruction bytes are fetched, and AMD architectural materials show that the predictor is a combination of BTB, return address stack, and indirect predictor. The pipeline dislikes unpredictable branches.

Public measurements on the Zen family indicate that a branch miss typically costs cycles in the tens; a published example for Zen 2 measures roughly 31.7 cycles in a microbenchmark with one mispredicted branch per iteration.
At 5.5 GHz, 30 cycles are roughly 5.45 ns.
Therefore, a bad branch costs more than a small, well-placed chain of arithmetic operations.

### D.7 Implication for Model Structure

From these numbers, a very strict rule of thumb emerges:

* If a transformation enters and leaves L1, you can afford very tight loops and regular logic.
* If the structure lives in L2, you can still perform well, but you must reduce dependencies and increase reuse.
* If you end up in L3, the model must have orderly and prefetchable accesses.
* If you go to RAM, you have to compensate with a massive amount of independent work or very large, repetitive blocks.
* If you introduce random branches, prediction becomes a secondary optimization problem.

This shifts design toward operators the CPU digests well:

* Sums and accumulators
* Branchless comparisons and selections
* Bitwise masking
* Popcount/XNOR for discrete matching
* Small, regular matrix blocks
* FMA or equivalent forms in a compact kernel
* Tiles that sit comfortably in cache

### D.8 Concrete Design Rule

For a CPU-native model, the target is not "a huge matrix everywhere."
The target is:

* Blocks small enough to fit in L1/L2
* Regular enough to be prefetched
* Dense enough to occupy multiple units per cycle
* Discrete enough to allow bitwise and integer-friendly compute
* Compositional enough to build global semantics out of many local cells

In practical terms, this means the true unit of design is not the layer, but the repeatable kernel that the CPU can keep warm in its pipeline.